import logging
import torch
from torch import nn
import os
import torch.nn.functional as F

from fastreid.utils.checkpoint import get_unexpected_parameters_message, get_missing_parameters_message
from .build import BACKBONE_REGISTRY, build_backbone
from .manifolds import lorentz as L
from fastreid.layers.mlp import MLP

logger = logging.getLogger(__name__)
import math



from .clip.model import build_model, ResidualAttentionBlock
def load_clip_to_cpu(pretrain, pretrain_path, backbone_type, h_resolution, w_resolution, vision_stride_size, with_view):
    if pretrain_path:
        # Load pretrain path if specifically
        pretrain_path = os.path.join(pretrain_path, backbone_type + ".pt")
        try:
            try:
                # model, preprocess = clip.load("ViT-B/16", device=device, download_root=pretrain_path).eval()
                backbone_model = torch.jit.load(pretrain_path, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                state_dict = torch.load(pretrain_path, map_location="cpu")
            backbone_model = build_model(state_dict or backbone_model.state_dict(), h_resolution, w_resolution, vision_stride_size, with_view=with_view, pretrain=pretrain)
            logger.info(f"Loading pretrained model from {pretrain_path}")
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e
    else:
        raise Exception("Please provide pretrain path.")

    return backbone_model


def init_level_prompts(param: torch.nn.Parameter, eps: float = 1e-3):
    """
    param: [P, D]
    eps: Small noise magnitude used to break symmetry; values between 1e-4 and 1e-2 are recommended.
    """
    with torch.no_grad():
        nn.init.zeros_(param)  # Make the initial cross-attention closer to uniform fusion.

        P, D = param.shape
        noise = torch.empty(P, D, device=param.device, dtype=param.dtype)

        # Use orthogonal initialization when P <= D so prompt directions differ from one another.
        if P <= D:
            nn.init.orthogonal_(noise)  # Orthogonal row vectors.
        else:
            nn.init.normal_(noise, std=1.0)  # Fall back to Gaussian noise when P > D.

        param.add_(noise * eps)


def init_residual_attention_block(block: nn.Module,
        proj_std: float = 0.02,
        residual_scale: float = 1e-3,
        zero_init_residual: bool = False):
    """
    Apply a stable initialization strategy to a ResidualAttentionBlock.

    Args:
      proj_std: Standard deviation for fixed-std normal initialization in compatible projects; currently unused.
      residual_scale: Scaling factor for non-zero initialization at the end of each residual branch.
      zero_init_residual: If True, zero-initialize the terminal residual-branch weights for a stronger identity initialization.
    """
    if not hasattr(block, "attn") or not hasattr(block, "mlp"):
        return

    # ---- 1) MultiheadAttention ----
    attn = block.attn  # nn.MultiheadAttention(d_model, n_head)

    # PyTorch MHA parameters are typically named in_proj_weight, in_proj_bias, and out_proj.
    if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
        nn.init.xavier_uniform_(attn.in_proj_weight)
    if hasattr(attn, "in_proj_bias") and attn.in_proj_bias is not None:
        nn.init.zeros_(attn.in_proj_bias)

    if hasattr(attn, "out_proj") and attn.out_proj is not None:
        nn.init.xavier_uniform_(attn.out_proj.weight)
        if attn.out_proj.bias is not None:
            nn.init.zeros_(attn.out_proj.bias)

        # Terminal projection of the residual branch: out_proj.
        if zero_init_residual:
            nn.init.zeros_(attn.out_proj.weight)
        else:
            attn.out_proj.weight.data.mul_(residual_scale)

    # ---- 2) MLP: c_fc / c_proj ----
    # block.mlp = Sequential(OrderedDict([("c_fc", Linear), ("gelu", ...), ("c_proj", Linear)]))
    c_fc = block.mlp[0]  # "c_fc"
    c_proj = block.mlp[2]  # "c_proj"

    # Initialize c_fc with Xavier weights and zero bias.
    nn.init.xavier_uniform_(c_fc.weight)
    if c_fc.bias is not None:
        nn.init.zeros_(c_fc.bias)

    # Initialize c_proj with Xavier weights and zero bias, then scale or zero the residual endpoint.
    nn.init.xavier_uniform_(c_proj.weight)
    if c_proj.bias is not None:
        nn.init.zeros_(c_proj.bias)

    if zero_init_residual:
        nn.init.zeros_(c_proj.weight)
    else:
        c_proj.weight.data.mul_(residual_scale)

    # ---- 3) LayerNorm ----
    # LayerNorm conventionally uses weight=1 and bias=0.
    if hasattr(block, "ln_1"):
        nn.init.ones_(block.ln_1.weight)
        nn.init.zeros_(block.ln_1.bias)
    if hasattr(block, "ln_2"):
        nn.init.ones_(block.ln_2.weight)
        nn.init.zeros_(block.ln_2.bias)

        
class VisionTransformerHierarchy(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ###### Configuration reading
        backbone_type = cfg.MODEL.BACKBONE.BACKBONE_TYPE
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        self.vision_stride_size = cfg.MODEL.BACKBONE.VISION_STRIDE_SIZE
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // self.vision_stride_size + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // self.vision_stride_size + 1)
        self.with_view = cfg.MODEL.BACKBONE.WITH_VIEW
        self.with_fusion = cfg.MODEL.BACKBONE.WITH_FUSION
        self.with_hierarchy = cfg.MODEL.BACKBONE.WITH_HIERARCHY
        if backbone_type == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif backbone_type == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024

        ###### vit clip pretrain module
        clip_model = load_clip_to_cpu(pretrain, pretrain_path, backbone_type, self.h_resolution, self.w_resolution, self.vision_stride_size, self.with_view)
        self.pretrained_image_encoder = clip_model.visual
        self.enc_dtype = next(self.pretrained_image_encoder.parameters()).dtype
        self.pretrained_text_encoder = TextEncoder(clip_model)
        for module_param_name, value in self.pretrained_text_encoder.named_parameters():
            value.requires_grad = False

        ###### Layers Select
        self.selected_layer_indices = [i - 1 for i in cfg.MODEL.BACKBONE.SELECTED_LAYER] # [0 ~ 11]
        self.vit_level_num = len(self.selected_layer_indices) #clip layer start with 0

        ###### Prompt
        self.template_num = 3
        self.text_prompt = PromptLearner(clip_model, ctx_init_std=0.02, n_shared=4, n_private=4)

        ###### Multi-granularity Fusion (TMF)
        self.patch_fusion = AdaptiveWeightGenerator(d_model=self.in_planes_proj, num_heads=4, dtype=self.enc_dtype, qk_init="small")
        self.prompt_fusion = AdaptiveWeightGenerator(d_model=self.in_planes_proj, num_heads=4, dtype=self.enc_dtype, qk_init="small")
        self.prompt_fusion2 = AdaptiveWeightGenerator(d_model=self.in_planes_proj, num_heads=4, dtype=self.enc_dtype, qk_init="small")
        self.patch_prompt_self_attn = ResidualAttentionBlock(d_model=self.in_planes_proj, n_head=4)
        init_residual_attention_block(self.patch_prompt_self_attn, residual_scale=1e-3, zero_init_residual=False)
        # self.fusion_project = MLP([self.in_planes_proj, self.in_planes_proj, self.in_planes_proj], activation="gelu", dropout=0.1)
        self.norm_fusion = nn.LayerNorm(self.in_planes_proj)
        self.norm_prompt = nn.LayerNorm(self.in_planes_proj)

        ###### Hierarchical Hyperbolic Learning (HHL)
        curv_init = float(cfg.MODEL.BACKBONE.CURV_INIT) #0.25 {0.1, 0.5, 1.0} 0.1085
        self.log_curv = nn.Parameter(torch.tensor(curv_init, dtype=self.enc_dtype).log(), requires_grad=True)
        self._curv_minmax = {"max": math.log(curv_init * 10),"min": math.log(curv_init / 10)}

        init_i1 = float(cfg.MODEL.BACKBONE.SCALING_FACTOR_I1) #0.2677
        init_i2 = float(cfg.MODEL.BACKBONE.SCALING_FACTOR_I2) #1.000
        init_i2 = max(init_i2, 1.0 + 1e-6)  # Avoid init_p == 1, which would make log(0) undefined.
        self.log_scaling_factor_i1 = nn.Parameter(torch.tensor(init_i1, dtype=torch.float32).log())
        self.log_scaling_factor_i2 = nn.Parameter(torch.tensor(init_i2 - 1.0, dtype=torch.float32).log())

    def forward(self, imgs=None, cv_embed=None, viewids=None, batched_inputs=None):
        B, C, H, W = imgs.shape  # B=64, C=3 H=256,W=128

        x, xproj, selected_layer, selected_layer_proj = self.pretrained_image_encoder(imgs, cv_emb=None,
            hierarchical=True, selected_layer_indices=self.selected_layer_indices, use_invite=False)  # start with layer 0
        x = x[:, 0]  # 64, 768
        xproj = xproj[:, 0]  # 64, 512

        ############ Layers Select
        selected_layer_cls = [layer[:, 0, :] for layer in selected_layer]  # [B, 768]
        selected_layer_cls = torch.stack(selected_layer_cls, dim=1)  # [B, 3, 768]
        selected_layer_cls_proj = [layer[:, 0, :] for layer in selected_layer_proj]  # [B, 512]
        selected_layer_cls_proj = torch.stack(selected_layer_cls_proj, dim=1)  # [B, 3, 512]
        selected_layer_patch_proj = [layer[:, 1:, :] for layer in selected_layer_proj]  # [B, 3, 128, 512]
        selected_layer_patch_proj = torch.stack(selected_layer_patch_proj, dim=1)  # [B, 3, 128, 512]


        ############ Prompt Generate
        prompt_embeddings, tokenized_prompts = self.text_prompt.forward_coop(selected_layer_cls_proj)  # [B, T, 77, D] [B, T, 77]
        prompt_final = self.pretrained_text_encoder(prompt_embeddings.reshape(B * self.template_num, 77, -1),
            tokenized_prompts.reshape(B * self.template_num, 77)).view(B, self.template_num, -1)  # [B, 2, D]


        ############ Multi-granularity Fusion (TMF)
        q = prompt_final
        kv = selected_layer_cls_proj

        # 1) prompts -> class tokens
        prompt_from_kv, attn_p2c = self.prompt_fusion(query=q, key=kv, value=kv, need_weights=True)  # B,2,512
        weighted_patch = (attn_p2c.unsqueeze(-1).unsqueeze(-1) * selected_layer_patch_proj.unsqueeze(1)).sum(dim=2)  # [B, T, P, D]
        Bp, Tp, Pp, Dp = weighted_patch.shape
        patch_prompt_tokens = torch.cat([prompt_from_kv.unsqueeze(2), weighted_patch], dim=2).reshape(Bp * Tp, Pp + 1, Dp)
        patch_prompt_tokens = self.patch_prompt_self_attn(patch_prompt_tokens)
        prompt_from_kv = patch_prompt_tokens[:, 0, :].reshape(Bp, Tp, Dp)
        q = prompt_from_kv
        selected_layer_fusion = q  # (B,2,D)

        selected_layer_fusion = self.norm_fusion(selected_layer_fusion)
        prompt_final = self.norm_prompt(prompt_final)
        assert torch.isfinite(selected_layer_fusion).all(), "selected_layer_fusion contains NaN/Inf"

        # 2) patch tokens -> class tokens
        # Select the corresponding second-level feature according to each sample's view information.
        if viewids is not None:
            sel = viewids.long() + 1  # (B,) values in {1,2}
            idx = sel.view(-1, 1, 1).expand(-1, 1, selected_layer_fusion.size(-1))  # (B,1,512)
            selected_layer_2 = selected_layer_fusion.gather(1, idx)  # (B,1,512)
            selected_layer_fusion = torch.cat([selected_layer_fusion[:, 0:1, :], selected_layer_2], dim=1) # (B,2,512)
            selected_prompt_2 = prompt_final.gather(1, idx) # (B,1,512)
            prompt_final = torch.cat([prompt_final[:, 0:1, :], selected_prompt_2], dim=1) # (B,2,512)


        ############ Hierarchical Hyperbolic Learning (HHL)
        # 1) Normalize the feature norms for each hierarchy level.
        B, N, D = selected_layer_fusion.shape # (B,2,512)
        curv = self.log_curv.float().exp()
        scaling_factor_i1 = torch.exp(self.log_scaling_factor_i1.float())
        scaling_factor_i2 = 1.0 + 1e-6 + torch.exp(self.log_scaling_factor_i2.float())
        scale = torch.stack([scaling_factor_i1, scaling_factor_i1 * scaling_factor_i2]).view(1, 2, 1).to(dtype=selected_layer_fusion.dtype) # (1,2,1)
        scale_hir = torch.stack([torch.tensor(1.0, device=scaling_factor_i2.device), scaling_factor_i2]).view(1, 2, 1).to(dtype=selected_layer_fusion.dtype) # (1,2,1)

        selected_layer_fusion = F.normalize(selected_layer_fusion, p=2, dim=-1, eps=1e-12)
        selected_layer_fusion_hir = selected_layer_fusion * scale  # (B,2,512) * (1,2,1)
        prompt_final = F.normalize(prompt_final, p=2, dim=-1, eps=1e-12)
        prompt_final = prompt_final * scale_hir  # (B,2,512) * (1,2,1)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            proto_per_sample = self.prototypes_by_id(targets, selected_layer_fusion_hir)  # (B,2,512)

            # 2) Map features to the hyperbolic space.
            image_tree_nodes_hyp = L.exp_map0(selected_layer_fusion_hir.reshape(-1, D), curv=curv).reshape(B, N, D)
            prompt_tree_nodes_hyp = L.exp_map0(prompt_final.reshape(-1, D), curv=curv).reshape(B, N, D)
            proto_tree_nodes_hyp = L.exp_map0(proto_per_sample.reshape(-1, D), curv=curv).reshape(B, N, D)

            # 3) Entailment Regularizations
            factor_loss = 5
            hier_path_consistency_loss_img = factor_loss * L.entailment_loss(proto_tree_nodes_hyp[:, :-1, :], image_tree_nodes_hyp[:, 1:, :], curv=curv)
            hier_path_consistency_loss_prompt = factor_loss * L.entailment_loss(prompt_tree_nodes_hyp[:, :-1, :], prompt_tree_nodes_hyp[:, 1:, :], curv=curv)
            prompt_image_consistency_loss = factor_loss * L.entailment_loss(prompt_tree_nodes_hyp, image_tree_nodes_hyp, curv=curv)

            # 4) Map features back to the Euclidean space.
            image_tree_eur_feat = L.log_map0(image_tree_nodes_hyp.reshape(-1, D), curv=curv).reshape(B, N, D) #B, N, D
            image_tree_eur_feat = image_tree_eur_feat / scale
            return x, xproj, image_tree_eur_feat, (hier_path_consistency_loss_img,hier_path_consistency_loss_prompt, prompt_image_consistency_loss)
        else:
            selected_layer_fusion_hir = selected_layer_fusion_hir / scale
            return x, xproj, selected_layer_fusion_hir, None

    def prototypes_by_id(self, targets: torch.Tensor, image_tree_nodes_hyp: torch.Tensor):
        """
        targets: (B,) int64/long person IDs for each sample.
        image_tree_nodes_hyp: (B, 2, 512) float tensor.

        returns:
          uniq_ids: (N,) unique IDs appearing in the current batch.
          protos: (N, 2, 512) averaged prototype feature for each ID.
          counts: (N,) occurrence count for each ID.
        """
        assert targets.dim() == 1
        targets = targets.long()
        assert image_tree_nodes_hyp.dim() == 3
        B = targets.size(0)
        assert image_tree_nodes_hyp.size(0) == B

        # uniq_ids: (N,), inv: (B,) maps each sample to a group index in [0, N - 1].
        uniq_ids, inv = torch.unique(targets, return_inverse=True)
        N = uniq_ids.size(0)
        feat = image_tree_nodes_hyp

        # Sum features within each ID group.
        protos_sum = torch.zeros((N, *feat.shape[1:]), device=feat.device, dtype=feat.dtype)
        protos_sum.index_add_(0, inv, feat)  # Accumulate features by group.

        # Count samples in each ID group.
        counts = torch.zeros((N,), device=feat.device, dtype=feat.dtype)
        ones = torch.ones((B,), device=feat.device, dtype=feat.dtype)
        counts.index_add_(0, inv, ones)

        # Average features within each ID group.
        protos = protos_sum / counts.view(N, 1, 1).clamp_min(1.0)

        # return uniq_ids, protos, counts
        return protos[inv]

    def clamp_curv(self):
        # with torch.no_grad():
        self.log_curv.clamp_(**self._curv_minmax)


@BACKBONE_REGISTRY.register()
def build_clip_vit_backbone_hqformer(cfg):
    """
    Create a Clip instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = VisionTransformerHierarchy(cfg)

    return model


# -----------------------------
# 1) Text Encoder (reuse CLIP text tower)
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @torch.no_grad()
    def _eot_indices(self, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        # In OpenAI CLIP, EOT token id is the largest id, so argmax finds EOT position.
        return tokenized_prompts.argmax(dim=-1)

    def forward(self, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        prompt_embeddings: [n_cls, 77, dim]
        tokenized_prompts: [n_cls, 77]
        return:           [n_cls, text_dim]
        """
        x = prompt_embeddings + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)  # [77, n_cls, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [n_cls, 77, dim]
        x = self.ln_final(x).type(self.dtype)

        # CLIP uses the hidden state of the EOT token as the sentence-level representation instead of [CLS] or mean pooling.
        eot = self._eot_indices(tokenized_prompts)  # [n_cls]
        idx = torch.arange(x.shape[0], device=x.device)
        x = x[idx, eot]  # [n_cls, dim]
        x = x @ self.text_projection  # [n_cls, text_dim]
        return x


from .clip.clip import tokenize as clip_tokenize
# -----------------------------
# 2) CoOp Prompt Learner
#    Template: "A photo of [] [] [] person."
# -----------------------------
class PromptLearner(nn.Module):
    """
    CoOp-style Prompt Learner with 3 templates:
      0) "A photo of [] [] [] [] person."
      1) "A photo of [] [] [] [] person from [] [] [] [] view."
      2) "A photo of [] [] [] [] person from [] [] [] [] view."
    """

    def __init__(
        self,
        clip_model,
        ctx_init_std: float = 0.02,
        n_shared: int = 4,
        n_private: int = 4,
    ):
        super().__init__()

        self.dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device

        self.n_shared = int(n_shared)
        self.n_private = int(n_private)

        self.templates = [
            {"prefix": "A photo of", "mid": "person.", "suffix": "", "use_private": False},
            {"prefix": "A photo of", "mid": "person from", "suffix": "view.", "use_private": True},
            {"prefix": "A photo of", "mid": "person from", "suffix": "view.", "use_private": True},
        ]
        self.n_templates = len(self.templates)

        ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.ctx_dim = ctx_dim

        self.ctx_shared = nn.Parameter(torch.empty(self.n_shared, ctx_dim, dtype=self.dtype))
        nn.init.normal_(self.ctx_shared, std=ctx_init_std)

        self.ctx_private = nn.Parameter(torch.empty(self.n_templates, self.n_private, ctx_dim, dtype=self.dtype))
        nn.init.normal_(self.ctx_private, std=ctx_init_std)

        ph_shared = " ".join(["X"] * self.n_shared)
        ph_private = " ".join(["Y"] * self.n_private)

        for i, tmpl in enumerate(self.templates):
            prefix = tmpl["prefix"].strip()
            mid = tmpl["mid"].strip()
            suffix = tmpl["suffix"].strip()
            use_private = bool(tmpl["use_private"])

            if use_private:
                prompt = f"{prefix} {ph_shared} {mid} {ph_private} {suffix}".strip()
            else:
                prompt = f"{prefix} {ph_shared} {mid} {suffix}".strip()

            tokenized = clip_tokenize([prompt]).to(device)
            self.register_buffer(f"tokenized_prompts_{i}", tokenized)

            tokenized_prefix = clip_tokenize([prefix]).to(device)
            prefix_eot = tokenized_prefix[0].argmax().item()
            prefix_len = prefix_eot - 1

            mid_len = 0
            if mid:
                tokenized_mid = clip_tokenize([mid]).to(device)
                mid_eot = tokenized_mid[0].argmax().item()
                mid_len = mid_eot - 1

            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized).type(self.dtype)

            token_prefix = embedding[:, : 1 + prefix_len, :]
            start_mid = 1 + prefix_len + self.n_shared
            end_mid = start_mid + mid_len
            token_mid = embedding[:, start_mid:end_mid, :]

            start_suffix = end_mid + (self.n_private if use_private else 0)
            token_suffix = embedding[:, start_suffix:, :]

            total_len = token_prefix.shape[1] + self.n_shared + token_mid.shape[1] + (self.n_private if use_private else 0) + token_suffix.shape[1]
            assert total_len == 77, f"Prompt length mismatch, got {total_len}, expected 77."

            self.register_buffer(f"token_prefix_{i}", token_prefix)
            self.register_buffer(f"token_mid_{i}", token_mid)
            self.register_buffer(f"token_suffix_{i}", token_suffix)

    def _build_one(self, B: int, tid: int, device):
        token_prefix = getattr(self, f"token_prefix_{tid}").to(device=device).expand(B, -1, -1)
        token_mid = getattr(self, f"token_mid_{tid}").to(device=device).expand(B, -1, -1)
        token_suffix = getattr(self, f"token_suffix_{tid}").to(device=device).expand(B, -1, -1)

        shared = self.ctx_shared.unsqueeze(0).expand(B, -1, -1).to(device=device)
        if self.templates[tid]["use_private"]:
            private = self.ctx_private[tid].unsqueeze(0).expand(B, -1, -1).to(device=device)
            prompt_embeddings = torch.cat([token_prefix, shared, token_mid, private, token_suffix], dim=1)
        else:
            prompt_embeddings = torch.cat([token_prefix, shared, token_mid, token_suffix], dim=1)

        tokenized_prompts = getattr(self, f"tokenized_prompts_{tid}").to(device=device).expand(B, -1)
        return prompt_embeddings, tokenized_prompts

    def forward(self, img_feat_or_B, template_idx=None):
        return self.forward_coop(img_feat_or_B, template_idx=template_idx)

    def forward_coop(self, img_feat_or_B, template_idx=None):
        if torch.is_tensor(img_feat_or_B):
            B = img_feat_or_B.shape[0]
            device = img_feat_or_B.device
        else:
            B = int(img_feat_or_B)
            device = self.ctx_shared.device

        if template_idx is None:
            pe_list, tp_list = [], []
            for tid in range(self.n_templates):
                pe, tp = self._build_one(B, tid, device)
                pe_list.append(pe)
                tp_list.append(tp)
            return torch.stack(pe_list, dim=1), torch.stack(tp_list, dim=1)

        if isinstance(template_idx, int):
            return self._build_one(B, template_idx, device)

        assert torch.is_tensor(template_idx) and template_idx.dim() == 1 and template_idx.numel() == B
        pe_all, tp_all = self.forward_coop(B, template_idx=None)
        idx = template_idx.to(device=pe_all.device, dtype=torch.long)
        b = torch.arange(B, device=pe_all.device)
        return pe_all[b, idx], tp_all[b, idx]


# -----------------------------
# 3) Adaptive Weight Generator
# -----------------------------
class AdaptiveWeightGenerator(nn.MultiheadAttention):
    def __init__(self, d_model, num_heads=1, device=None, dtype=None, qk_init="zero", eps=1e-6):
        super().__init__(embed_dim=d_model, num_heads=num_heads, batch_first=True, device=device, dtype=dtype)

        self.d_model = d_model
        self.v_start = 2 * d_model

        eye = torch.eye(d_model, device=device, dtype=dtype)

        with torch.no_grad():
            # --- 1) Initialize Q/K projections to zero or very small values. ---
            if self.in_proj_weight is not None:
                if qk_init == "zero":
                    self.in_proj_weight[:self.v_start, :].zero_()  # Set all Q+K weights to zero.
                elif qk_init == "small":
                    self.in_proj_weight[:self.v_start, :].normal_(0.0, eps)  # Use small random Q+K weights.
                else:
                    raise ValueError("qk_init must be 'zero' or 'small'")

                if self.in_proj_bias is not None:
                    self.in_proj_bias[:self.v_start].zero_()  # Set Q+K bias to zero.

                # --- 2) Initialize the V projection as an identity mapping. ---
                if self.in_proj_weight.shape[0] >= 3 * d_model:
                    self.in_proj_weight[self.v_start:self.v_start + d_model, :].copy_(eye)
                    if self.in_proj_bias is not None:
                        self.in_proj_bias[self.v_start:self.v_start + d_model].zero_()

            else:
                # Compatibility path for modules that expose q_proj_weight, k_proj_weight, and v_proj_weight separately.
                # This branch is not expected for the current construction, but it keeps the initialization robust.
                if qk_init == "zero":
                    self.q_proj_weight.zero_()
                    self.k_proj_weight.zero_()
                else:
                    self.q_proj_weight.normal_(0.0, eps)
                    self.k_proj_weight.normal_(0.0, eps)
                self.v_proj_weight.copy_(eye)

            # --- 3) Initialize out_proj as an identity mapping and freeze it. ---
            self.out_proj.weight.copy_(eye)
            if self.out_proj.bias is not None:
                self.out_proj.bias.zero_()

        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False

    def forward(self, query, key, value, **kwargs):
        return super().forward(query=query, key=key, value=value, **kwargs)
