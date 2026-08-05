# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN DETECTED] {name} has NaN or Inf")
        pdb.set_trace()

@META_ARCH_REGISTRY.register()
class Baseline_vit_HiHR(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            cfg,
            backbone,
            # head_global,
            pixel_mean,
            pixel_std,
            camera_num,
            view_num,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.MODEL.BACKBONE.BACKBONE_TYPE
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024

        self.camera_num = camera_num
        self.view_num = view_num
        self.use_cv_embed = bool(cfg.MODEL.BACKBONE.WITH_VIEW)
        if self.camera_num and self.view_num:
            self.cv_embed = nn.Parameter(torch.zeros(int(self.camera_num) * int(self.view_num), self.in_planes))
            trunc_normal_(self.cv_embed, std=1.0)
            print('camera*view number is : {}'.format(int(self.camera_num) * int(self.view_num)))
        elif self.camera_num:
            self.cv_embed = nn.Parameter(torch.zeros(int(self.camera_num), self.in_planes))
            trunc_normal_(self.cv_embed, std=1.0)
            print('camera number is : {}'.format(int(self.camera_num)))
        elif self.view_num:
            self.cv_embed = nn.Parameter(torch.zeros(int(self.view_num), self.in_planes))
            trunc_normal_(self.cv_embed, std=1.0) #.02
            print('view number is : {}'.format(int(self.view_num)))
        self.sie_coe = 1.0

        # backbone
        self.backbone = backbone

        # head
        self.head_global = build_heads(cfg, dim=self.in_planes)
        self.head_global_proj = build_heads(cfg, dim=self.in_planes_proj)
        self.head_image_tree_1 = build_heads(cfg, dim=self.in_planes_proj)  # 3 * self.in_planes_proj
        self.head_image_tree_2 = build_heads(cfg, dim=self.in_planes_proj)

        # loss
        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        ###### camera and view embedding
        if 'camids' in batched_inputs.keys():
            # camids = batched_inputs['camids'].tolist()
            camids = batched_inputs['camids'].to(self.device).long()
        else:
            camids = None
        if 'viewids' in batched_inputs.keys():
            viewids = []
            for v in batched_inputs['viewids']:
                if v == 'Aerial':
                    viewids.append(1)
                elif v == 'Ground':
                    viewids.append(0)
                else:
                    raise NotImplementedError
        else:
            viewids = None
        if viewids is not None:
            viewids = torch.tensor(viewids, dtype=torch.long, device=self.device)

        cv_embed = None
        if self.use_cv_embed and hasattr(self, "cv_embed"):
            if camids is not None and viewids is not None:
                cv_embed = self.sie_coe * self.cv_embed[(camids * int(self.view_num) + viewids).tolist()]
            elif camids is not None:
                cv_embed = self.sie_coe * self.cv_embed[camids.tolist()]
            elif viewids is not None:
                cv_embed = self.sie_coe * self.cv_embed[viewids.tolist()]

        losses_all = {}
        ###### input into backbone
        global_feats, global_feats_proj, tree_feats_eur, backbone_losses = self.backbone(
            imgs=images, cv_embed=cv_embed, viewids=viewids, batched_inputs=batched_inputs)  # 64, 512
        if backbone_losses is not None:
            for i, backbone_loss in enumerate(backbone_losses):
                losses_all[f'backbone_loss{i+1}'] = backbone_loss

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            if targets.sum() < 0: targets.zero_()

            outputs_global = self.head_global(global_feats.reshape(global_feats.shape[0], -1, 1, 1), targets)
            outputs_global_proj = self.head_global_proj(global_feats_proj.reshape(global_feats_proj.shape[0], -1, 1, 1), targets)
            outputs_tree_eur_1 = self.head_image_tree_1(
                tree_feats_eur[:, 0, :].reshape(tree_feats_eur[:, 0, :].shape[0], -1, 1, 1), targets)
            outputs_tree_eur_2 = self.head_image_tree_2(
                tree_feats_eur[:, 1, :].reshape(tree_feats_eur[:, 1, :].shape[0], -1, 1, 1), targets)
            losses_re = self.losses((outputs_global, outputs_global_proj), gt_labels=targets,
                outputs_special=(outputs_tree_eur_1,outputs_tree_eur_2,), viewids=viewids)
            losses_all = {**losses_all, **losses_re}
            return losses_all
        else:
            outputs_global = self.head_global(global_feats.reshape(global_feats.shape[0], -1, 1, 1))
            outputs_global_proj = self.head_global_proj(global_feats_proj.reshape(global_feats_proj.shape[0], -1, 1, 1))
            outputs_tree_eur_1 = self.head_image_tree_1(
                tree_feats_eur[:, 0, :].reshape(tree_feats_eur[:, 0, :].shape[0], -1, 1, 1))
            outputs_tree_eur_2 = self.head_image_tree_2(
                tree_feats_eur[:, 1, :].reshape(tree_feats_eur[:, 1, :].shape[0], -1, 1, 1))
            return torch.cat((outputs_global, outputs_global_proj, outputs_tree_eur_1, outputs_tree_eur_2), dim=1)

    def losses(self, outputs=None, gt_labels=None, outputs_special=None, viewids=None):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs[0]['pred_class_logits'].detach()
        cls_outputs = outputs[0]['cls_outputs']
        pred_features = outputs[0]['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            for i, output in enumerate(outputs):
                loss_dict[f'loss_CE{i+1}'] = cross_entropy_loss(
                    output['cls_outputs'],
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale')

            if outputs_special is not None:
                loss_dict['loss_CE_tree1'] = cross_entropy_loss(
                    outputs_special[0]['cls_outputs'],
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale')
            if self.cfg.MODEL.BACKBONE.WITH_HIERARCHY:
                loss_dict['loss_CE_tree2'] = 0.25 * cross_entropy_loss(
                    outputs_special[1]['cls_outputs'],
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            for i, output in enumerate(outputs):
                loss_dict[f'loss_Tri{i+1}'] = triplet_loss(
                    output['features'],
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale')

            if outputs_special is not None:
                loss_dict[f'loss_Tri_tree1'] = triplet_loss(
                    outputs_special[0]['features'],
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale')
            if self.cfg.MODEL.BACKBONE.WITH_HIERARCHY:
                loss_dict[f'loss_Tri_tree2'] = self.view_unique_triplet_loss(
                    outputs_special[1]['features'], gt_labels, viewids, tri_kwargs, use_hir_triplet=False)

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict

    def view_unique_triplet_loss(self, features, gt_labels, viewids, tri_kwargs, use_hir_triplet):
        mask = viewids.bool()  # 1 -> True, 0 -> False
        features_aerial = features[mask]  # 所有 index==1 的行，形状 [B1, 512]
        features_ground = features[~mask]  # 所有 index==0 的行，形状 [B0, 512]
        gt_labels_aerial = gt_labels[mask]  # 所有 index==1 的行，形状 [B1]
        gt_labels_ground = gt_labels[~mask]  # 所有 index==0 的行，形状 [B0]

        if use_hir_triplet:
            return triplet_loss_hyperbolic(
                features_aerial,
                gt_labels_aerial,
                tri_kwargs.get('margin'),
                tri_kwargs.get('hard_mining'),
                use_expmap=True,
                curv=float(self.backbone.log_curv.exp())
            ) * tri_kwargs.get('scale') + triplet_loss_hyperbolic(
                features_ground,
                gt_labels_ground,
                tri_kwargs.get('margin'),
                tri_kwargs.get('hard_mining'),
                use_expmap=True,
                curv=float(self.backbone.log_curv.exp())
            ) * tri_kwargs.get('scale')
        else:
            return triplet_loss(
                features_aerial,
                gt_labels_aerial,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') + triplet_loss(
                features_ground,
                gt_labels_ground,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        cfg0 = cfg.clone()
        if cfg0.is_frozen(): cfg0.defrost()
        cfg0.MODEL.HEADS.NUM_CLASSES = 2

        return {
            'cfg': cfg,
            'backbone': backbone,
            # 'head_global': head_global,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'camera_num': cfg.DATASETS.CAMERANMU,
            'view_num': cfg.DATASETS.VIEWNMU,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'view_id': cfg.MODEL.LOSSES.CE.VIEW_ID,
                        'view_oreg': cfg.MODEL.LOSSES.CE.VIEW_OREG,
                        'view_lambda': cfg.MODEL.LOSSES.CE.VIEW_LAMBDA,
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },

                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images