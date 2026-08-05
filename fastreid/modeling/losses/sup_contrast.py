"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

## 多正样本的图文对比损失
def sup_con_loss(text_features, image_features, t_label, i_targets, temperature = 1.0):
    """ Supervised Contrastive Loss
        text_features: [batch, feature_dim]
        image_features: [batch, feature_dim]
        t_label: 对应文本的标签 [batch]
        i_targets: 对应图像的标签 [batch]
    """
    # 1. 构造相似度 mask，标签相同即为正样本
    batch_size = text_features.shape[0]
    batch_size_N = image_features.shape[0]
    mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float()

    # 2. 相似度计算
    logits = torch.div(torch.matmul(text_features, image_features.T),temperature)
    # 3. 数值稳定性处理 for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    # 4. softmax 概率计算
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # 5. 只取正样本的平均对比损失
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = - mean_log_prob_pos.mean()
    return loss

## 多正样本的图像内部对比损失
def proto_supcon_loss(
    img_features: torch.Tensor,   # [N, D]
    img_ids: torch.Tensor,        # [N] (int/long)
    temperature: float = 0.07,
    normalize: bool = True,
    exclude_own_members: bool = False,  # True: anchor 不看本ID的成员（一般不建议）
    eps: float = 1e-12,
):
    """
    Prototype supervised contrastive loss (image-only):
    - anchor: mean feature per ID (prototype)
    - positives: all samples with same ID (multi-positive)
    - negatives: samples with different IDs
    """

    x = img_features
    y = img_ids

    if normalize:
        x = F.normalize(x, dim=-1)

    # unique IDs and inverse index: y = uniq[inv]
    uniq, inv = torch.unique(y, sorted=True, return_inverse=True)  # uniq: [C], inv: [N]
    C = uniq.numel()
    N, D = x.shape

    # ---- compute prototypes p_c = mean_{k in c} x_k ----
    # sum per class
    p = torch.zeros(C, D, device=x.device, dtype=x.dtype)
    p.index_add_(0, inv, x)

    # count per class
    counts = torch.bincount(inv, minlength=C).to(x.dtype).unsqueeze(1)  # [C,1]
    p = p / counts.clamp_min(1.0)

    if normalize:
        p = F.normalize(p, dim=-1)

    # ---- logits: [C, N] ----
    logits = (p @ x.t()) / temperature

    # numerical stability (row-wise)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # ---- mask positives: [C, N], pos[c,k]=1 if y_k==uniq[c] ----
    pos_mask = (inv.unsqueeze(0) == torch.arange(C, device=x.device).unsqueeze(1)).to(x.dtype)

    if exclude_own_members:
        # prototype来自本ID均值，如果你想避免“包含自身”的正样本泄漏，可把本ID成员全排除（通常会导致无正样本）
        pos_mask = pos_mask * 0.0

    # log-softmax over N
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # multi-positive mean
    pos_cnt = pos_mask.sum(dim=1)  # [C]
    # 防 NaN：只对 pos_cnt>0 的类算
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    mean_log_prob_pos = (pos_mask[valid] * log_prob[valid]).sum(dim=1) / pos_cnt[valid].clamp_min(eps)

    loss = -mean_log_prob_pos.mean()
    return loss
