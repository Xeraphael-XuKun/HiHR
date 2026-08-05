# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

from .utils import euclidean_dist, cosine_dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss

import torch
import torch.nn.functional as F
from ..backbones.manifolds.lorentz import pairwise_dist, exp_map0

def triplet_loss_hyperbolic(
    embedding, targets, margin,
    hard_mining=True,
    use_expmap=False,
    curv=1.0,
    eps=1e-6,
):
    """
    embedding: (N, D)
      - 若 use_expmap=False：embedding 应该已经是 hyperboloid 的空间分量
      - 若 use_expmap=True ：embedding 是欧式(tangent)向量，将用 exp_map0 投到 hyperboloid

    targets: (N,)
    """
    # Check for NaN/Inf in input embedding
    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
        # Replace NaN/Inf with zeros
        embedding = torch.where(torch.isfinite(embedding), embedding, torch.zeros_like(embedding))
    
    if use_expmap:
        embedding = exp_map0(embedding, curv=curv, eps=eps)  # 输出仍是空间分量 :contentReference[oaicite:3]{index=3}
        # Check again after exp_map0
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            embedding = torch.where(torch.isfinite(embedding), embedding, torch.zeros_like(embedding))

    dist_mat = pairwise_dist(embedding, embedding, curv=curv, eps=eps)  # (N,N) :contentReference[oaicite:4]{index=4}
    
    # Check for NaN/Inf in distance matrix and replace with small positive values
    if torch.isnan(dist_mat).any() or torch.isinf(dist_mat).any():
        # Replace NaN/Inf with eps to maintain valid distances
        dist_mat = torch.where(torch.isfinite(dist_mat), dist_mat, torch.ones_like(dist_mat) * eps)
        # Ensure diagonal is zero (distance to self)
        dist_mat = dist_mat - torch.diag(torch.diag(dist_mat))

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).eq(targets.view(1, N)).float()
    is_neg = 1.0 - is_pos

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)
    
    # Check for NaN/Inf in dist_ap and dist_an
    if torch.isnan(dist_ap).any() or torch.isinf(dist_ap).any():
        dist_ap = torch.where(torch.isfinite(dist_ap), dist_ap, torch.ones_like(dist_ap) * eps)
    if torch.isnan(dist_an).any() or torch.isinf(dist_an).any():
        dist_an = torch.where(torch.isfinite(dist_an), dist_an, torch.ones_like(dist_an) * (eps + 1.0))

    y = dist_an.new_ones(dist_an.size())

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if torch.isnan(loss) or torch.isinf(loss) or loss == float("Inf"):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
    
    # Final check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        # Fallback to a small positive loss value
        loss = torch.tensor(eps, device=loss.device, dtype=loss.dtype, requires_grad=True)

    return loss
