# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import euclidean_dist, cosine_dist


# from triplet_loss import triplet_loss

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


def prototype_alignment_loss(proto1, proto2, align_choice):
    """
    同类原型对齐的 L2 损失
    """
    if proto1 is None or proto2 is None:
        return torch.tensor(0.0)
    if align_choice == "MSE":
        return F.mse_loss(proto1, proto2)  # mean L2 loss
    elif align_choice == "SE":
        return torch.norm(proto1 - proto2, p=2)  # L2 loss
    else:
        raise NotImplementedError


# def prototype_triplet_loss(all_ids, proto_view1, proto_view2):
#     """
#     原型级 Triplet Loss
#     从 view1 和 view2 各取一个版本的原型作为 anchor 和 positive
#     """
#     if len(all_ids) < 2:
#         return torch.tensor(0.0, device='cuda')
#
#     # anchor 是 view1 原型，positive 是 view2 原型
#     anchor = proto_view1
#     positive = proto_view2
#
#     # 组织 negative：从其他 ID 随机选或全量对比
#     neg_list = []
#     for i in range(anchor.size(0)):
#         neg_candidates = torch.cat([anchor[:i], anchor[i+1:]], dim=0)
#         neg_list.append(neg_candidates.mean(dim=0, keepdim=True))  # 可用 batch hard 挖掘 hardest negative
#     negative = torch.cat(neg_list, dim=0)
#
#     loss = nn.TripletMarginLoss(anchor, positive, negative)
#     return loss
# def prototype_triplet_loss(proto_view1, proto_view2, targets, margin, norm_feat, hard_mining):
def prototype_triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
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


def prototype_loss(prototype_manager, margin=0.3, norm_feat=False, hard_mining=True, align_choice="MSE"):
    """
    ids: 当前 batch 中包含的 ID（去重后的）
    """
    valid_ids, all_proto_view1, all_proto_view2 = prototype_manager.get_all_prototypes()
    alignment_losses = []

    # 收集原型用于计算 align loss
    for i, pid in enumerate(valid_ids):
        # if (not torch.equal(all_proto_view1[i], prototype_manager.proto_view1[pid])) or (not torch.equal(all_proto_view2[i], prototype_manager.proto_view2[pid])):
        #     raise Exception
        alignment_losses.append(
            prototype_alignment_loss(all_proto_view1[i], all_proto_view2[i], align_choice=align_choice))

    if len(alignment_losses) > 0:
        alignment_loss = torch.stack(alignment_losses).mean()
    else:
        alignment_loss = torch.tensor(0.0, device=prototype_manager.device).float()

    # 计算 triplet loss
    if len(valid_ids) > 4:
        embedding = torch.cat((all_proto_view1, all_proto_view2), dim=0)
        targets = torch.tensor(valid_ids, dtype=torch.long, device=embedding.device).repeat(2)
        proto_triplet_loss = prototype_triplet_loss(embedding=embedding, targets=targets, margin=margin,
                                                    norm_feat=norm_feat, hard_mining=hard_mining)
    else:
        proto_triplet_loss = torch.tensor(0.0, device=prototype_manager.device).float()

    return alignment_loss, proto_triplet_loss

# def prototype_loss(prototype_manager, margin=0.3, norm_feat=False, hard_mining=True):
#     """
#     ids: 当前 batch 中包含的 ID（去重后的）
#     """
#     alignment_losses = []
#     all_proto_view1 = []
#     all_proto_view2 = []
#     valid_ids = []
#
#     # 收集原型用于计算 triplet loss
#     for _,pid in enumerate(prototype_manager.proto_view1):
#         proto1, proto2 = prototype_manager.get_prototype(pid)
#         if proto1 is not None and proto2 is not None:
#             alignment_losses.append(prototype_alignment_loss(proto1, proto2))
#             all_proto_view1.append(proto1.unsqueeze(0)) # B, dim
#             all_proto_view2.append(proto2.unsqueeze(0))
#             valid_ids.append(pid) #[tensor]
#
#     if len(alignment_losses) > 0:
#         alignment_loss = torch.stack(alignment_losses).mean()
#     else:
#         alignment_loss = torch.tensor(0.0, device='cuda')
#
#     if len(valid_ids) > 4:
#         embedding = torch.stack(all_proto_view1 + all_proto_view2, dim=0)
#         targets = torch.tensor(valid_ids + valid_ids, dtype=torch.long, device=embedding.device)
#         proto_triplet_loss = prototype_triplet_loss(embedding=embedding, targets=targets, margin=margin,
#                                                     norm_feat=norm_feat, hard_mining=hard_mining)
#     else:
#         proto_triplet_loss = torch.tensor(0.0, device='cuda')
#
#     return alignment_loss, proto_triplet_loss
