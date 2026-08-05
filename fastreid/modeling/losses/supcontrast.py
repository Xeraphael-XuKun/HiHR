"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

def SupConLoss(text_features, image_features, t_label, i_targets, temperature = 1.0):
    """ Supervised Contrastive Loss
        text_features: [batch, feature_dim]
        image_features: [batch, feature_dim]
        t_label: 对应文本的标签 [batch]
        i_targets: 对应图像的标签 [batch]
    """
    # 1. 构造相似度 mask，标签相同即为正样本
    batch_size = text_features.shape[0]
    batch_size_N = image_features.shape[0]
    mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
        i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float()

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