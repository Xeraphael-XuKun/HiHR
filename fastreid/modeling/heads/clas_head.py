# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from fastreid.config import configurable
from fastreid.modeling.heads import REID_HEADS_REGISTRY, EmbeddingHead


@REID_HEADS_REGISTRY.register()
class ClasHead(EmbeddingHead):
    @configurable
    def __init__(self, *, dim=None, **kwargs):
        """.. experimental::"""
        if dim is not None:
            kwargs["feat_dim"] = dim
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, **kwargs):
        params = super().from_config(cfg)
        params["dim"] = kwargs.get("dim", None)
        return params

    def forward(self, features, targets=None):
        """
        See :class:`ClsHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)

        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Evaluation
        if not self.training: return logits.mul_(self.cls_layer.s)

        cls_outputs = self.cls_layer(logits.clone(), targets)

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul_(self.cls_layer.s),
            "features": neck_feat,
        }

# @REID_HEADS_REGISTRY.register()
# class ClasHead(EmbeddingHead):
#     def forward(self, features, targets=None):
#         """
#         See :class:`ClsHeads.forward`.
#         """
#         pool_feat = self.pool_layer(features)
#         neck_feat = self.bottleneck(pool_feat)
#         neck_feat = neck_feat.view(neck_feat.size(0), -1)
#
#         if self.cls_layer.__class__.__name__ == 'Linear':
#             logits = F.linear(neck_feat, self.weight)
#         else:
#             logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))
#
#         # Evaluation
#         if not self.training: return logits.mul_(self.cls_layer.s)
#
#         cls_outputs = self.cls_layer(logits.clone(), targets)
#
#         return {
#             "cls_outputs": cls_outputs,
#             "pred_class_logits": logits.mul_(self.cls_layer.s),
#             "features": neck_feat,
#         }
