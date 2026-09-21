#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="One-batch WHU-MARS model smoke test")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pretrain-path", required=True)
    return parser.parse_args()


def finite_gradient(parameter):
    return parameter.grad is not None and torch.isfinite(parameter.grad).all()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the HiHR model smoke test")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.ROOT = args.data_root
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = args.pretrain_path
    cfg.DATALOADER.NUM_WORKERS = 2
    loader = DefaultTrainer.build_train_loader(cfg)
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg, loader.dataset.num_classes)
    batch = next(iter(loader))
    if hasattr(loader, "shutdown"):
        loader.shutdown()
    inference_batch = {
        key: (value[:2].clone() if isinstance(value, torch.Tensor) else value[:2])
        for key, value in batch.items()
    }

    model = DefaultTrainer.build_model(cfg)
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=cfg.SOLVER.AMP.ENABLED):
        losses = model(batch)
        total_loss = sum(losses.values())
    assert torch.isfinite(total_loss), losses
    total_loss.backward()

    baseline_keys = {"loss_CE1", "loss_CE2", "loss_Tri1", "loss_Tri2"}
    full_keys = baseline_keys | {
        "loss_CE_tree1", "loss_CE_tree2", "loss_Tri_tree1", "loss_Tri_tree2",
        "backbone_loss1", "backbone_loss2", "backbone_loss3",
    }
    expected_keys = baseline_keys if cfg.MODEL.HIHR_MODE == "baseline" else full_keys
    loss_keys = set(losses)
    assert loss_keys == expected_keys, (loss_keys, expected_keys)

    if cfg.MODEL.HIHR_MODE == "baseline":
        assert not hasattr(model.backbone, "text_prompt")
        assert not hasattr(model.backbone, "log_curv")
    else:
        assert finite_gradient(model.backbone.log_curv)
        assert any(finite_gradient(p) for p in model.backbone.patch_fusion.parameters())
        assert any(finite_gradient(p) for p in model.backbone.text_prompt.parameters())

    model.zero_grad(set_to_none=True)
    del batch, losses, total_loss
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.SOLVER.AMP.ENABLED):
        features = model(inference_batch)
    expected_dim = 1280 if cfg.MODEL.HIHR_MODE == "baseline" else 2304
    assert features.shape == (2, expected_dim), features.shape
    assert torch.isfinite(features).all()

    print("WHUMARS_MODEL_SMOKE_OK")
    print("mode", cfg.MODEL.HIHR_MODE)
    print("loss_keys", sorted(loss_keys))
    print("feature_shape", tuple(features.shape))


if __name__ == "__main__":
    main()
