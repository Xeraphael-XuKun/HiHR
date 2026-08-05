#!/usr/bin/env python
# encoding: utf-8

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext

import torch

sys.path.append(".")


DEFAULT_WEIGHTS = "./logs/AG_ReID_v2/HiHR/model_best.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Profile FastReID model efficiency")
    parser.add_argument(
        "--config-file",
        default="./configs/AGReIDv2/HiHR.yml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for profiling, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for dummy inference",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Warmup forward iterations before timing",
    )
    parser.add_argument(
        "--repeat-iters",
        type=int,
        default=200,
        help="Timed forward iterations",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast during profiling",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    return parser.parse_args()


def infer_num_classes_from_checkpoint(weights_path):
    if not weights_path or not os.path.isfile(weights_path):
        return None

    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        return None

    candidates = []
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key.endswith(".weight") and "head" in clean_key and hasattr(value, "shape"):
            if len(value.shape) == 2 and value.shape[0] > 0:
                candidates.append(int(value.shape[0]))

    if not candidates:
        return None

    # Classification heads share the same class count; use the most frequent
    # leading dimension to avoid picking BN or projection weights.
    return max(set(candidates), key=candidates.count)


def setup_cfg(args):
    from fastreid.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    opts = list(args.opts or [])
    opts.extend(["MODEL.DEVICE", args.device])
    if args.weights:
        opts.extend(["MODEL.WEIGHTS", args.weights])

    cfg.merge_from_list(opts)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False

    num_classes = infer_num_classes_from_checkpoint(cfg.MODEL.WEIGHTS)
    if num_classes is not None:
        cfg.MODEL.HEADS.NUM_CLASSES = num_classes
    elif cfg.MODEL.HEADS.NUM_CLASSES == 0:
        # The HQFormer eval graph does not use classifier logits, but the
        # EmbeddingHead still needs a non-empty classifier parameter at build.
        cfg.MODEL.HEADS.NUM_CLASSES = 2

    cfg.freeze()
    return cfg


def build_dummy_inputs(cfg, batch_size):
    height, width = cfg.INPUT.SIZE_TEST
    device = torch.device(cfg.MODEL.DEVICE)
    return {
        "images": torch.randn(batch_size, 3, height, width, device=device),
        "camids": torch.zeros(batch_size, dtype=torch.long, device=device),
        "viewids": ["Ground"] * batch_size,
    }


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def forward_once(model, inputs, amp=False):
    with torch.no_grad(), autocast_context(next(model.parameters()).device, amp):
        return model(inputs)


def profile_flops(model, inputs, amp=False):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            forward_once(model, inputs, amp=amp)
        total_flops = sum(getattr(event, "flops", 0) for event in prof.key_averages())
    except Exception as exc:
        return None, "torch.profiler FLOPs failed: {}".format(exc)

    if total_flops <= 0:
        return None, "torch.profiler returned 0 FLOPs; install/use fvcore or thop for a fuller estimate."
    return float(total_flops), "torch.profiler estimated FLOPs"


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_speed_and_memory(model, inputs, warmup_iters, repeat_iters, amp=False):
    device = next(model.parameters()).device

    for _ in range(max(0, warmup_iters)):
        forward_once(model, inputs, amp=amp)
    synchronize_if_cuda(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(repeat_iters):
            forward_once(model, inputs, amp=amp)
        end_event.record()
        synchronize_if_cuda(device)

        elapsed_ms = start_event.elapsed_time(end_event)
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        start_time = time.perf_counter()
        for _ in range(repeat_iters):
            forward_once(model, inputs, amp=amp)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        memory_mb = None

    batch_size = inputs["images"].shape[0]
    total_images = repeat_iters * batch_size
    ms_per_image = elapsed_ms / max(total_images, 1)
    fps = 1000.0 / ms_per_image if ms_per_image > 0 else float("inf")
    return ms_per_image, fps, memory_mb


def format_params(num_params):
    return "{:.2f} M".format(num_params / 1e6)


def format_flops(flops):
    if flops is None:
        return "N/A"
    return "{:.2f} GFLOPs".format(flops / 1e9)


def main():
    args = parse_args()
    from fastreid.modeling import build_model
    from fastreid.utils.checkpoint import Checkpointer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.repeat_iters <= 0:
        raise ValueError("--repeat-iters must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    cfg = setup_cfg(args)
    device = torch.device(cfg.MODEL.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False")

    model = build_model(cfg)
    model.eval()

    if cfg.MODEL.WEIGHTS:
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    inputs = build_dummy_inputs(cfg, args.batch_size)
    total_params = sum(param.numel() for param in model.parameters())

    # Validate the forward path before collecting timings.
    forward_once(model, inputs, amp=args.amp)
    synchronize_if_cuda(device)

    flops, flops_note = profile_flops(model, inputs, amp=args.amp)
    ms_per_image, fps, memory_mb = profile_speed_and_memory(
        model,
        inputs,
        warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        amp=args.amp,
    )

    input_shape = list(inputs["images"].shape)
    memory_text = "N/A" if memory_mb is None else "{:.2f} MB".format(memory_mb)

    print("Input shape: {}".format(input_shape))
    print("Total Params: {}".format(format_params(total_params)))
    print("FLOPs Forward: {}".format(format_flops(flops)))
    print("GPU Memory Allocate: {}".format(memory_text))
    print("Inference Speed: {:.2f} ms/image, {:.2f} FPS".format(ms_per_image, fps))
    print("FLOPs Note: {}".format(flops_note))
    print("AMP: {}".format("enabled" if args.amp else "disabled"))


if __name__ == "__main__":
    main()
