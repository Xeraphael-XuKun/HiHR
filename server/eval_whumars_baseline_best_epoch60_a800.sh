#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/HiHR

# model_final.pth is the completion sentinel for the fixed 60-epoch run.
test -f /mnt/cache/wanghanzhi/XK/HiHR/logs/WHUMARS/Baseline/model_final.pth
test -f /mnt/cache/wanghanzhi/XK/HiHR/logs/WHUMARS/Baseline/model_best.pth

exec env CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
  /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/train_net.py \
  --config-file configs/WHUMARS/Baseline.yml \
  --num-gpus 1 \
  --eval-only \
  MODEL.WEIGHTS /mnt/cache/wanghanzhi/XK/HiHR/logs/WHUMARS/Baseline/model_best.pth \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/HiHR/logs/WHUMARS/Baseline/eval_model_best_epoch60 \
  TEST.WHU_DIAGNOSTICS True
