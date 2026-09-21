#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/HiHR
test -f /mnt/cache/wanghanzhi/XK/HiHR/logs/WHUMARS/HiHR/model_final.pth

exec env CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
  /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/train_net.py \
  --config-file configs/WHUMARS/HiHR.yml \
  --num-gpus 1 \
  --eval-only \
  MODEL.WEIGHTS /mnt/cache/wanghanzhi/XK/HiHR/logs/WHUMARS/HiHR/model_final.pth \
  TEST.WHU_DIAGNOSTICS True
