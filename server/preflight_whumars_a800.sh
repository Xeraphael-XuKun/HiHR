#!/usr/bin/env bash
set -euo pipefail

test -x /mnt/cache/wanghanzhi/envs/llmpar/bin/python3
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS
test -f /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
test -f /mnt/cache/wanghanzhi/XK/CVPR26_UAD/utils/metrics.py

cd /mnt/cache/wanghanzhi/XK/HiHR

env CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
  /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/smoke_whumars.py \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --uad-root /mnt/cache/wanghanzhi/XK/CVPR26_UAD \
  --batches 20

env CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
  /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/smoke_whumars_model.py \
  --config-file configs/WHUMARS/Baseline.yml \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --pretrain-path /mnt/cache/wanghanzhi/Datasets

env CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
  /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/smoke_whumars_model.py \
  --config-file configs/WHUMARS/HiHR.yml \
  --data-root /mnt/cache/wanghanzhi/Datasets \
  --pretrain-path /mnt/cache/wanghanzhi/Datasets
