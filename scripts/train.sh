#!/bin/bash

# Training script for VTF-18V model
# Usage: bash scripts/train.sh

cd "$(dirname "$0")/.."

python src/train.py \
    --gpu 0 \
    --data_dir dataset/data \
    --cross_dir dataset/splits \
    --input_h 1536 \
    --input_w 512 \
    --down_ratio 4 \
    --epoch 100 \
    --batch_size 4 \
    --save_path checkpoints \
    --phase train \
    --backbone hrnet18