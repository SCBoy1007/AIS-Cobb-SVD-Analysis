#!/bin/bash

# Testing script for VTF-18V model
# Usage: bash scripts/test.sh

cd "$(dirname "$0")/.."

python src/test.py \
    --model_path checkpoints/model.pth \
    --data_dir dataset/data \
    --labels_dir dataset/labels \
    --test_list dataset/splits/fold0/test.txt \
    --input_h 1536 \
    --input_w 512 \
    --down_ratio 4 \
    --backbone hrnet18 \
    --output_dir results \
    --visualize