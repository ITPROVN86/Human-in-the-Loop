#!/usr/bin/env bash
set -euo pipefail
python -m fpt_stdd.finetune \
  --strict-split data/v1_1/splits/strict_group_split_v1_1.csv \
  --synthetic data/v1_1/augmentation/synthetic_duplicate_train_v1_1.csv \
  --hard-negative data/v1_1/augmentation/hard_negative_train_v1_1.csv \
  --mode e3_combined --output-dir models/sbert_augmented_v1_1 \
  --epochs 3 --batch-size 8
