#!/usr/bin/env bash
set -euo pipefail
strict_split="data/v1_1/splits/strict_group_split_v1_1.csv"
synthetic="data/v1_1/augmentation/synthetic_duplicate_train_v1_1.csv"
hard_negative="data/v1_1/augmentation/hard_negative_train_v1_1.csv"
base_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
for mode in e0_real_only e1_positive_only e2_hard_only e3_combined; do
  for seed in 2026 2027 2028 2029 2030; do
    model_dir="models/component_ablation/$mode/seed_$seed"
    result_dir="results/component_ablation/$mode/seed_$seed"
    python -m fpt_stdd.finetune --strict-split "$strict_split" --synthetic "$synthetic" --hard-negative "$hard_negative" --model "$base_model" --mode "$mode" --seed "$seed" --output-dir "$model_dir" --epochs 3 --batch-size 8 --lr 0.00002
    python -m fpt_stdd.cli --config configs/v1_1_ablation.yaml --baselines sbert --protocols strict --text-view title --sbert-model "$model_dir" --seed "$seed" --device cuda --batch-size 32 --output-dir "$result_dir"
  done
done
python -m fpt_stdd.aggregate_ablation --root results/component_ablation --output results/component_ablation_summary.csv
