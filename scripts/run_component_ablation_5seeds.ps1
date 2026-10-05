$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }
$strictSplit = "data/v1_1/splits/strict_group_split_v1_1.csv"
$synthetic = "data/v1_1/augmentation/synthetic_duplicate_train_v1_1.csv"
$hardNegative = "data/v1_1/augmentation/hard_negative_train_v1_1.csv"
$baseModel = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
$modes = @("e0_real_only", "e1_positive_only", "e2_hard_only", "e3_combined")
$seeds = @(2026, 2027, 2028, 2029, 2030)
foreach ($mode in $modes) {
  foreach ($seed in $seeds) {
    $modelDir = "models/component_ablation/$mode/seed_$seed"
    $resultDir = "results/component_ablation/$mode/seed_$seed"
    Write-Host "TRAIN $mode seed=$seed" -ForegroundColor Cyan
    python -m fpt_stdd.finetune --strict-split $strictSplit --synthetic $synthetic --hard-negative $hardNegative --model $baseModel --mode $mode --seed $seed --output-dir $modelDir --epochs 3 --batch-size 8 --lr 0.00002
    Write-Host "EVAL $mode seed=$seed" -ForegroundColor Green
    python -m fpt_stdd.cli --config configs/v1_1_ablation.yaml --baselines sbert --protocols strict --text-view title --sbert-model $modelDir --seed $seed --device cuda --batch-size 32 --output-dir $resultDir
  }
}
python -m fpt_stdd.aggregate_ablation --root results/component_ablation --output results/component_ablation_summary.csv
