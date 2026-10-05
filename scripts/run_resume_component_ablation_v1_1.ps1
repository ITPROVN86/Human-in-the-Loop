param(
    [int]$Epochs = 3,
    [int]$TrainBatchSize = 8,
    [int]$EvalBatchSize = 32,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Stop-WithMessage([string]$Message) {
    Write-Host "" 
    Write-Host $Message -ForegroundColor Red
    Write-Host "Xem file log được in ngay phía trên để biết lỗi Python gốc." -ForegroundColor Yellow
    exit 1
}

function Backup-IncompleteDirectory([string]$Path) {
    if (Test-Path $Path) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backup = "${Path}_incomplete_${timestamp}"
        Move-Item -Path $Path -Destination $backup
        Write-Host "Đã sao lưu lần chạy dở: $backup" -ForegroundColor DarkYellow
    }
}

Write-Host "FPT-STDD v1.1 - Resume Component Ablation E0-E3" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)"

$requiredFiles = @(
    "data\v1_1\splits\strict_group_split_v1_1.csv",
    "data\v1_1\augmentation\synthetic_duplicate_train_v1_1.csv",
    "data\v1_1\augmentation\hard_negative_train_v1_1.csv",
    "configs\v1_1_ablation.yaml"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Stop-WithMessage "Thiếu file bắt buộc: $file"
    }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Stop-WithMessage "Không tìm thấy python. Hãy chạy: conda activate fpt"
}

python -c "import torch; print('Python/Torch OK'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage "Không import được PyTorch trong môi trường hiện tại."
}

if ($Device -eq "cuda") {
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "CUDA không khả dụng. Kiểm tra môi trường fpt hoặc chạy file với tham số -Device cpu."
    }
}

$strictSplit = "data\v1_1\splits\strict_group_split_v1_1.csv"
$synthetic = "data\v1_1\augmentation\synthetic_duplicate_train_v1_1.csv"
$hardNegative = "data\v1_1\augmentation\hard_negative_train_v1_1.csv"
$baseModel = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
$config = "configs\v1_1_ablation.yaml"
$modes = @("e0_real_only", "e1_positive_only", "e2_hard_only", "e3_combined")
$seeds = @(2026, 2027, 2028, 2029, 2030)
$logRoot = "logs\component_ablation_v1_1"
New-Item -ItemType Directory -Force $logRoot | Out-Null

$completed = 0
$skipped = 0

foreach ($mode in $modes) {
    foreach ($seed in $seeds) {
        $modelDir = "models\component_ablation\$mode\seed_$seed"
        $resultDir = "results\component_ablation\$mode\seed_$seed"
        $manifest = Join-Path $modelDir "training_manifest.json"
        $metrics = Join-Path $resultDir "metrics_summary.csv"
        $trainLog = Join-Path $logRoot "train_${mode}_seed_${seed}.log"
        $evalLog = Join-Path $logRoot "eval_${mode}_seed_${seed}.log"

        if ((Test-Path $manifest) -and (Test-Path $metrics)) {
            Write-Host "SKIP hoàn thành: $mode seed=$seed" -ForegroundColor DarkGray
            $skipped++
            continue
        }

        Write-Host "" 
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host "RUN: $mode seed=$seed" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor DarkCyan

        if (-not (Test-Path $manifest)) {
            Backup-IncompleteDirectory $modelDir

            $trainArgs = @(
                "-m", "fpt_stdd.finetune",
                "--strict-split", $strictSplit,
                "--synthetic", $synthetic,
                "--hard-negative", $hardNegative,
                "--model", $baseModel,
                "--mode", $mode,
                "--seed", "$seed",
                "--output-dir", $modelDir,
                "--epochs", "$Epochs",
                "--batch-size", "$TrainBatchSize",
                "--lr", "0.00002"
            )

            Write-Host "TRAIN - log: $trainLog" -ForegroundColor Yellow
            & python @trainArgs 2>&1 | Tee-Object -FilePath $trainLog
            $trainExitCode = $LASTEXITCODE
            if ($trainExitCode -ne 0) {
                Stop-WithMessage "TRAIN thất bại: $mode seed=$seed, exit code=$trainExitCode"
            }
            if (-not (Test-Path $manifest)) {
                Stop-WithMessage "Train kết thúc nhưng không tạo training_manifest.json: $mode seed=$seed"
            }
        }
        else {
            Write-Host "Model đã có manifest, bỏ qua training: $mode seed=$seed" -ForegroundColor DarkGray
        }

        if (-not (Test-Path $metrics)) {
            Backup-IncompleteDirectory $resultDir

            $evalArgs = @(
                "-m", "fpt_stdd.cli",
                "--config", $config,
                "--baselines", "sbert",
                "--protocols", "strict",
                "--text-view", "title",
                "--sbert-model", $modelDir,
                "--seed", "$seed",
                "--device", $Device,
                "--batch-size", "$EvalBatchSize",
                "--output-dir", $resultDir
            )

            Write-Host "EVAL - log: $evalLog" -ForegroundColor Green
            & python @evalArgs 2>&1 | Tee-Object -FilePath $evalLog
            $evalExitCode = $LASTEXITCODE
            if ($evalExitCode -ne 0) {
                Stop-WithMessage "EVALUATE thất bại: $mode seed=$seed, exit code=$evalExitCode"
            }
            if (-not (Test-Path $metrics)) {
                Stop-WithMessage "Evaluate kết thúc nhưng không tạo metrics_summary.csv: $mode seed=$seed"
            }
        }

        Write-Host "HOÀN THÀNH: $mode seed=$seed" -ForegroundColor Green
        $completed++
    }
}

Write-Host "" 
Write-Host "Đang tổng hợp kết quả..." -ForegroundColor Cyan
python -m fpt_stdd.aggregate_ablation --root "results\component_ablation" --output "results\component_ablation_summary.csv"
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage "Không tổng hợp được kết quả component ablation."
}

$metricFiles = @(Get-ChildItem "results\component_ablation" -Recurse -Filter "metrics_summary.csv" -ErrorAction SilentlyContinue)
Write-Host "" 
Write-Host "============================================================" -ForegroundColor Green
Write-Host "HOÀN TẤT" -ForegroundColor Green
Write-Host "Đã chạy mới: $completed"
Write-Host "Đã bỏ qua: $skipped"
Write-Host "Số metrics_summary.csv: $($metricFiles.Count)/20"
Write-Host "Bảng tổng hợp: results\component_ablation_summary.csv"
Write-Host "Log: $logRoot"
Write-Host "============================================================" -ForegroundColor Green

if ($metricFiles.Count -ne 20) {
    Write-Host "CẢNH BÁO: Chưa đủ 20 lượt. Kiểm tra thư mục log." -ForegroundColor Yellow
    exit 2
}

exit 0
