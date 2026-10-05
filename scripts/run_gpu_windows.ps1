$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    throw "Khong tim thay .venv. Hay chay: .\scripts\setup_gpu_windows.ps1"
}
& .\.venv\Scripts\Activate.ps1
python scripts/check_gpu.py
python -m fpt_stdd.cli --config configs/default.yaml --device cuda --batch-size 32

