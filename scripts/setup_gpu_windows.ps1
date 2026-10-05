$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python scripts/check_gpu.py

