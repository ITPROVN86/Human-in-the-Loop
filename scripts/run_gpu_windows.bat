@echo off
if not exist ".venv\Scripts\activate.bat" (
  echo Khong tim thay .venv. Hay cai moi truong truoc.
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\check_gpu.py
if errorlevel 1 exit /b 1
python -m fpt_stdd.cli --config configs\default.yaml --device cuda --batch-size 32

