@echo off
echo ==========================================
echo   HPMA Quiz Assistant Environment Setup
echo ==========================================

:: 1. Create Virtual Environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: 2. Upgrade pip
echo Upgrading pip...
venv\Scripts\python -m pip install --upgrade pip

:: 3. Install Core Dependencies
echo Installing requirements...
venv\Scripts\pip install -r requirements.txt

:: 4. Install CUDA-enabled PyTorch (Override CPU version if present)
echo Installing PyTorch with CUDA 12.1 support...
venv\Scripts\pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo.
echo ==========================================
echo           Setup Complete!
echo ==========================================
pause
