@echo off
setlocal

set ENV_NAME=llm-emotions
set PYTHON_VERSION=3.11.6
set CUDA_TAG=cu126

echo [1/4] Creating conda environment "%ENV_NAME%" with Python %PYTHON_VERSION%...
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if errorlevel 1 (echo ERROR: conda create failed & exit /b 1)

echo [2/4] Activating environment...
call conda activate %ENV_NAME%
if errorlevel 1 (echo ERROR: conda activate failed & exit /b 1)

echo [3/4] Installing PyTorch for CUDA %CUDA_TAG%...
python -m pip install torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/%CUDA_TAG%
if errorlevel 1 (echo ERROR: torch install failed & exit /b 1)

echo [4/4] Installing project dependencies...
python -m pip install transformers accelerate
if errorlevel 1 (echo ERROR: dependency install failed & exit /b 1)

echo.
echo Setup complete. Activate with:  conda activate %ENV_NAME%
endlocal
