@echo off
setlocal

call conda activate llm-emotions
if errorlevel 1 ( echo [FAILED] conda activate llm-emotions & exit /b 1 )

echo =========================================================================
echo  Step 1 / 5 -- Collect activations
echo =========================================================================
python collect-activations.py
if errorlevel 1 ( echo [FAILED] collect-activations.py & exit /b 1 )

echo.
echo =========================================================================
echo  Step 2 / 5 -- Train SAE
echo =========================================================================
python train-sae.py
if errorlevel 1 ( echo [FAILED] train-sae.py & exit /b 1 )

echo.
echo =========================================================================
echo  Step 3 / 5 -- Analyse features
echo =========================================================================
python analyse-features.py
if errorlevel 1 ( echo [FAILED] analyse-features.py & exit /b 1 )

echo.
echo =========================================================================
echo  Step 4 / 5 -- Extract steering vectors
echo =========================================================================
python extract-vectors.py
if errorlevel 1 ( echo [FAILED] extract-vectors.py & exit /b 1 )

echo.
echo =========================================================================
echo  Step 5 / 5 -- Ablation and steering demo
echo =========================================================================
python ablation-et-steering.py
if errorlevel 1 ( echo [FAILED] ablation-et-steering.py & exit /b 1 )

echo.
echo =========================================================================
echo  Done.
echo =========================================================================
endlocal
