@echo off
echo === Creating Python 3.12 venv with CUDA PyTorch ===

py -3.12 -m venv venv312
if errorlevel 1 (
    echo ERROR: Python 3.12 not found. Install from python.org first.
    pause
    exit /b 1
)

echo Installing base packages...
venv312\Scripts\pip install --upgrade pip
venv312\Scripts\pip install rasterio numpy scipy boto3 geopandas shapely scikit-learn lightgbm tqdm matplotlib joblib ipykernel

echo Installing PyTorch with CUDA 12.4...
venv312\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo Registering Jupyter kernel...
venv312\Scripts\python -m ipykernel install --user --name venv312 --display-name "Python 3.12 (CUDA)"

echo.
echo === Done! ===
echo In Jupyter: Kernel menu -> Change Kernel -> "Python 3.12 (CUDA)"
pause