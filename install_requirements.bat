@echo off
echo Installing required Python packages...
python -m pip install --upgrade pip
pip install numpy pandas geopandas rasterio shapely scikit-learn xgboost lightgbm joblib scipy openpyxl
echo.
echo Installation complete.
pause