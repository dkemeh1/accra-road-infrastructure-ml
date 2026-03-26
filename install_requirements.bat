@echo off
echo ============================================
echo INSTALLING REQUIRED PYTHON PACKAGES
echo ============================================

pip install numpy pandas geopandas rasterio shapely scikit-learn xgboost lightgbm joblib

echo ============================================
echo INSTALLATION FINISHED
echo ============================================

pause