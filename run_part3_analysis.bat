@echo off
echo ==========================================
echo RUNNING PART 3: TRANSPORT EXCLUSION ANALYSIS
echo ==========================================
echo.

cd /d "%~dp0"

python --version
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not added to PATH.
    pause
    exit /b
)

echo.
echo Running transport_exclusion_analysis.py ...
python transport_exclusion_analysis.py

echo.
echo ==========================================
echo PART 3 COMPLETE
echo ==========================================
pause