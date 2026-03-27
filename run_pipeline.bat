@echo off
echo ============================================
echo RUNNING ACCRA ROAD SURFACE PIPELINE
echo ============================================

python accra_sentinel2_road_surface_pipeline.py --project-root . --run-step-1 --run-step-2 --run-step-3 --run-step-4 --run-step-5-6 --run-step-8

echo ============================================
echo PIPELINE FINISHED
echo OUTPUTS ARE IN THE outputs FOLDER
echo ============================================

pause