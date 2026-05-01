@echo off
REM Run unified training pipeline for all 19 models
REM Usage: run_training_full.bat

echo ==========================================
echo Unified Training Pipeline - All 19 Models
echo City: Hyderabad
echo Compute: RTX 2050 GPU
echo ==========================================
echo.

conda run -n dl-env python scripts/unified_training_pipeline.py

echo.
echo ==========================================
echo Training Complete!
echo Results in: outputs\hyderabad\
echo ==========================================
pause
