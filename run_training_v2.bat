@echo off
echo ==========================================
echo FIXED Training Pipeline v2
echo ==========================================
echo.
echo Options:
echo 1. Train all models (skip CatBoost - RECOMMENDED)
echo 2. Train all models (include CatBoost)
echo 3. Train specific models only
echo 4. Train with pauses between models
echo.
choice /C 1234 /M "Select option"

if errorlevel 4 goto pause_mode
if errorlevel 3 goto specific
if errorlevel 2 goto all_with_catboost
if errorlevel 1 goto skip_catboost

:skip_catboost
echo.
echo Training all models EXCEPT CatBoost (RECOMMENDED)
echo This will take ~30-45 minutes
conda run -n dl-env python scripts/unified_training_pipeline_v2.py --skip-catboost
goto end

:all_with_catboost
echo.
echo Training ALL models including CatBoost
echo WARNING: CatBoost may take 1+ hours!
conda run -n dl-env python scripts/unified_training_pipeline_v2.py
goto end

:specific
echo.
echo Training specific models...
echo Available: ARIMA, SARIMA, VAR, SVR, Random_Forest, XGBoost, LightGBM, CatBoost
set /p models="Enter model names separated by space: "
conda run -n dl-env python scripts/unified_training_pipeline_v2.py --models %models%
goto end

:pause_mode
echo.
echo Training with pauses between models...
conda run -n dl-env python scripts/unified_training_pipeline_v2.py --skip-catboost --pause
goto end

:end
echo.
echo ==========================================
echo Training Complete!
echo Check outputs/hyderabad/ for results
echo ==========================================
pause
