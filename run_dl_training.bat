@echo off
echo ==========================================
echo Deep Learning Training Pipeline
echo RTX 2050 Optimized (4GB VRAM)
echo ==========================================
echo.
echo This will train 12 DL models:
echo - RNN, LSTM, GRU, BiLSTM
echo - CNN-LSTM, CNN-GRU, BiLSTM+Attention
echo - Transformer, Informer, Autoformer, TFT
echo - ST-GCN
echo.
echo Estimated time: 2-3 hours
echo.
echo Options:
echo 1. Train ALL DL models (12 models)
echo 2. Train Phase 3 only (RNN, LSTM, GRU, BiLSTM)
echo 3. Train Phase 4 only (CNN-LSTM, CNN-GRU, BiLSTM_Attention, Transformer, Informer, Autoformer, TFT)
echo 4. Train Phase 5 only (ST-GCN)
echo 5. Train specific model(s)
echo 6. Skip specific model(s)
echo.
choice /C 123456 /M "Select option"

if errorlevel 6 goto skip_specific
if errorlevel 5 goto specific
if errorlevel 4 goto phase5
if errorlevel 3 goto phase4
if errorlevel 2 goto phase3
if errorlevel 1 goto all

:all
echo.
echo Training ALL 12 DL models (2-3 hours estimated)
conda run -n dl-env python scripts/train_dl_models.py
goto end

:phase3
echo.
echo Training Phase 3: Standard DL Sequence Models (4 models)
conda run -n dl-env python scripts/train_dl_models.py --models RNN LSTM GRU BiLSTM
goto end

:phase4
echo.
echo Training Phase 4: Hybrid and Attention (7 models)
conda run -n dl-env python scripts/train_dl_models.py --models CNN-LSTM CNN-GRU BiLSTM_Attention Transformer Informer Autoformer TFT
goto end

:phase5
echo.
echo Training Phase 5: Spatio-Temporal (1 model)
conda run -n dl-env python scripts/train_dl_models.py --models ST-GCN
goto end

:specific
echo.
echo Available models: RNN, LSTM, GRU, BiLSTM, CNN-LSTM, CNN-GRU, BiLSTM_Attention, Transformer, Informer, Autoformer, TFT, ST-GCN
set /p models="Enter model names separated by space: "
conda run -n dl-env python scripts/train_dl_models.py --models %models%
goto end

:skip_specific
echo.
echo Available models: RNN, LSTM, GRU, BiLSTM, CNN-LSTM, CNN-GRU, BiLSTM_Attention, Transformer, Informer, Autoformer, TFT, ST-GCN
set /p skip_models="Enter models to SKIP separated by space: "
conda run -n dl-env python scripts/train_dl_models.py --skip %skip_models%
goto end

:end
echo.
echo ==========================================
echo Training Complete!
echo Check outputs/hyderabad/ for results
echo ==========================================
pause
