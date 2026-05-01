@echo off
echo ==========================================
echo FIXED Deep Learning Training Pipeline
echo Training 8 Remaining DL Models
echo ==========================================
echo.
echo Models to train:
echo - CNN-LSTM
echo - CNN-GRU
echo - BiLSTM_Attention
echo - Transformer
echo - Informer
echo - Autoformer
echo - TFT
echo - ST-GCN
echo.
echo Estimated time: 1.5-2 hours
echo.
pause

conda run -n dl-env python scripts/train_dl_models_fixed.py

echo.
echo ==========================================
echo Training Complete!
echo Check outputs/hyderabad/ for results
echo ==========================================
pause
