#!/bin/bash
# Pi Deployment Script - Deploy all 17 models and run benchmarks
# Run on Raspberry Pi

echo "=============================================="
echo "Pi Deployment & Benchmarking Script"
echo "=============================================="
echo ""

# Create directory structure
echo "[1/5] Creating directory structure..."
mkdir -p ~/projects/final_year/deployment_models/hyderabad/{statistical,classical_ml,dl_sequence,dl_hybrid,transformers,spatiotemporal}

echo "✓ Directories created"
echo ""

# Check available disk space
echo "[2/5] Checking resources..."
df -h /dev/mmcblk0p2 | tail -1
free -h | grep "Mem:"
echo ""

# List models to deploy
echo "[3/5] Models ready for deployment:"
echo "  Statistical (3): ARIMA, SARIMA, VAR"
echo "  Classical ML (4): SVR, Random_Forest, XGBoost, LightGBM"
echo "  DL Sequence (4): RNN, LSTM, GRU, BiLSTM"
echo "  DL Hybrid (3): CNN-LSTM, CNN-GRU, BiLSTM_Attention"
echo "  Transformers (3): Transformer, Informer, Autoformer"
echo "  Spatio-Temporal (1): ST-GCN"
echo "  TOTAL: 17 models"
echo ""

# Note: Models need to be copied from PC using scp
echo "[4/5] Next steps:"
echo "  1. Copy models from PC:"
echo "     scp -r outputs/hyderabad/*/ bura@100.111.13.58:~/projects/final_year/deployment_models/hyderabad/"
echo ""
echo "  2. Run benchmark script:"
echo "     python3 pi_benchmark.py"
echo ""
echo "  3. Results will be saved to:"
echo "     ~/projects/final_year/deployment_models/benchmark_results.json"
echo ""

echo "=============================================="
echo "Ready for deployment!"
echo "=============================================="
