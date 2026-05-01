#!/bin/bash
# SCP Transfer Script - Copy all model artifacts from PC to Pi
# Run this on your Windows PC (Git Bash or WSL)

echo "=============================================="
echo "Model Transfer to Raspberry Pi"
echo "=============================================="
echo ""

PI_USER="bura"
PI_IP="100.111.13.58"
PI_PATH="~/projects/final_year/deployment_models/hyderabad"
LOCAL_PATH="outputs/hyderabad"

# Create remote directory structure
echo "[1/3] Creating remote directory structure..."
ssh ${PI_USER}@${PI_IP} "mkdir -p ${PI_PATH}/{statistical,classical_ml,dl_sequence,dl_hybrid,transformers,spatiotemporal}"

# Define model categories
declare -A MODEL_CATEGORIES=(
    ["ARIMA"]="statistical"
    ["SARIMA"]="statistical"
    ["VAR"]="statistical"
    ["Random_Forest"]="classical_ml"
    ["XGBoost"]="classical_ml"
    ["LightGBM"]="classical_ml"
    ["SVR"]="classical_ml"
    ["RNN"]="dl_sequence"
    ["LSTM"]="dl_sequence"
    ["GRU"]="dl_sequence"
    ["BiLSTM"]="dl_sequence"
    ["CNN-LSTM"]="dl_hybrid"
    ["CNN-GRU"]="dl_hybrid"
    ["BiLSTM_Attention"]="dl_hybrid"
    ["Transformer"]="transformers"
    ["Informer"]="transformers"
    ["Autoformer"]="transformers"
    ["ST-GCN"]="spatiotemporal"
)

echo "[2/3] Copying model artifacts..."
echo ""

# Copy each model
for model in "${!MODEL_CATEGORIES[@]}"; do
    category="${MODEL_CATEGORIES[$model]}"
    
    echo "  → $model (${category})"
    
    # Create model directory on Pi
    ssh ${PI_USER}@${PI_IP} "mkdir -p ${PI_PATH}/${category}/${model}"
    
    # Copy model files
    scp -r "${LOCAL_PATH}/${model}/"* ${PI_USER}@${PI_IP}:${PI_PATH}/${category}/${model}/ 2>/dev/null || echo "    Warning: Some files may not exist"
done

echo ""
echo "[3/3] Verifying transfer..."
ssh ${PI_USER}@${PI_IP} "find ${PI_PATH} -name '*.pth' -o -name '*.joblib' | wc -l && echo 'models transferred'"

echo ""
echo "=============================================="
echo "Transfer Complete!"
echo "=============================================="
echo ""
echo "Next steps on Pi:"
echo "  1. SSH to Pi: ssh ${PI_USER}@${PI_IP}"
echo "  2. Run benchmark: cd ~/projects/final_year && python3 scripts/pi_benchmark.py"
echo "  3. View results: cat deployment_models/benchmark_results.json"
