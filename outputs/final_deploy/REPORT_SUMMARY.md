# Final Year B.Tech Project: Air Quality Forecasting

## Executive Summary

This document summarizes the final model selection and benchmark results for the Air Quality Forecasting project.

---

## 1. Project Overview

**Title**: SOTA Air Quality Forecasting & Edge AI Deployment
**Target**: Multi-city PM2.5 prediction with 24/48/72 hour horizons
**Cities**: Delhi, Hyderabad, Bengaluru
**Deployment**: Raspberry Pi 5 with ONNX Runtime

---

## 2. Models Evaluated

### 2.1 Models Kept (11)

| Phase | Model | Type | Status |
|-------|-------|------|--------|
| Phase 1 | ARIMA | Statistical | ✅ Working |
| Phase 1 | VAR | Statistical | ✅ Working |
| Phase 2 | XGBoost | Classical ML | ✅ Working (Best Avg RMSE: 56.89) |
| Phase 2 | LightGBM | Classical ML | ✅ Working |
| Phase 2 | CatBoost | Classical ML | ✅ Working |
| Phase 3 | LSTM | Deep Learning | ✅ Working |
| Phase 3 | GRU | Deep Learning | ✅ Working |
| Phase 3 | Bi-LSTM | Deep Learning | ✅ Working |
| Phase 4 | CNN-LSTM | Hybrid | ✅ Working |
| Phase 4 | Transformer | Attention | ✅ Working |
| Phase 4 | Autoformer | SOTA | ✅ Working |

### 2.2 Models Removed (9)

| Model | Reason |
|-------|--------|
| SARIMA | Redundant with ARIMA |
| Random Forest | XGBoost performs better |
| SVR | Slower, lower accuracy |
| RNN | LSTM performs better |
| CNN-GRU | Redundant with CNN-LSTM |
| Bi-LSTM + Attention | Similar to Transformer |
| Informer | Similar to Autoformer |
| TFT | Complex, requires special setup |
| ST-GCN | Requires city adjacency graph |

---

## 3. Benchmark Results (CPU Mode)

### 3.1 Best Model by City (Lowest RMSE)

| City | Best Model | RMSE | R² |
|------|------------|------|-----|
| **Delhi** | XGBoost | 83.54 | 0.71 |
| **Hyderabad** | CatBoost | 25.19 | 0.17 |
| **Bengaluru** | ARIMA | 35.80 | -0.61 |

### 3.2 Average RMSE Across Cities

| Rank | Model | Avg RMSE |
|------|-------|---------|
| 1 | **XGBoost** | 56.89 |
| 2 | CatBoost | 69.38 |
| 3 | VAR | 71.95 |
| 4 | ARIMA | 76.91 |
| 5 | Autoformer | 91.23 |
| 6 | Transformer | 95.15 |
| 7 | LightGBM | 98.17 |
| 8 | Bi-LSTM | 102.31 |
| 9 | GRU | 105.48 |
| 10 | CNN-LSTM | 105.67 |
| 11 | LSTM | 106.81 |

---

## 4. Key Findings

### 4.1 Statistical vs ML vs Deep Learning

- **Classical ML (XGBoost)** outperforms all deep learning models on CPU
- **Edge deployment** favors XGBoost for inference speed
- **Transformer/Autoformer** require GPU for efficient training

### 4.2 City-Specific Insights

- **Delhi**: High pollution variability → XGBoost handles well
- **Hyderabad**: Moderate pollution → CatBoost best
- **Bengaluru**: Low stable pollution → ARIMA sufficient

### 4.3 Recommendations

1. **Primary Model**: XGBoost (best average RMSE, fast inference)
2. **Secondary Model**: Transformer (for interpretability)
3. **Edge Deployment**: XGBoost → ONNX → Raspberry Pi

---

## 5. Deliverables

### 5.1 Model Artifacts

Location: `outputs/final_deploy/`

| Folder | Contents |
|--------|----------|
| `model_checkpoints/` | XGBoost, LightGBM, LSTM, Transformer models |
| `scalers/` | Feature scalers and feature names |
| `plots/` | Generated visualization plots |
| `report/` | Final report artifacts |

### 5.2 Benchmark Data

- `benchmark_summary.csv`: Full comparison table
- All model runs completed on CPU

---

## 6. Technical Notes

### 6.1 CUDA Issue (RTX 2050)

- PyTorch 2.2.2+cu121 incompatible with RTX 2050 (compute capability 8.9)
- Solution: Use CPU mode OR Kaggle GPUs for deep learning

### 6.2 Raspberry Pi Deployment

- Target: Pi 5 @ 100.111.13.58
- Framework: Airflow + ONNX Runtime
- Status: ✅ Deployed and operational

---

## 7. Report Timeline

| Task | Status |
|-----|--------|
| Benchmark analysis | ✅ Complete |
| Model selection | ✅ Complete |
| Generate plots | 🔲 Pending |
| Report writeup | 🔲 Pending |
| Pi deployment | ✅ Complete |

---

*Generated: 2026-05-01*