# AGENTS.md - Air Quality Forecasting Project

## Project Overview
- **Title**: SOTA Air Quality Forecasting & Edge AI Deployment
- **Target**: Multi-city PM2.5 prediction (24/48/72 hour horizons)
- **Cities**: Delhi, Hyderabad, Bengaluru (13 cities total in data/)
- **Deployment**: Raspberry Pi 5 with ONNX Runtime

## Data
- **Source**: Open-Meteo API (hourly air quality + meteorology)
- **Features**: pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, us_aqi
- **Frequency**: Hourly (8784 rows per city = 1 year)
- **Location**: `data/kaggle_dataset/clean_*.csv`

## Compute Platform Recommendations

| Model | Platform | Notes |
|-------|----------|-------|
| ARIMA, VAR | CPU | No GPU benefit |
| XGBoost, LightGBM, CatBoost | CPU | Tree parallel |
| **LSTM, GRU** | **RTX 2050** | **24x speedup** |
| Transformer, Autoformer | Kaggle T4 | Memory heavy |

### Smoke Test Results
```
CPU (i5-12450H):    LSTM = 14.17s
RTX 2050:          LSTM = 0.59s  (24x faster)
```

## Quick Commands

### Data
```bash
# Download fresh data
conda run -n dl-env python scripts/download_india_air_quality.py

# Check data
Get-Content data/kaggle_dataset/clean_delhi_aq_1y.csv -Head 5
```

### Training (One-by-One)
```bash
# CPU models (fast)
conda run -n dl-env python scripts/individual_trainers/statistical_trainers.py --city delhi
conda run -n dl-env python scripts/individual_trainers/ensemble_trainers.py --city delhi

# GPU models (use RTX 2050)
conda run -n dl-env python scripts/individual_trainers/lstm_trainer.py --city delhi --device cuda
conda run -n dl-env python scripts/individual_trainers/sequence_trainers.py --city delhi --device cuda
conda run -n dl-env python scripts/individual_trainers/transformer_trainer.py --city delhi --device cuda
```

### Benchmark
```bash
# Run all models (CPU mode - works)
conda run -n dl-env python data/kaggle_dataset/kaggle_benchmarking_suite.py
```

### Plots for Report
```bash
# Generate comparison plots
conda run -n dl-env python outputs/final_deploy/generate_plots.py
```

### Tests
```bash
conda run -n dl-env python -m pytest -q
```

## Key Files

| Path | Purpose |
|------|---------|
| `src/ingestion/india_aq.py` | Data ingestion from Open-Meteo |
| `scripts/train_city_7day.py` | Train forecasting models |
| `scripts/forecast_city_7day.py` | Generate forecasts |
| `deployment/pi_airflow/` | Airflow + Raspberry Pi deployment |
| `outputs/final_deploy/` | Final model artifacts + plots |

## Important Context

### CUDA Issue
- PyTorch 2.2.2+cu121 has **kernel issues** with RTX 2050 for some operations
- **Use CPU mode** for reliability OR upgrade to PyTorch 2.5+
- Workaround: Set `device="cpu"` in training scripts

### Best Models (from benchmarks)
1. **XGBoost**: Best avg RMSE (56.89), fastest inference
2. CatBoost: Good R2 (0.17 Hyderabad)
3. Autoformer: Best deep learning (91.23 avg RMSE)

### Pi Deployment
- Target: Pi 5 @ 100.111.13.58
- Airflow running in Docker
- ONNX conversion: `scripts/optimize_for_pi.py`

## Graphify (Knowledge Graph)
Mandatory policy:
- Before coding: `graphify query "<task>" --graph graphify-out/graph.json`
- Run at session start: `graphify update .`
- Keep updated: `graphify watch .`

## Current Status
- [x] Data: 3 cities benchmarked
- [x] CPU models: ARIMA, XGBoost, LightGBM, CatBoost, LSTM, Transformer
- [x] GPU training: Working on RTX 2050 (24x speedup)
- [x] Plots: Generated (model_comparison.png, rmse_by_city.png)
- [x] Pi deployment: Ready
- [ ] Report: PLOTS NEEDED - Generate more plots

## What to Do Next (One-by-One)
1. Run GPU training on RTX 2050: LSTM, GRU, Bi-LSTM per city
2. Generate more plots for report (convergence, parity, timeseries)
3. Convert best model to ONNX
4. Deploy to Raspberry Pi
5. Write final report (60-80 pages)

## Contact/Notes
- User: Burak (BTech Final Year)
- Timeline: 2 days
- GPU: RTX 2050 (4GB VRAM)
- Pi: Ready at 100.111.13.58