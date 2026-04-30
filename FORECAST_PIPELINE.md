# Forecast Pipeline Guide

This document describes the complete pipeline for training benchmark models on Kaggle, downloading forecasts, and loading them into the dashboard.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Kaggle GPU Benchmark Kernel                        │
│  (aqi-gpu-benchmark-3-city-model-zoo)                               │
│  • Loads 3 cities (Delhi, Hyderabad, Bengaluru)                      │
│  • Trains 20 model variants (ARIMA, RF, LSTM, Transformer, etc.)     │
│  • Outputs: benchmark_summary.csv, forecast_rows.csv, artifacts      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│         orchestrate_benchmark_ingestion.py                          │
│  1. Polls kernel status (--poll-interval 60s, --max-wait 3600s)     │
│  2. Downloads outputs from Kaggle API                                │
│  3. Calls load_forecasts_to_db.py                                    │
│  4. Validates /api/predictions/[citySlug]                            │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│         load_forecasts_to_db.py                                      │
│  • Reads forecast_rows.csv                                           │
│  • Validates required columns                                        │
│  • Upserts into PostgreSQL aq.forecasts table (ON CONFLICT)         │
│  • Returns summary: (city_slug, forecast_count)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│              PostgreSQL aq.forecasts Table                           │
│  • forecast_id (PK)                                                  │
│  • city_slug, city_name                                              │
│  • forecast_timestamp, horizon_timestamp                             │
│  • predicted_us_aqi, model_type, confidence                          │
│  • Indexed on (city_slug, horizon_timestamp DESC)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│            Dashboard API (/api/predictions/[citySlug])               │
│  • Queries past 24h actual from aq.hourly_aggregates                │
│  • Queries forecasts from aq.forecasts (horizon_timestamp >= NOW()) │
│  • Returns timeline: {timestamp, actual_aqi|predicted_aqi}           │
│  • Reports model name, confidence, trend, risk band                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Usage

### 1. Run Kaggle Benchmark (Already Running as of 2026-04-26)

The kernel `kamalbura/aqi-gpu-benchmark-3-city-model-zoo` is deployed on Kaggle and runs the 20-model suite on all 3 cities.

**Local equivalent** (if running locally):
```bash
conda activate dl-env
python scripts/kaggle_benchmarking_suite.py \
  --data-dir data/kaggle_dataset \
  --output-dir outputs/kaggle_benchmarks \
  --lookback 168 \
  --horizon 24 \
  --epochs 6 \
  --device cuda
```

Outputs:
- `outputs/kaggle_benchmarks/benchmark_summary.csv`
- `outputs/kaggle_benchmarks/forecast_rows.csv`
- `outputs/kaggle_benchmarks/benchmarks_results.json`
- City artifact directories: `outputs/kaggle_benchmarks/{city}/*.pt`, `*.joblib`

### 2. Monitor and Download (Orchestrate Pipeline)

Once the Kaggle kernel completes, run:

```bash
conda activate dl-env
python scripts/orchestrate_benchmark_ingestion.py \
  --kernel kamalbura/aqi-gpu-benchmark-3-city-model-zoo \
  --output-dir outputs/kaggle_benchmarks \
  --poll-interval 60 \
  --max-wait 3600 \
  --dashboard-url http://localhost:3000
```

**What it does:**
1. Polls `kaggle kernels status kamalbura/aqi-gpu-benchmark-3-city-model-zoo` every 60 seconds.
2. Once complete, runs `kaggle kernels output kamalbura/aqi-gpu-benchmark-3-city-model-zoo -p outputs/kaggle_benchmarks`.
3. Calls `python scripts/load_forecasts_to_db.py --csv outputs/kaggle_benchmarks/forecast_rows.csv`.
4. Validates that `/api/predictions/delhi`, `/api/predictions/hyderabad`, `/api/predictions/bengaluru` return forecasts.

**Exit codes:**
- `0` = Success
- `1` = Kernel failed, download failed, or load failed

### 3. Manual Load (Skip Polling)

If the kernel has already completed or you're running locally:

```bash
conda activate dl-env
python scripts/load_forecasts_to_db.py \
  --csv outputs/kaggle_benchmarks/forecast_rows.csv \
  --dsn "postgresql://user:pass@host/db"
```

Or use environment variable:
```bash
export DATABASE_URL="postgresql://user:pass@host/db"
python scripts/load_forecasts_to_db.py --csv outputs/kaggle_benchmarks/forecast_rows.csv
```

**Output:**
```json
{
  "loaded_forecast_rows": 744,
  "total_forecasts_in_db": 2232,
  "per_city": [
    {"city": "bengaluru", "count": 744},
    {"city": "delhi", "count": 744},
    {"city": "hyderabad", "count": 744}
  ]
}
```

### 4. Validate Dashboard

Test that the dashboard reads forecasts:

```bash
# Check a city's timeline
curl http://localhost:3000/api/predictions/delhi | jq '.timeline[:3]'

# Sample output:
# [
#   {
#     "timestamp": "2026-04-26T12:00:00Z",
#     "actual_aqi": 145,
#     "predicted_aqi": null,
#     "uncertainty": 0
#   },
#   {
#     "timestamp": "2026-04-27T00:00:00Z",
#     "actual_aqi": null,
#     "predicted_aqi": 138,
#     "uncertainty": 18
#   }
# ]
```

## Forecast Data Schema

### forecast_rows.csv (Benchmark Output)

Columns:
- `forecast_id`: Unique identifier, format: `{city}-{safe_model_name}-{timestamp}-{horizon_index}`
- `city_slug`: `delhi`, `hyderabad`, `bengaluru`
- `city_name`: `Delhi`, `Hyderabad`, `Bengaluru`
- `forecast_timestamp`: ISO timestamp when forecast was made
- `horizon_timestamp`: ISO timestamp of the prediction (future)
- `horizon_hours`: 1–24 (hours ahead)
- `predicted_us_aqi`: Float, the AQI prediction
- `model_type`: `ARIMA`, `Random Forest`, `LSTM`, `Transformer`, etc.
- `model_version`: `kaggle-benchmark-v1`
- `confidence`: Float 0.0–1.0, derived from RMSE

### aq.forecasts Table (PostgreSQL)

```sql
CREATE TABLE aq.forecasts (
    forecast_id TEXT PRIMARY KEY,
    city_slug TEXT NOT NULL,
    city_name TEXT NOT NULL,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    horizon_timestamp TIMESTAMPTZ NOT NULL,
    horizon_hours INT NOT NULL,
    predicted_us_aqi DOUBLE PRECISION NOT NULL,
    predicted_pm2_5 DOUBLE PRECISION,
    predicted_pm10 DOUBLE PRECISION,
    model_type TEXT NOT NULL,
    model_version TEXT,
    confidence DOUBLE PRECISION,
    actual_us_aqi DOUBLE PRECISION,
    actual_pm2_5 DOUBLE PRECISION,
    actual_pm10 DOUBLE PRECISION,
    prediction_error DOUBLE PRECISION,
    absolute_error DOUBLE PRECISION,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_forecasts_city_horizon 
    ON aq.forecasts (city_slug, horizon_timestamp DESC);
```

## Troubleshooting

### Kaggle CLI not found
```bash
pip install kaggle
# Ensure ~/.kaggle/kaggle.json is set up (see TRAINING_CONTEXT.md for auth details)
```

### Kernel status shows "Error"
```bash
# Check detailed error in Kaggle web UI:
# https://www.kaggle.com/code/kamalbura/aqi-gpu-benchmark-3-city-model-zoo/notebooks
```

### Download timeout
Increase `--max-wait`:
```bash
python scripts/orchestrate_benchmark_ingestion.py --max-wait 7200  # 2 hours
```

### Database connection error
```bash
# Verify DATABASE_URL
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

### Dashboard shows "Awaiting Training"
```bash
# Check if forecasts are actually in the database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM aq.forecasts WHERE horizon_timestamp > NOW()"

# Check dashboard API response
curl http://localhost:3000/api/predictions/delhi | jq '.model'
```

## Next Steps After Ingestion

1. **Pi ARM Validation**: Download the best model artifacts and test inference on Raspberry Pi 4.
   ```bash
   python scripts/verify_model_on_pi.py \
     --model outputs/kaggle_benchmarks/delhi/transformer.pt \
     --city delhi
   ```

2. **Documentation/Book**: Once benchmark results are confirmed, start the thesis/documentation pass.

3. **Continuous Retraining**: Set up `monitor_and_retrain.py` to automatically retrain when forecast drift exceeds thresholds.
   ```bash
   conda run -n dl-env python scripts/monitor_and_retrain.py --check-interval 3600
   ```

## References

- Benchmark runner: [scripts/kaggle_benchmarking_suite.py](scripts/kaggle_benchmarking_suite.py)
- Forecast loader: [scripts/load_forecasts_to_db.py](scripts/load_forecasts_to_db.py)
- Orchestrator: [scripts/orchestrate_benchmark_ingestion.py](scripts/orchestrate_benchmark_ingestion.py)
- Dashboard API: [web/dashboard/app/api/predictions/\[citySlug\]/route.js](web/dashboard/app/api/predictions/[citySlug]/route.js)
- Training context: [TRAINING_CONTEXT.md](TRAINING_CONTEXT.md)
