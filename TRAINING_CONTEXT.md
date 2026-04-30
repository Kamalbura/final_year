# Air Quality Training Context

Last updated: 2026-04-26

## Objective

Train and benchmark one-year, city-specific time-series forecasting models for Delhi, Hyderabad, and Bengaluru. The final model must be selected by measured performance and by whether it can run reliably on Raspberry Pi 4 ARM, because the project deploys on the Pi dashboard/backend stack.

The immediate target for the dashboard is `predicted_us_aqi`, because `aq.forecasts` and `/api/predictions/[citySlug]` are already wired around AQI forecasts. The input feature policy is intentionally narrower than the raw warehouse:

- Required core inputs: `pm2_5`, `pm10`.
- Allowed extra inputs: up to three pollutant variables with good continuity and deployment value.
- Current recommended five-input set: `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`, `ozone`.
- Exclude `us_aqi` from model inputs for the main benchmark so the model learns from pollutants rather than copying a derived historical target.
- Keep `sulphur_dioxide` as an ablation feature only, not part of the first production pass.

## Local Data Audit

Clean files audited:

- `data/kaggle_dataset/clean_delhi_aq_1y.csv`
- `data/kaggle_dataset/clean_hyderabad_aq_1y.csv`
- `data/kaggle_dataset/clean_bengaluru_aq_1y.csv`

All three have strict hourly coverage:

| City | Rows | Range | Expected hours | Missing timestamps | Duplicate timestamps | Missing pollutant/AQI values |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Delhi | 8784 | 2025-04-25 00:00 UTC to 2026-04-25 23:00 UTC | 8784 | 0 | 0 | 0 |
| Hyderabad | 8784 | 2025-04-25 00:00 UTC to 2026-04-25 23:00 UTC | 8784 | 0 | 0 | 0 |
| Bengaluru | 8784 | 2025-04-25 00:00 UTC to 2026-04-25 23:00 UTC | 8784 | 0 | 0 | 0 |

Important value ranges:

| City | PM2.5 | PM10 | CO | NO2 | O3 | AQI |
| --- | --- | --- | --- | --- | --- | --- |
| Delhi | 6.0-333.4 | 6.1-1408.2 | 185-4984 | 2.0-176.2 | 0-331 | 61-779 |
| Hyderabad | 1.4-119.0 | 1.4-124.3 | 126-2627 | 0-83.6 | 4-218 | 18-195 |
| Bengaluru | 1.0-93.8 | 1.1-104.8 | 109-2269 | 0-149.3 | 0-232 | 15-204 |

The existing preparation pipeline already exposes hidden gaps by resampling to `1h`, then imputes values with:

- Linear interpolation for gaps up to 6 hours.
- Forward fill up to 24 hours.
- Backward fill for dataset start gaps.
- Monthly median fallback.
- Global median fallback.

This is acceptable for the first benchmark because the final clean files have no timestamp loss and no remaining null values. For documentation, every training run must preserve before/after missingness counts and row counts.

## Research Notes

The external literature supports a staged comparison, not a single assumed winner.

- Sparse-attention Transformer work for PM2.5 compares ARIMA, SVR, RF, XGBoost, CNN, LSTM, and Transformer-style models with RMSE, MAE, and R2. It also reports normalized time-series inputs and a fixed historical lookup window. Source: https://link.springer.com/article/10.1007/s13762-023-04900-1
- PM2.5 Transformer vs CNN-LSTM-attention work argues that Transformers can model long-term dependence, but this is not automatically a deployment win for a Pi. Source: https://www.sciencedirect.com/science/article/pii/S1309104223001873
- PM2.5 imputation literature supports interpolation-based handling for short gaps, especially before moving to heavier learned imputers. Source: https://www.mdpi.com/2073-431X/12/8/165
- The time-series Transformer resource list includes Informer and Autoformer as long-sequence forecasting families, but they should remain benchmark candidates, not guaranteed production picks. Source: https://github.com/qingsongedu/time-series-transformers-review
- The broader time-series forecasting resource list shows modern LLM/time-series methods, but these are too large for our first Pi deployment pass. Source: https://github.com/TongjiFinLab/awesome-time-series-forecasting
- General ML/DL awesome lists confirm the ecosystem choices we are already using: scikit-learn, PyTorch, TensorFlow/LiteRT, and ONNX-style deployment paths. Sources: https://github.com/josephmisiti/awesome-machine-learning and https://github.com/ChristosChristofidis/awesome-deep-learning

## Raspberry Pi 4 Deployment Constraints

The winner cannot be chosen by Kaggle GPU accuracy alone. It must have an ARM inference path.

- PyTorch official Raspberry Pi tutorial says PyTorch pip packages are for ARM 64-bit/aarch64 and require a 64-bit Pi OS. Source: https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html
- Google LiteRT/TFLite runtime provides prebuilt wheels for Raspberry Pi ARM32 and ARM64, and is much smaller than full TensorFlow. Source: https://ai.google.dev/edge/litert/guide/python
- ONNX Runtime supports ARM64/ARM32 in its getting-started architecture selector, so ONNX export is a possible Pi inference route. Source: https://onnxruntime.ai/getting-started
- scikit-learn documentation warns that Raspberry Pi installs can accidentally compile NumPy/SciPy from source; use binary wheels or conda/apt packaging. Source: https://scikit-learn.org/stable/install.html

Production preference order for Pi:

1. Small PyTorch sequence model exported with a pinned ARM64 PyTorch environment, or converted to ONNX/TFLite if practical.
2. Small scikit-learn Random Forest if Pi inference latency and dependency install are acceptable.
3. XGBoost/LightGBM/CatBoost only if ARM install and runtime are verified on the Pi, not merely because Kaggle scores are good.
4. Large Transformers are research/comparison models unless their parameter count, latency, and runtime stack are proven on Pi.

## Current Model Budget

Current deep models in `scripts/kaggle_benchmarking_suite.py` use `hidden=48`. With the recommended five input features and a 168-hour forecast horizon, parameter counts are:

| Model | Approx parameters |
| --- | ---: |
| RNN | 15,576 |
| LSTM | 37,608 |
| GRU | 30,264 |
| Bi-LSTM | 93,480 |
| CNN-LSTM | 46,408 |
| CNN-GRU | 37,768 |
| Bi-LSTM + Attention | 93,577 |
| Transformer Lite | 78,248 |
| Informer Lite | 78,328 |
| Autoformer Lite | 78,248 |
| TFT Lite | 30,552 |
| ST-GCN Lite | 18,312 |

These are small enough to benchmark. The production shortlist should probably come from GRU, LSTM, CNN-GRU, TFT Lite, Random Forest, or ST-GCN Lite unless the larger models clearly outperform and pass Pi inference checks.

## Training Protocol

Run city-specific training, not pooled training:

- Delhi model trained only on Delhi data.
- Hyderabad model trained only on Hyderabad data.
- Bengaluru model trained only on Bengaluru data.

Primary split:

- Chronological split only.
- Train 70 percent, validation 15 percent, test 15 percent.
- No random train/test split for time-series evaluation.

Initial benchmark settings:

- Lookback: 168 hours.
- Horizon: 24 hours for the first serious pass.
- After 24-hour results are stable, optionally run 168-hour horizon as a long-horizon research pass.
- Metrics: RMSE, MAE, R2, training seconds, inference seconds.
- Save artifact, scaler, feature list, lookback, horizon, and training config per city/model.

Missing-data safeguards:

- Verify row counts before and after imputation.
- Verify timestamp continuity after `1h` resampling.
- Verify null counts by column before and after filling.
- Fail the run if any selected input or target column still has nulls.
- Keep an audit JSON/CSV per city so the paper can show data was not silently lost.

## Kaggle Status

Kaggle read/auth evidence:

- `scripts/kaggle_auth_check.ps1` succeeds and prints Kaggle CLI config for user `kamalbura`.
- Kaggle MCP can read the `kamalbura` user profile.

Kaggle write/run blockers found on 2026-04-26:

- Dataset create/upload failed at `BlobApiService/StartBlobUpload` with `401 Unauthorized`.
- Kaggle CLI `kernels list --user kamalbura` failed with `401 Unauthorized`.
- Kaggle MCP `save_notebook` returned `Unauthenticated`.

Kaggle recovery update:

- Classic `kaggle.json` works for private kernel/dataset read operations when `access_token` is temporarily hidden.
- The KGAT access-token path can upload blobs but still cannot list private kernels from this Codex session; MCP still returns `Unauthenticated` until the host process is restarted with the corrected token.
- The private dataset `kamalbura/indian-air-quality-3-city-benchmark-1y` was created and reached `ready`.
- Updating the existing notebook `kamalbura/aqi-final-year` through CLI currently hits Kaggle's known `Kernel push error: Notebook not found` edge case.
- A fresh GPU script kernel was created successfully at `https://www.kaggle.com/code/kamalbura/aqi-gpu-benchmark-3-city-model-zoo`.
- Versions 1 and 2 failed because Kaggle mounted the attached dataset under `/kaggle/input/datasets` instead of the expected direct slug folder. Version 3 adds recursive input discovery and is running.

Conclusion: Kaggle GPU training is now started through the fresh `aqi-gpu-benchmark-3-city-model-zoo` kernel. Keep using classic-auth CLI for push/status/output in this session; use MCP only after restarting Codex/IDE so it can inherit the corrected KGAT token.

## Implementation Checkpoint

Completed after the data/research context pass:

- `scripts/kaggle_benchmarking_suite.py` now uses the five-input pollutant feature policy for supervised ML/DL models: `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`, and `ozone`.
- `us_aqi` is preserved as the target only. It is no longer a supervised model input, preventing the benchmark from leaning on a derived historical AQI feature.
- VAR keeps an explicit endogenous set of the five pollutant variables plus `us_aqi`, so the statistical multivariate baseline can still forecast the target without breaking the supervised feature policy.
- `scripts/prepare_kaggle_data.py` writes `benchmark_manifest.json` with row counts, timestamp coverage, duplicate counts, and before/after missingness for each city.
- The Kaggle runner now uses the first serious benchmark horizon: 168-hour lookback and 24-hour forecast horizon.
- A lightweight local verification confirmed Delhi supervised tensors shape as `(10, 168, 5)` inputs and `(10, 24)` targets, with `target_in_features=False`.

## Ingestion Pipeline Implementation

Completed:

- `scripts/load_forecasts_to_db.py` — Loads forecast_rows.csv into aq.forecasts table with upsert logic (ON CONFLICT DO UPDATE).
- `scripts/orchestrate_benchmark_ingestion.py` — End-to-end orchestration: polls Kaggle kernel status, downloads outputs, loads forecasts, validates dashboard API.

Usage:
```bash
# Monitor kernel and orchestrate ingestion (waits for kernel to complete)
python scripts/orchestrate_benchmark_ingestion.py \
  --kernel kamalbura/aqi-gpu-benchmark-3-city-model-zoo \
  --output-dir outputs/kaggle_benchmarks \
  --dashboard-url http://localhost:3000

# Or skip polling if kernel is already done
python scripts/orchestrate_benchmark_ingestion.py --skip-poll
```

## Kaggle Kernel v3 Results (2026-04-27)

Kernel `kamalbura/aqi-gpu-benchmark-3-city-model-zoo` completed with status `COMPLETE`.

### Critical Issue: CUDA `cudaErrorNoKernelImageForDevice`
All PyTorch deep learning models (Phases 3–5: RNN, LSTM, GRU, Bi-LSTM, CNN-LSTM, CNN-GRU, Bi-LSTM+Attention, Transformer, Informer, Autoformer, TFT, ST-GCN) were **skipped** due to:
```
CUDA error: no kernel image is available for execution on the device
```
This is a PyTorch/CUDA version mismatch on the Kaggle GPU. The installed PyTorch was compiled for a different CUDA compute capability than the assigned GPU. Fix: pin `torch` to a version matching the Kaggle GPU (T4 = sm_75) or add `--extra-index-url https://download.pytorch.org/whl/cu118` in the kernel setup cell.

### Benchmark Results (Classical ML only — valid)

| City | Best Model | RMSE | MAE | R² | Confidence |
|------|-----------|------|-----|-----|------------|
| Delhi | LightGBM | 73.45 | 44.85 | 0.650 | 0.577 |
| Hyderabad | XGBoost | 15.29 | 10.63 | 0.587 | 0.867 |
| Bengaluru | CatBoost | 24.08 | 15.52 | 0.278 | 0.806 |

All statistical baselines (ARIMA, SARIMA, VAR) had negative R² — confirming they cannot model the non-stationary AQI signal.

### Forecast Rows Validated
- 72 rows total (24 per city), stored at `outputs/kaggle_benchmarks/forecast_rows.csv`
- Delhi: LightGBM, AQI range [442.6, 595.5] (high pollution season)
- Hyderabad: XGBoost, AQI range [70.5, 146.3]
- Bengaluru: CatBoost, AQI range [82.2, 101.7]

### MCP Configuration
Workspace MCP servers configured at `.kiro/settings/mcp.json`:
- `kaggle`: Kaggle MCP for kernel/dataset operations (uses `KAGGLE_KEY` env var)
- `fetch`: HTTP fetch for web resources
- `filesystem`: Read access to `outputs/`, `data/`, `logs/`

## Next Actions

1. ✓ Benchmark suite created and data pipeline validated.
2. ✓ Forecast ingestion layer implemented.
3. ✓ Kaggle kernel v3 completed — classical ML results pulled and validated.
4. ✓ MCP servers configured (workspace `.kiro/settings/mcp.json`).
5. **Fix CUDA kernel image error** — create kernel v4 with explicit PyTorch CUDA wheel pinned to T4 (sm_75). Add to kernel setup: `pip install torch --index-url https://download.pytorch.org/whl/cu118`.
6. **Run `python scripts/orchestrate_benchmark_ingestion.py --skip-poll`** to load the 72 forecast rows into the database.
7. Verify dashboard predictions at `/api/predictions/[citySlug]` show model-backed forecasts.
8. After DL models run successfully, compare against classical ML winners and select production model per city.
9. Run Pi ARM inference validation for the winning model family.
10. Start the documentation/book pass after DL benchmark results are confirmed.
