# `src/aq_forecast` — model library

Merged in from the former `final-yesr-project` repository. It is kept as a
self-contained package rather than dissolved into `src/`, because it is a
**different layer of the same problem** and folding the two together would have
meant duplicating work that already exists on both sides.

## How this relates to the rest of `src/`

| Concern | Lives in | Notes |
|---|---|---|
| Ingestion, DB, deployment | `src/ingestion`, `src/data`, `src/integrations` | Open-Meteo → PostgreSQL platform path |
| Production encoder | `src/models/transformers.py` | Robust, standalone, no config coupling; used by the deployed pipeline |
| Regression metrics | `src/evaluation/metrics.py` | RMSE, MAE, R², MASE |
| **Model zoo + tuning** | **`src/aq_forecast/`** | **This package** |

Nothing was overwritten. The two transformer implementations coexist deliberately:
`src/models/transformers.py` handles edge cases (odd `model_dim`, sequence overflow)
and takes no config object, while `src/aq_forecast/models/transformer.py` is the
textbook encoder wired to `ModelConfig` for sweep-driven experimentation.
Likewise the two metrics modules differ in kind — the platform one is pure
regression, this one adds AQI-category classification (F1, confusion matrix, MAPE).

## Layout

```
src/aq_forecast/
├── config.py       DataConfig, ModelConfig, TrainConfig, OptunaConfig,
│                   feature lists, AQI_CATEGORIES, model registries
├── aqi.py          get_aqi_category — canonical AQI band mapping
├── data/           dataset, download (AQI computation), preprocessing
├── models/         RNN, LSTM, BiLSTM, +attention variants, TCN,
│                   Transformer, XGBoost, LightGBM, behind model_factory
├── training/       trainer.py, optuna_tuner.py
└── evaluation/      metrics.py (regression + AQI class), visualizer.py
```

## Usage

```python
from src.aq_forecast.config import ModelConfig
from src.aq_forecast.models.model_factory import build_model

model = build_model("BiLSTM_Attention", ModelConfig(), horizon=24)
```

Available names: `RNN`, `LSTM`, `BiLSTM`, `LSTM_Attention`, `BiLSTM_Attention`,
`TCN`, `Transformer`, `XGBoost`, `LightGBM`.

Supporting scripts live in `scripts/aq_forecast/`.

## Changes made during the merge

- Package renamed `air_quality_prediction` → `src.aq_forecast`; all imports rewritten.
- `PROJECT_ROOT` in `config.py` walks up one extra level, so paths still resolve
  to the repository root from the new depth.
- `get_aqi_category` extracted from `data/download.py` into `aqi.py` and imported
  back, so the mapping has exactly one definition and the evaluation layer no
  longer pulls in the download pipeline.

## Data

Seven CPCB monitoring stations around Hyderabad, hourly from January 2023:
Bollaram Industrial Area, Central University Hyderabad, ICRISAT Patancheru,
IDA Pashamylaram, Nacharam, Sanathnagar and Zoo Park. Columns cover PM2.5, PM10,
CO, NO2, SO2, O3, ambient temperature, humidity, wind speed and direction,
rainfall, barometric pressure and solar radiation, with `AQI`, `AQI_Category`
and a land-use `zone` label already computed.

This is a **different dataset** from `data/hyderabad_station_aq_1y/`, which covers
3 stations from December 2024 with 10 columns and no computed AQI. Both are kept.

The files live in `data/raw/` **gzipped**, because the repository ignores `*.csv`
wholesale. The pipeline reads plain `.csv`, so decompress before the first run:

```bash
gunzip -k data/raw/*.csv.gz
```

`all_stations_combined.csv` is not committed — `data/download.py` regenerates it by
concatenating the per-station files. `data/processed/` is gitignored for the same
reason; `scripts/aq_forecast/run_preprocessing.py` rebuilds it.
