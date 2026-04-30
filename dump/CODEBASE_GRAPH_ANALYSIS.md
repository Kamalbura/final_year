# Codebase Graph Analysis (Graphify)

**Generated**: 2026-04-26  
**Analysis Tool**: Graphify v0.5.0  
**Scope**: final_year project

---

## 📊 Graph Metrics

| Metric | Value |
|--------|-------|
| **Total Files** | 386 |
| **Total Nodes** | 5,137 |
| **Total Edges** | 16,230 |
| **Communities Detected** | 76 |
| **Extraction Confidence** | 40% EXTRACTED / 60% INFERRED |
| **Code Words** | ~511,591 |

**⚠️ Important Note**: Graph is heavily weighted toward embedded tooling (`deployment/graphify/` and `deployment/graphiti/`) which consume ~40% of the graph. Core product code is in `src/`, `scripts/`, and `deployment/pi_airflow/`.

---

## 🎯 Core Product Architecture

### **Top-Level Dependency Map**

```
final_year/
├── src/                          (Core product logic)
│   ├── ingestion/                (Data ingestion pipeline)
│   ├── models/                   (ML models: transformers, etc.)
│   ├── data/                     (Dataset & preprocessing)
│   ├── evaluation/               (Model metrics & evaluation)
│   ├── integrations/             (ThingSpeaks, external APIs)
│   └── utils/                    (Helpers: seeding, etc.)
├── scripts/                      (Operational scripts)
│   ├── fetch_hyderabad_station_data.py
│   ├── download_india_air_quality.py
│   ├── bootstrap_india_aq_db.py
│   ├── forecast_city_7day.py
│   ├── train_city_7day.py
│   └── monitor_and_retrain.py
├── deployment/                   (Deployment & orchestration)
│   ├── pi_airflow/              (Airflow DAG factory)
│   ├── graphify/                (⚠️ Embedded external tool)
│   └── graphiti/                (⚠️ Embedded external tool)
├── web/dashboard/               (Next.js frontend + API routes)
├── tests/                        (Unit & integration tests)
└── Phase notebooks/             (Exploration & research)
```

---

## 🔧 Core Module Breakdown

### **1. Data Ingestion (`src/ingestion/`)**

**Key Node**: `india_aq.py` — 400+ functions and dependencies

**Purpose**: Open-Meteo API integration + PostgreSQL lifecycle management

**Key Classes**:
- `IngestionSettings` — Configuration dataclass with DSN, timeouts, retry logic
- Connection protocol — PostgreSQL cursor abstraction

**Database Schema**:
- `aq.stations` — Metadata for monitoring stations
- `aq.observations` — Time-series PM2.5, PM10, O₃, NO₂, SO₂, CO, AQI
- `aq.watermark` — Incremental ingestion tracking (city_id, last_timestamp)
- `aq.archive` — Historical observations (time-partitioned)

**Key Functions**:
- `run_incremental_cycle_for_cities()` — Main ingestion loop
- `archive_stale_observations()` — Partition management
- `log_run()` — Track run history & failures

**Dependencies**:
- `pandas` — Data manipulation
- PostgreSQL adapter (psycopg2/asyncpg expected)
- `src.data.cities` — City/slug definitions

---

### **2. Models (`src/models/transformers.py`)**

**Core Purpose**: Attention-based sequence forecasting

**Key Components**:

1. **Positional Encoding**
   - `SinusoidalPositionalEncoding` — Absolute position awareness
   - Uses sine/cosine waves for temporal positions

2. **Transformer Variants**
   - `TransformerForecaster` — Base transformer with encoder/decoder
   - `RTTransformerForecaster` — Reverse-time transformer (bidirectional context)
   - Standard PyTorch `nn.TransformerEncoder`, `nn.TransformerDecoder`

3. **Architecture**
   - Input: (batch, seq_len, features)
   - Output: (batch, forecast_horizon, features)
   - Supports variable sequence lengths

**Key Hyperparameters**:
- `d_model` — Embedding dimension
- `nhead` — Attention heads
- `num_layers` — Encoder/decoder depth
- `dropout` — Regularization
- `activation` — ReLU/GELU

**Usage Pattern**:
```python
model = TransformerForecaster(...)
outputs = model(X_train, y_train)  # Training
forecasts = model.forecast(X_test, horizon=24)  # Inference
```

---

### **3. Data Pipeline (`src/data/`)**

**Files**:
- `dataset.py` — Dataset builders, chronological split, scaler selection
- `cities.py` — City definitions, slug mapping, static metadata
- `live_air_quality.py` — Real-time data fetching
- `preprocess.py` — Normalization, imputation, alignment

**Key Classes**:
- `City` — City metadata (slug, coordinates, country)
- `StandardScaler`/`RobustScaler`/`MinMaxScaler` — Feature scaling

**Key Functions**:
- `chronological_split()` — Train/val/test split (time-aware, no leakage)
- `load_config()` — YAML configuration loading
- `build_datasets()` — Full pipeline orchestration

---

### **4. Evaluation (`src/evaluation/metrics.py`)**

**Supported Metrics**:
- `mape()` — Mean Absolute Percentage Error
- `rmse()` — Root Mean Squared Error
- `mae()` — Mean Absolute Error
- `r2()` — Coefficient of Determination

**Integration**: Called from model training & inference notebooks

---

### **5. Integrations (`src/integrations/thingspeak.py`)**

**Purpose**: Send forecasts to ThingSpeak IoT platform

**Key Functions**:
- `push_to_thingspeak()` — HTTP POST to ThingSpeak API
- Handles channel configuration, API keys, rate limiting

---

## 📜 Operational Scripts

| Script | Purpose | Entry Point |
|--------|---------|-------------|
| `fetch_hyderabad_station_data.py` | Download Hyderabad station observations | Station-specific ingestion |
| `download_india_air_quality.py` | Bulk download all city data | Historical data bootstrap |
| `bootstrap_india_aq_db.py` | Initialize PostgreSQL schema | One-time setup |
| `forecast_city_7day.py` | Generate 7-day forecast for city | Inference pipeline |
| `train_city_7day.py` | Retrain model for city | Model retraining |
| `monitor_and_retrain.py` | Automated retraining loop | Scheduled background task |
| `pi_runtime_loop.py` | Raspberry Pi runtime orchestration | Edge deployment |
| `sync_india_air_quality_to_thingspeak.py` | Push observations to ThingSpeak | Cloud sync |

---

## 🔄 Deployment & Orchestration

### **Airflow DAG Factory (`deployment/pi_airflow/dags/aq_city_factory.py`)**

**Purpose**: Generate one Airflow DAG per city for parallelized scheduling

**Architecture**:
```
for each city:
    _build_city_dag(city) →
        Task 1: Fetch observations (fetch_hyderabad_station_data.py)
        Task 2: Preprocess (src/data/preprocess.py)
        Task 3: Train/Retrain (src/models/transformers.py)
        Task 4: Forecast (forecast_city_7day.py)
        Task 5: Push to ThingSpeak (sync_india_air_quality_to_thingspeak.py)
        Task 6: Log results (src/evaluation/metrics.py)
```

**DAG ID**: `aq_{city_slug}` (e.g., `aq_hyderabad`, `aq_delhi`)

**Scheduling**: Configurable per city (daily, weekly, etc.)

---

## 🌐 Web Dashboard

### **Next.js API Routes (`web/dashboard/app/api/`)**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/status` | GET | System health + last run timestamp |
| `/graph` | GET | DAG lineage visualization |
| `/observations` | GET | Ranked observations by AQI |
| `/observations/[citySlug]` | GET | City-specific time-series |
| `/dag-alerts` | GET | Airflow alerts + failures |

**Response Format**:
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2026-04-26T10:30:00Z"
}
```

---

## 🌳 Community Structure (Graphify Analysis)

### **High-Cohesion Communities** (Product Code)

| Community | Nodes | Key Files | Purpose |
|-----------|-------|-----------|---------|
| **Community 16** | 43 | `aq_city_factory.py` | **Airflow DAG generation** |
| **Community 10** | 102 | `analyze_india_aq.py` | India AQ data analysis |
| **Community 9** | 124 | `cities.py`, `live_air_quality.py` | City metadata & live fetch |

### **Low-Cohesion Communities** (Embedded Tools)

| Community | Nodes | Source | Note |
|-----------|-------|--------|------|
| **Community 0-2** | 254-489 | Graphiti core | ⚠️ Large dependency |
| **Community 3-5** | 328-340 | Graphiti LLM clients | ⚠️ External tool |

---

## 🔗 Core Dependency Graph

```
Training Pipeline:
├── src/data/dataset.py         (Load & preprocess)
│   └── src/data/cities.py      (City metadata)
├── src/models/transformers.py  (Model definition)
├── src/evaluation/metrics.py   (Performance tracking)
└── train_city_7day.py          (Training orchestration)

Ingestion Pipeline:
├── src/ingestion/india_aq.py   (DB + API integration)
│   └── src/data/cities.py      (City slugs)
├── scripts/fetch_hyderabad_station_data.py
└── scripts/bootstrap_india_aq_db.py

Forecasting Pipeline:
├── src/models/transformers.py
├── forecast_city_7day.py
├── src/integrations/thingspeak.py
└── web/dashboard/app/api/observations.js

Orchestration:
└── deployment/pi_airflow/dags/aq_city_factory.py
    └── [All above scripts + modules]
```

---

## 📌 God Nodes (Most Connected)

**From Graphify Analysis** (filtered to core code):

1. **`execute_query()`** — 420 edges (Graphiti DB query execution)
2. **`EntityNode`** — 309 edges (Graphiti knowledge graph entity)
3. **`GET()`** — 256 edges (HTTP verb in APIs)
4. **`run()`** — 154 edges (Airflow task execution)
5. **`TransformerForecaster`** — Core forecasting model (inferred 100+ edges)
6. **`IngestionSettings`** — Configuration hub (50+ edges)
7. **`City`** — City metadata hub (80+ edges across scripts)

---

## 🚨 Architecture Issues Detected

### **Graph-Based Findings**

| Issue | Severity | Impact | Recommendation |
|-------|----------|--------|-----------------|
| **Embedded tools bloat** | HIGH | ~40% of graph noise | Remove `deployment/graphify/` & `graphiti/` |
| **Phase notebook duplication** | MEDIUM | 13 redundant notebooks | Archive to `/archive/notebooks/` |
| **Duplicate data files** | MEDIUM | 3x storage for same data | Keep only latest version |
| **Legacy models** | LOW | `best_rf.pkl`, `best_xgboost.pkl` | Document or archive |

---

## 🎓 Data Flow (Chronological)

```
1. [Scheduler] → dispatch(city) to Airflow
2. [Fetch] → open-meteo API → raw observations
3. [Store] → PostgreSQL watermark + observations table
4. [Preprocess] → StandardScaler, chronological split
5. [Train] → TransformerForecaster on (X_train, y_train)
6. [Evaluate] → RMSE, MAE, MAPE on validation set
7. [Forecast] → 24-168h horizon on test set
8. [Push] → ThingSpeak API + Dashboard DB
9. [Alert] → Airflow dag-alerts on failure
```

---

## ✅ Verification Checklist

- [x] **Graphify graph generated** successfully (5,137 nodes, 16,230 edges)
- [x] **Core modules identified** (src/, scripts/, deployment/pi_airflow/)
- [x] **Community detection** reveals 76 communities (3 high-cohesion product clusters)
- [x] **Dependency tracing** complete (city → fetch → train → forecast → push)
- [ ] **Embedded tool removal** — blocks 20-30MB cleanup
- [ ] **Test coverage** — check if tests/ reflect graph structure
- [ ] **Documentation** — ensure README matches graph findings

---

## 🔍 Next Steps

1. **Clean embedded tools** → Remove `deployment/graphify/` and `deployment/graphiti/`
2. **Archive notebooks** → Move phase notebooks to `/archive/`
3. **Consolidate data** → Keep only latest CSV and model files
4. **Update `.gitignore`** → Ensure cache dirs are ignored
5. **Regenerate graph** → Run Graphify again after cleanup to verify improvements

