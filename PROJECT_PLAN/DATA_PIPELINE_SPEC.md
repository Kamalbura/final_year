# Data Management & Pipeline Specification

## 1. Input Variables (Features)
Following 2024-2025 SOTA standards, the feature set is multi-modal:

### 1.1 Air Quality Indicators
*   **PM2.5 / PM10:** Particulate matter (Target & Feature).
*   **NO2:** Nitrogen Dioxide (Key traffic indicator).
*   **SO2:** Sulfur Dioxide.
*   **CO:** Carbon Monoxide.
*   **O3:** Ozone (Secondary pollutant).

### 1.2 Meteorological Features
*   **T2M:** 2-meter Temperature.
*   **RH:** Relative Humidity.
*   **WS_u / WS_v:** Wind speed vectors (U and V components for drift).
*   **PBLH:** Planetary Boundary Layer Height (Critical for vertical dispersion).
*   **Precip:** Hourly precipitation.

### 1.3 Temporal Features
*   **Periodic:** Sine/Cosine encoding of hour of day and month.
*   **Categorical:** Weekend/Weekday flag.

## 2. Pipeline Stages

### Stage 1: Ingestion & Standardization
*   Convert all units to $\mu g/m^3$ or US AQI standards.
*   Ensure all timestamps are UTC-synchronized.

### Stage 2: SOTA Quality Control (QC)
*   **Outlier Detection:** Implement "Probability of Residuals" method. Compare current observation with spatial neighbors and temporal lags. Flag if residual probability < 0.05.
*   **Sensor Drift Compensation:** Apply non-linear correction factors for high humidity (>90%) interference.

### Stage 3: SOTA Imputation
*   **Small Gaps (< 6h):** Linear or Cubic Spline interpolation.
*   **Large Gaps (6h - 24h):** KNN Imputer based on similar meteorological conditions.
*   **Extreme Gaps (> 24h):** **SAITS** (Self-Attention-based Imputation) using multivariate correlations.

### Stage 4: Scaling & Windowing
*   **Scaling:** Fit `RobustScaler` on Train set ONLY.
*   **Windowing:** 
    *   Lookback Window: 168 hours (7 days).
    *   Forecast Horizon: 24/48/72 hours.

## 3. Data Quality Metrics
*   **Completeness:** % of non-null values per pollutant.
*   **Stationarity:** ADF (Augmented Dickey-Fuller) test results for ARIMA/VAR models.
