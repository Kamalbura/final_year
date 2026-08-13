"""
config.py — Central configuration for the Air Quality Prediction Pipeline.

All hyperparameters, paths, feature definitions, and training settings
are defined here. Optuna overrides specific fields during tuning.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
          CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Feature Definitions
# ──────────────────────────────────────────────────────────────

# Pollutant features from Open-Meteo Air Quality API (CAMS reanalysis)
POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"
]

# Meteorological features (when available)
METEO_FEATURES = [
    "AT",   # Ambient Temperature (°C)
    "RH",   # Relative Humidity (%)
    "WS",   # Wind Speed (m/s)
    "WD",   # Wind Direction (degrees)
    "RF",   # Rainfall (mm)
    "SR",   # Solar Radiation (W/m²)
    "BP",   # Barometric Pressure (mmHg)
]

# Temporal features engineered during preprocessing
TEMPORAL_FEATURES = [
    "hour", "day_of_week", "day_of_month", "month",
    "hour_sin", "hour_cos",           # Cyclical encoding for hour
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "is_weekend",
]

# The target variable for prediction
TARGET = "AQI"

# AQI category breakpoints (India NAQI standard)
AQI_CATEGORIES = {
    "Good":       (0, 50),
    "Satisfactory": (51, 100),
    "Moderate":   (101, 200),
    "Poor":       (201, 300),
    "Very Poor":  (301, 400),
    "Severe":     (401, 500),
}


# ──────────────────────────────────────────────────────────────
# Data Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    city: str = "Hyderabad"
    state: str = "Telangana"

    # Train / Validation / Test split ratios (chronological)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Lookback window (how many past timesteps to use as input)
    lookback: int = 72    # 72 hours = 3 days of hourly data

    # Forecast horizon (how many timesteps ahead to predict)
    horizon: int = 24     # Predict 24 hours ahead

    # Missing value imputation strategy: "linear", "knn", "forward_fill"
    imputation_method: str = "linear"

    # Features to use — assembled dynamically based on dataset availability
    use_meteo: bool = True
    use_temporal: bool = True

    # Scaling method: "standard" (z-score) or "minmax"
    scaler_type: str = "standard"


# ──────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Base configuration shared across all models."""
    model_name: str = "LSTM"
    input_dim: int = 0        # Set dynamically based on feature count
    output_dim: int = 1       # Predicting AQI (single value)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Attention-specific
    use_attention: bool = False
    attention_heads: int = 4
    attention_dim: int = 64

    # TCN-specific
    tcn_num_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64])
    tcn_kernel_size: int = 3

    # Transformer-specific
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_encoder_layers: int = 3
    transformer_dim_feedforward: int = 256

    # XGBoost / LightGBM specific
    n_estimators: int = 1000
    max_depth: int = 8
    learning_rate_gb: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


# ──────────────────────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Hyperparameters for training deep learning models."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15              # Early stopping patience
    min_delta: float = 1e-4         # Minimum improvement for early stopping
    scheduler: str = "cosine"       # "cosine", "step", "plateau"
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    gradient_clip: float = 1.0      # Max gradient norm
    seed: int = 42


# ──────────────────────────────────────────────────────────────
# Optuna Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class OptunaConfig:
    """Settings for Optuna hyperparameter search."""
    n_trials: int = 50
    timeout: Optional[int] = 3600   # Max seconds per model tuning
    direction: str = "minimize"     # Minimize validation RMSE
    study_name: str = "aqi_hpo"
    pruner: str = "median"          # "median" or "hyperband"
    sampler: str = "tpe"            # "tpe" or "cmaes"


# ──────────────────────────────────────────────────────────────
# Registry of all model names in the pipeline
# ──────────────────────────────────────────────────────────────
ALL_MODELS = [
    "RNN",
    "LSTM",
    "BiLSTM",
    "LSTM_Attention",
    "BiLSTM_Attention",
    "TCN",
    "Transformer",
    "XGBoost",
    "LightGBM",
]

# Deep-learning models (use PyTorch training loop)
DL_MODELS = [m for m in ALL_MODELS if m not in ("XGBoost", "LightGBM")]

# Gradient-boosting models (use sklearn-style fit/predict)
GB_MODELS = ["XGBoost", "LightGBM"]
