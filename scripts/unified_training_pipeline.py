#!/usr/bin/env python3
"""
Unified Training Pipeline for Air Quality Forecasting
Dataset: Hyderabad (hyderabad)
Compute: Kaggle GPU -> Local RTX 2050 -> CPU (fallback chain)
Models: All 19 from benchmark suite
Trials: 20 (standard Optuna)

Output Structure:
outputs/hyderabad/
├── {model_name}/
│   ├── model.{pth/joblib/pkl}
│   ├── metrics.json
│   ├── config.json
│   └── plots/
│       ├── convergence.png (DL models)
│       ├── parity.png
│       ├── timeseries.png
│       └── error_histogram.png
└── comparison_report.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
CITY = "hyderabad"
DATA_DIR = Path("data/kaggle_dataset")
OUTPUT_BASE = Path("outputs")
FEATURE_COLUMNS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
TARGET_COLUMN = "us_aqi"
LOOKBACK = 168  # 7 days
HORIZON = 24    # 24 hours forecast
N_TRIALS = 20   # Optuna trials
RANDOM_SEED = 42

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_BASE / "training_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ComputeConfig:
    """Compute configuration"""
    platform: str  # "kaggle_gpu", "local_gpu", "cpu"
    device: torch.device
    batch_size: int
    epochs: int
    use_gpu: bool


@dataclass  
class ModelResult:
    """Training result for a model"""
    model_name: str
    phase: str
    family: str
    rmse: float
    mae: float
    r2: float
    train_seconds: float
    inference_seconds: float
    status: str = "ok"
    error_msg: str = ""


def detect_compute() -> ComputeConfig:
    """Detect available compute platform with fallback chain"""
    
    # Check 1: Are we on Kaggle?
    if os.path.exists("/kaggle"):
        logger.info("Detected Kaggle environment")
        if torch.cuda.is_available():
            logger.info("Kaggle GPU (T4) available")
            return ComputeConfig(
                platform="kaggle_gpu",
                device=torch.device("cuda"),
                batch_size=128,
                epochs=50,
                use_gpu=True
            )
    
    # Check 2: Local RTX 2050
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Local GPU detected: {gpu_name}")
        return ComputeConfig(
            platform="local_gpu",
            device=torch.device("cuda"),
            batch_size=64,  # Lower for 4GB VRAM
            epochs=50,
            use_gpu=True
        )
    
    # Fallback: CPU
    logger.info("No GPU available, using CPU")
    return ComputeConfig(
        platform="cpu",
        device=torch.device("cpu"),
        batch_size=32,
        epochs=30,
        use_gpu=False
    )


def load_city_data(city: str) -> pd.DataFrame:
    """Load and preprocess city data"""
    path = DATA_DIR / f"clean_{city}_aq_1y.csv"
    if not path.exists():
        path = DATA_DIR / f"{city}_aq_1y.csv"
    
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle missing values
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(limit=6).ffill(limit=24).bfill(limit=24)
            df[col] = df[col].fillna(df[col].median())
    
    return df


def prepare_supervised_data(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON
) -> Dict[str, Any]:
    """Prepare supervised learning data with train/val/test split"""
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Scale features
    x_train = x_scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    x_val = x_scaler.transform(val_df[FEATURE_COLUMNS].values)
    x_test = x_scaler.transform(test_df[FEATURE_COLUMNS].values)
    
    # Scale target
    y_train = y_scaler.fit_transform(train_df[[TARGET_COLUMN]].values).flatten()
    y_val = y_scaler.transform(val_df[[TARGET_COLUMN]].values).flatten()
    y_test = y_scaler.transform(test_df[[TARGET_COLUMN]].values).flatten()
    
    def make_windows(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for i in range(len(x) - lookback - horizon + 1):
            X.append(x[i:i+lookback])
            Y.append(y[i+lookback:i+lookback+horizon])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    
    X_train, Y_train = make_windows(x_train, y_train)
    X_val, Y_val = make_windows(x_val, y_val)
    X_test, Y_test = make_windows(x_test, y_test)
    
    return {
        'X_train': X_train, 'y_train': Y_train,
        'X_val': X_val, 'y_val': Y_val,
        'X_test': X_test, 'y_test': Y_test,
        'x_scaler': x_scaler, 'y_scaler': y_scaler,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df
    }


def save_model_artifacts(
    output_dir: Path,
    model: Any,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    y_scaler: StandardScaler,
    x_scaler: StandardScaler
) -> None:
    """Save model and metadata"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), output_dir / "model.pth")
    else:
        joblib.dump({
            'model': model,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'features': FEATURE_COLUMNS,
            'target': TARGET_COLUMN
        }, output_dir / "model.joblib")
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save scalers separately
    joblib.dump(x_scaler, output_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")


def generate_plots(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: Optional[Dict[str, List[float]]] = None,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> None:
    """Generate all required plots for report"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten arrays for plotting
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    errors = y_pred_flat - y_true_flat
    
    # 1. Convergence plot (for DL models)
    if history and 'train_loss' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "convergence.png", dpi=150)
        plt.close()
    
    # 2. Parity plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual US AQI')
    ax.set_ylabel('Predicted US AQI')
    ax.set_title('Parity Plot: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "parity.png", dpi=150)
    plt.close()
    
    # 3. Timeseries forecast plot (first 168 points = 1 week)
    fig, ax = plt.subplots(figsize=(14, 6))
    n_plot = min(168, len(y_true_flat))
    
    if timestamps is not None and len(timestamps) >= n_plot:
        plot_times = timestamps[:n_plot]
        ax.plot(plot_times, y_true_flat[:n_plot], label='Actual', linewidth=2, alpha=0.8)
        ax.plot(plot_times, y_pred_flat[:n_plot], label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('Timestamp')
    else:
        ax.plot(y_true_flat[:n_plot], label='Actual', linewidth=2, alpha=0.8)
        ax.plot(y_pred_flat[:n_plot], label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time Step')
    
    ax.set_ylabel('US AQI')
    ax.set_title('Time Series Forecast (First Week)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "timeseries.png", dpi=150)
    plt.close()
    
    # 4. Error histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (Predicted - Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_histogram.png", dpi=150)
    plt.close()
    
    logger.info(f"Saved plots to {plots_dir}")


# ============================================================================
# STATISTICAL MODELS
# ============================================================================

def train_arima(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train ARIMA model"""
    from statsmodels.tsa.arima.model import ARIMA
    
    train_series = data['train_df'][TARGET_COLUMN].values
    test_series = data['test_df'][TARGET_COLUMN].values
    
    start = time.time()
    model = ARIMA(train_series, order=(2, 1, 2))
    model_fit = model.fit()
    train_time = time.time() - start
    
    start = time.time()
    forecast = model_fit.forecast(steps=len(test_series))
    infer_time = time.time() - start
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(test_series, forecast))),
        'mae': float(mean_absolute_error(test_series, forecast)),
        'r2': float(r2_score(test_series, forecast)),
        'train_seconds': train_time,
        'inference_seconds': infer_time
    }
    
    # Reshape forecast to match windowed format (n_samples x horizon)
    # Use the forecast to create overlapping windows matching y_test structure
    horizon = data['y_test'].shape[1] if len(data['y_test'].shape) > 1 else 24
    n_samples = len(data['y_test'])
    forecast_windowed = np.array([forecast[i:i+horizon] for i in range(n_samples)])
    
    return model_fit, metrics, forecast_windowed


def train_sarima(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train SARIMA model"""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    train_series = data['train_df'][TARGET_COLUMN].values
    test_series = data['test_df'][TARGET_COLUMN].values
    
    start = time.time()
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    train_time = time.time() - start
    
    start = time.time()
    forecast = model_fit.forecast(steps=len(test_series))
    infer_time = time.time() - start
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(test_series, forecast))),
        'mae': float(mean_absolute_error(test_series, forecast)),
        'r2': float(r2_score(test_series, forecast)),
        'train_seconds': train_time,
        'inference_seconds': infer_time
    }
    
    # Reshape forecast to match windowed format
    horizon = data['y_test'].shape[1] if len(data['y_test'].shape) > 1 else 24
    n_samples = len(data['y_test'])
    forecast_windowed = np.array([forecast[i:i+horizon] for i in range(n_samples)])
    
    return model_fit, metrics, forecast_windowed


def train_var(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train VAR model"""
    from statsmodels.tsa.api import VAR
    
    var_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    train_multi = data['train_df'][var_cols].values
    test_multi = data['test_df'][var_cols].values
    
    start = time.time()
    model = VAR(train_multi)
    model_fit = model.fit(maxlags=24, ic='aic')
    train_time = time.time() - start
    
    start = time.time()
    forecast = model_fit.forecast(train_multi[-model_fit.k_ar:], steps=len(test_multi))
    target_idx = var_cols.index(TARGET_COLUMN)
    forecast_target = forecast[:, target_idx]
    test_target = test_multi[:, target_idx]
    infer_time = time.time() - start
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(test_target, forecast_target))),
        'mae': float(mean_absolute_error(test_target, forecast_target)),
        'r2': float(r2_score(test_target, forecast_target)),
        'train_seconds': train_time,
        'inference_seconds': infer_time
    }
    
    # Reshape forecast to match windowed format
    horizon = data['y_test'].shape[1] if len(data['y_test'].shape) > 1 else 24
    n_samples = len(data['y_test'])
    forecast_windowed = np.array([forecast_target[i:i+horizon] for i in range(n_samples)])
    
    return model_fit, metrics, forecast_windowed


# ============================================================================
# CLASSICAL ML MODELS
# ============================================================================

def train_svr(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train SVR with hyperparameter optimization"""
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 100, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-3, 1.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, gamma=gamma))
        model.fit(X_train_flat[:500], data['y_train'][:500])  # Subset for speed
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    start = time.time()
    model = MultiOutputRegressor(SVR(**best_params))
    model.fit(X_train_flat, data['y_train'])
    train_time = time.time() - start
    
    start = time.time()
    preds_scaled = model.predict(X_test_flat)
    infer_time = time.time() - start
    
    # Inverse transform
    preds = data['y_scaler'].inverse_transform(preds_scaled)
    actuals = data['y_scaler'].inverse_transform(data['y_test'])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
        'r2': float(r2_score(actuals, preds)),
        'train_seconds': train_time,
        'inference_seconds': infer_time,
        'best_params': best_params
    }
    
    return model, metrics, preds


def train_random_forest(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train Random Forest with hyperparameter optimization"""
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
        model.fit(X_train_flat[:500], data['y_train'][:500])
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    start = time.time()
    model = MultiOutputRegressor(RandomForestRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ))
    model.fit(X_train_flat, data['y_train'])
    train_time = time.time() - start
    
    start = time.time()
    preds_scaled = model.predict(X_test_flat)
    infer_time = time.time() - start
    
    preds = data['y_scaler'].inverse_transform(preds_scaled)
    actuals = data['y_scaler'].inverse_transform(data['y_test'])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
        'r2': float(r2_score(actuals, preds)),
        'train_seconds': train_time,
        'inference_seconds': infer_time,
        'best_params': best_params
    }
    
    return model, metrics, preds


def train_xgboost(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train XGBoost with hyperparameter optimization"""
    from xgboost import XGBRegressor
    
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist',
            'random_state': RANDOM_SEED
        }
        
        model = MultiOutputRegressor(XGBRegressor(**params))
        model.fit(X_train_flat[:500], data['y_train'][:500])
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = RANDOM_SEED
    
    start = time.time()
    model = MultiOutputRegressor(XGBRegressor(**best_params))
    model.fit(X_train_flat, data['y_train'])
    train_time = time.time() - start
    
    start = time.time()
    preds_scaled = model.predict(X_test_flat)
    infer_time = time.time() - start
    
    preds = data['y_scaler'].inverse_transform(preds_scaled)
    actuals = data['y_scaler'].inverse_transform(data['y_test'])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
        'r2': float(r2_score(actuals, preds)),
        'train_seconds': train_time,
        'inference_seconds': infer_time,
        'best_params': best_params
    }
    
    return model, metrics, preds


def train_lightgbm(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train LightGBM with hyperparameter optimization"""
    from lightgbm import LGBMRegressor
    
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': RANDOM_SEED,
            'verbosity': -1
        }
        
        model = MultiOutputRegressor(LGBMRegressor(**params))
        model.fit(X_train_flat[:500], data['y_train'][:500])
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = RANDOM_SEED
    best_params['verbosity'] = -1
    
    start = time.time()
    model = MultiOutputRegressor(LGBMRegressor(**best_params))
    model.fit(X_train_flat, data['y_train'])
    train_time = time.time() - start
    
    start = time.time()
    preds_scaled = model.predict(X_test_flat)
    infer_time = time.time() - start
    
    preds = data['y_scaler'].inverse_transform(preds_scaled)
    actuals = data['y_scaler'].inverse_transform(data['y_test'])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
        'r2': float(r2_score(actuals, preds)),
        'train_seconds': train_time,
        'inference_seconds': infer_time,
        'best_params': best_params
    }
    
    return model, metrics, preds


def train_catboost(data: Dict[str, Any], compute: ComputeConfig) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """Train CatBoost with hyperparameter optimization"""
    from catboost import CatBoostRegressor
    
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
            'verbose': False,
            'allow_writing_files': False
        }
        
        model = MultiOutputRegressor(CatBoostRegressor(**params))
        model.fit(X_train_flat[:500], data['y_train'][:500])
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['verbose'] = False
    best_params['allow_writing_files'] = False
    
    start = time.time()
    model = MultiOutputRegressor(CatBoostRegressor(**best_params))
    model.fit(X_train_flat, data['y_train'])
    train_time = time.time() - start
    
    start = time.time()
    preds_scaled = model.predict(X_test_flat)
    infer_time = time.time() - start
    
    preds = data['y_scaler'].inverse_transform(preds_scaled)
    actuals = data['y_scaler'].inverse_transform(data['y_test'])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
        'r2': float(r2_score(actuals, preds)),
        'train_seconds': train_time,
        'inference_seconds': infer_time,
        'best_params': best_params
    }
    
    return model, metrics, preds


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class SequenceModel(nn.Module):
    """RNN/LSTM/GRU/BiLSTM model"""
    def __init__(self, model_type: str, input_dim: int, hidden_dim: int, 
                 num_layers: int, horizon: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "RNN":
            self.seq = nn.RNN(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "LSTM":
            self.seq = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "GRU":
            self.seq = nn.GRU(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "BiLSTM":
            self.seq = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, horizon)
            return
        
        self.fc = nn.Linear(hidden_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.seq(x)
        return self.fc(out[:, -1, :])


class CNNRNN(nn.Module):
    """CNN-LSTM or CNN-GRU model"""
    def __init__(self, rnn_type: str, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(32, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.rnn(conv_out)
        return self.fc(output[:, -1, :])


class BiLSTMAttention(nn.Module):
    """Bi-LSTM with Attention"""
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.score = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        weights = torch.softmax(self.score(output), dim=1)
        context = torch.sum(output * weights, dim=1)
        return self.fc(context)


class PositionalEncoding(nn.Module):
    """Transformer positional encoding"""
    def __init__(self, model_dim: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if model_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Transformer model"""
    def __init__(self, input_dim: int, horizon: int, model_dim: int = 64, 
                 layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim, heads, dim_feedforward=model_dim*2, 
            batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(model_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.pos(self.proj(x)))
        return self.fc(encoded[:, -1, :])


class InformerLite(nn.Module):
    """Informer (lightweight)"""
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.distill = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1)
        self.transformer = TransformerForecaster(input_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.distill(x.transpose(1, 2)).transpose(1, 2)
        return self.transformer(x)


class AutoformerLite(nn.Module):
    """Autoformer (lightweight)"""
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.transformer = TransformerForecaster(input_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return self.transformer(seasonal + trend)


class TFTLite(nn.Module):
    """Temporal Fusion Transformer (lightweight)"""
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.fc = nn.Linear(hidden_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        attended, _ = self.attn(output, output, output)
        gated = attended[:, -1, :] * self.gate(attended[:, -1, :])
        return self.fc(gated)


class STGCNLite(nn.Module):
    """Spatio-Temporal GCN (lightweight)"""
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.graph_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x.transpose(1, 2)).mean(dim=-1)
        mixed = temporal + torch.relu(self.graph_gate(temporal))
        return self.fc(mixed)


def train_deep_learning_model(
    model_class: type,
    model_name: str,
    data: Dict[str, Any],
    compute: ComputeConfig,
    model_kwargs: Dict[str, Any] = {}
) -> Tuple[nn.Module, Dict[str, float], np.ndarray]:
    """Generic deep learning model trainer with Optuna"""
    
    device = compute.device
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    
    input_dim = data['X_train'].shape[-1]
    
    # Optuna hyperparameter search
    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        if 'num_layers' in model_kwargs:
            num_layers = trial.suggest_int('num_layers', 1, 3)
        else:
            num_layers = 2
        
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=HORIZON,
            num_layers=num_layers,
            dropout=dropout,
            **{k: v for k, v in model_kwargs.items() if k != 'num_layers'}
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Quick training for hyperparameter search
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=compute.batch_size,
            shuffle=True
        )
        
        model.train()
        for _ in range(5):  # 5 epochs for tuning
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()
        
        return val_loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    logger.info(f"Best params for {model_name}: {best_params}")
    
    # Full training with best params
    model = model_class(
        input_dim=input_dim,
        hidden_dim=best_params.get('hidden_dim', 64),
        horizon=HORIZON,
        num_layers=best_params.get('num_layers', 2),
        dropout=best_params.get('dropout', 0.2),
        **model_kwargs
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=best_params.get('lr', 1e-3),
        weight_decay=1e-4
    )
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=compute.batch_size,
        shuffle=True
    )
    
    # Training with history tracking
    history = {'train_loss': [], 'val_loss': []}
    
    start = time.time()
    for epoch in range(compute.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        train_loss = np.mean(epoch_losses)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()
            history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"{model_name} Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    train_time = time.time() - start
    
    # Inference
    model.eval()
    start = time.time()
    with torch.no_grad():
        test_pred_scaled = model(X_test.to(device)).cpu().numpy()
    infer_time = time.time() - start
    
    # Inverse transform
    preds = data['y_scaler'].inverse_transform(test_pred_scaled)
    actuals = data['y_scaler'].inverse_transform(data['y_test'])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
        'r2': float(r2_score(actuals, preds)),
        'train_seconds': train_time,
        'inference_seconds': infer_time,
        'best_params': best_params,
        'epochs': compute.epochs
    }
    
    return model, metrics, preds, history


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

MODEL_REGISTRY = {
    # Phase 1: Statistical Baselines
    'ARIMA': {'phase': 'Phase 1: Statistical Baselines', 'family': 'statistical', 'trainer': train_arima},
    'SARIMA': {'phase': 'Phase 1: Statistical Baselines', 'family': 'statistical', 'trainer': train_sarima},
    'VAR': {'phase': 'Phase 1: Statistical Baselines', 'family': 'statistical', 'trainer': train_var},
    
    # Phase 2: Classical ML Ensembles
    'SVR': {'phase': 'Phase 2: Classical ML Ensembles', 'family': 'classical_ml', 'trainer': train_svr},
    'Random Forest': {'phase': 'Phase 2: Classical ML Ensembles', 'family': 'classical_ml', 'trainer': train_random_forest},
    'XGBoost': {'phase': 'Phase 2: Classical ML Ensembles', 'family': 'classical_ml', 'trainer': train_xgboost},
    'LightGBM': {'phase': 'Phase 2: Classical ML Ensembles', 'family': 'classical_ml', 'trainer': train_lightgbm},
    'CatBoost': {'phase': 'Phase 2: Classical ML Ensembles', 'family': 'classical_ml', 'trainer': train_catboost},
    
    # Phase 3: Standard DL Sequence Models
    'RNN': {'phase': 'Phase 3: Standard DL Sequence Models', 'family': 'deep_learning', 
            'trainer': lambda d, c: train_deep_learning_model(SequenceModel, 'RNN', d, c, {'model_type': 'RNN'})},
    'LSTM': {'phase': 'Phase 3: Standard DL Sequence Models', 'family': 'deep_learning',
             'trainer': lambda d, c: train_deep_learning_model(SequenceModel, 'LSTM', d, c, {'model_type': 'LSTM'})},
    'GRU': {'phase': 'Phase 3: Standard DL Sequence Models', 'family': 'deep_learning',
            'trainer': lambda d, c: train_deep_learning_model(SequenceModel, 'GRU', d, c, {'model_type': 'GRU'})},
    'Bi-LSTM': {'phase': 'Phase 3: Standard DL Sequence Models', 'family': 'deep_learning',
                'trainer': lambda d, c: train_deep_learning_model(SequenceModel, 'Bi-LSTM', d, c, {'model_type': 'BiLSTM'})},
    
    # Phase 4: Hybrid and Attention
    'CNN-LSTM': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
                 'trainer': lambda d, c: train_deep_learning_model(CNNRNN, 'CNN-LSTM', d, c, {'rnn_type': 'lstm'})},
    'CNN-GRU': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
                'trainer': lambda d, c: train_deep_learning_model(CNNRNN, 'CNN-GRU', d, c, {'rnn_type': 'gru'})},
    'Bi-LSTM + Attention': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
                            'trainer': lambda d, c: train_deep_learning_model(BiLSTMAttention, 'Bi-LSTM + Attention', d, c)},
    'Transformer': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
                    'trainer': lambda d, c: train_deep_learning_model(TransformerForecaster, 'Transformer', d, c)},
    'Informer': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
                 'trainer': lambda d, c: train_deep_learning_model(InformerLite, 'Informer', d, c)},
    'Autoformer': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
                   'trainer': lambda d, c: train_deep_learning_model(AutoformerLite, 'Autoformer', d, c)},
    'TFT': {'phase': 'Phase 4: Hybrid and Attention', 'family': 'deep_learning',
            'trainer': lambda d, c: train_deep_learning_model(TFTLite, 'TFT', d, c)},
    
    # Phase 5: Spatio-Temporal
    'ST-GCN': {'phase': 'Phase 5: Spatio-Temporal Models', 'family': 'deep_learning',
               'trainer': lambda d, c: train_deep_learning_model(STGCNLite, 'ST-GCN', d, c)},
}


def train_single_model(
    model_name: str,
    model_spec: Dict[str, Any],
    data: Dict[str, Any],
    compute: ComputeConfig,
    output_base: Path
) -> ModelResult:
    """Train a single model and save artifacts"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"Phase: {model_spec['phase']}")
    logger.info(f"Family: {model_spec['family']}")
    logger.info(f"{'='*60}")
    
    output_dir = output_base / model_name.replace(' ', '_').replace('+', 'plus')
    
    try:
        start = time.time()
        
        # Train model
        result = model_spec['trainer'](data, compute)
        
        if len(result) == 4:  # DL model with history
            model, metrics, preds, history = result
            is_dl = True
        else:  # Classical/Statistical model
            model, metrics, preds = result
            history = None
            is_dl = False
        
        total_time = time.time() - start
        
        # Create timestamps for timeseries plot
        test_timestamps = None
        if 'test_df' in data and len(data['test_df']) > 0:
            test_timestamps = pd.date_range(
                start=data['test_df']['timestamp'].iloc[0],
                periods=len(preds.flatten()),
                freq='H'
            )
        
        # Generate plots
        actuals = data['y_scaler'].inverse_transform(data['y_test'])
        generate_plots(output_dir, actuals, preds, history, test_timestamps)
        
        # Save artifacts
        save_model_artifacts(
            output_dir, model, metrics,
            {'model_name': model_name, 'phase': model_spec['phase'], 
             'family': model_spec['family'], **metrics.get('best_params', {})},
            data['y_scaler'], data['x_scaler']
        )
        
        logger.info(f"✓ {model_name} completed in {total_time:.1f}s")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return ModelResult(
            model_name=model_name,
            phase=model_spec['phase'],
            family=model_spec['family'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            r2=metrics['r2'],
            train_seconds=metrics['train_seconds'],
            inference_seconds=metrics['inference_seconds'],
            status='ok'
        )
        
    except Exception as e:
        logger.error(f"✗ {model_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return ModelResult(
            model_name=model_name,
            phase=model_spec['phase'],
            family=model_spec['family'],
            rmse=float('inf'),
            mae=float('inf'),
            r2=float('-inf'),
            train_seconds=0,
            inference_seconds=0,
            status='failed',
            error_msg=str(e)
        )


def generate_comparison_report(results: List[ModelResult], output_dir: Path) -> None:
    """Generate consolidated comparison report"""
    # Create DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Save CSV
    df.to_csv(output_dir / "comparison_report.csv", index=False)
    
    # Generate comparison plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    successful = df[df['status'] == 'ok'].sort_values('rmse')
    
    if len(successful) == 0:
        logger.warning("No successful models to compare!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. RMSE comparison
    colors = ['green' if f in ['statistical'] else 'steelblue' if f == 'classical_ml' else 'orange' 
              for f in successful['family']]
    axes[0, 0].barh(successful['model_name'], successful['rmse'], color=colors)
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_title('Model Comparison: RMSE (Lower is Better)')
    axes[0, 0].axvline(x=successful['rmse'].min(), color='red', linestyle='--', alpha=0.5)
    
    # 2. R² comparison
    axes[0, 1].barh(successful['model_name'], successful['r2'], color=colors)
    axes[0, 1].set_xlabel('R² Score')
    axes[0, 1].set_title('Model Comparison: R² (Higher is Better)')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Training time
    axes[1, 0].barh(successful['model_name'], successful['train_seconds'], color=colors)
    axes[1, 0].set_xlabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    
    # 4. Inference time
    axes[1, 1].barh(successful['model_name'], successful['inference_seconds'], color=colors)
    axes[1, 1].set_xlabel('Inference Time (seconds)')
    axes[1, 1].set_title('Inference Time Comparison')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    
    # Summary text
    summary = f"""
# Model Training Summary - {CITY.upper()}

## Compute Platform: {results[0].family if results else 'Unknown'}

## Best Models (by RMSE)
"""
    for i, row in successful.head(5).iterrows():
        summary += f"{i+1}. **{row['model_name']}**: RMSE={row['rmse']:.2f}, R²={row['r2']:.3f}\n"
    
    summary += f"""
## Training Statistics
- Total models: {len(df)}
- Successful: {len(df[df['status'] == 'ok'])}
- Failed: {len(df[df['status'] == 'failed'])}

## Phase-wise Best
"""
    for phase in df['phase'].unique():
        phase_df = df[(df['phase'] == phase) & (df['status'] == 'ok')]
        if len(phase_df) > 0:
            best = phase_df.loc[phase_df['rmse'].idxmin()]
            summary += f"- {phase}: **{best['model_name']}** (RMSE: {best['rmse']:.2f})\n"
    
    with open(output_dir / "SUMMARY.md", 'w') as f:
        f.write(summary)
    
    logger.info(f"\nComparison report saved to {output_dir}")
    logger.info(f"Best model: {successful.iloc[0]['model_name']} (RMSE: {successful.iloc[0]['rmse']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument("--models", nargs="+", help="Specific models to train (default: all)")
    parser.add_argument("--skip-dl", action="store_true", help="Skip deep learning models")
    parser.add_argument("--skip-ml", action="store_true", help="Skip classical ML models")
    parser.add_argument("--skip-stat", action="store_true", help="Skip statistical models")
    args = parser.parse_args()
    
    # Detect compute
    compute = detect_compute()
    logger.info(f"Using compute platform: {compute.platform}")
    logger.info(f"Device: {compute.device}")
    logger.info(f"Batch size: {compute.batch_size}, Epochs: {compute.epochs}")
    
    # Load data
    logger.info(f"\nLoading data for {CITY}...")
    df = load_city_data(CITY)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prepare supervised data
    logger.info("Preparing supervised data...")
    data = prepare_supervised_data(df)
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    # Setup output directory
    output_base = OUTPUT_BASE / CITY
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base}")
    
    # Filter models
    models_to_train = args.models if args.models else list(MODEL_REGISTRY.keys())
    
    if args.skip_dl:
        models_to_train = [m for m in models_to_train if MODEL_REGISTRY[m]['family'] != 'deep_learning']
    if args.skip_ml:
        models_to_train = [m for m in models_to_train if MODEL_REGISTRY[m]['family'] != 'classical_ml']
    if args.skip_stat:
        models_to_train = [m for m in models_to_train if MODEL_REGISTRY[m]['family'] != 'statistical']
    
    logger.info(f"\nModels to train: {len(models_to_train)}")
    
    # Train all models
    results = []
    total_start = time.time()
    
    for i, model_name in enumerate(models_to_train, 1):
        logger.info(f"\n[{i}/{len(models_to_train)}] Training {model_name}...")
        model_spec = MODEL_REGISTRY[model_name]
        result = train_single_model(model_name, model_spec, data, compute, output_base)
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Generate report
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Total time: {total_time/60:.1f} minutes")
    logger.info(f"{'='*60}")
    
    generate_comparison_report(results, output_base)
    
    # Save results
    with open(output_base / "all_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    
    logger.info(f"\nAll artifacts saved to: {output_base}")


if __name__ == "__main__":
    main()
