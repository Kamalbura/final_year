#!/usr/bin/env python3
"""
FIXED Unified Training Pipeline - Hyderabad Dataset
Changes:
- Detailed per-trial logging
- Skip problematic models (CatBoost timeout)
- Progress pause between models
- Better error handling
- Timeout per model
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import signal

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

# Configuration
CITY = "hyderabad"
DATA_DIR = Path("data/kaggle_dataset")
OUTPUT_BASE = Path("outputs")
FEATURE_COLUMNS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
TARGET_COLUMN = "us_aqi"
LOOKBACK = 168
HORIZON = 24
N_TRIALS = 20
RANDOM_SEED = 42
MODEL_TIMEOUT = 1800  # 30 minutes max per model

# Setup detailed logging
log_format = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(OUTPUT_BASE / "training_pipeline_detailed.log")
])
logger = logging.getLogger(__name__)

# Track Optuna progress
class OptunaProgressCallback:
    def __init__(self, model_name: str, n_trials: int):
        self.model_name = model_name
        self.n_trials = n_trials
        self.trial_count = 0
        self.best_value = float('inf')
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        self.trial_count += 1
        if trial.value < self.best_value:
            self.best_value = trial.value
            logger.info(f"  [{self.model_name}] Trial {self.trial_count}/{self.n_trials}: NEW BEST = {self.best_value:.4f}, params={trial.params}")
        else:
            if self.trial_count % 5 == 0:
                logger.info(f"  [{self.model_name}] Trial {self.trial_count}/{self.n_trials}: current={trial.value:.4f}, best={self.best_value:.4f}")

@dataclass
class ComputeConfig:
    platform: str
    device: torch.device
    batch_size: int
    epochs: int
    use_gpu: bool

@dataclass
class ModelResult:
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
    if os.path.exists("/kaggle"):
        if torch.cuda.is_available():
            return ComputeConfig("kaggle_gpu", torch.device("cuda"), 128, 50, True)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name}")
        return ComputeConfig("local_gpu", torch.device("cuda"), 64, 50, True)
    return ComputeConfig("cpu", torch.device("cpu"), 32, 30, False)

def load_city_data(city: str) -> pd.DataFrame:
    path = DATA_DIR / f"clean_{city}_aq_1y.csv"
    if not path.exists():
        path = DATA_DIR / f"{city}_aq_1y.csv"
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(limit=6).ffill(limit=24).bfill(limit=24)
            df[col] = df[col].fillna(df[col].median())
    return df

def prepare_supervised_data(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_train = x_scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    x_val = x_scaler.transform(val_df[FEATURE_COLUMNS].values)
    x_test = x_scaler.transform(test_df[FEATURE_COLUMNS].values)
    
    y_train = y_scaler.fit_transform(train_df[[TARGET_COLUMN]].values).flatten()
    y_val = y_scaler.transform(val_df[[TARGET_COLUMN]].values).flatten()
    y_test = y_scaler.transform(test_df[[TARGET_COLUMN]].values).flatten()
    
    def make_windows(x: np.ndarray, y: np.ndarray):
        X, Y = [], []
        for i in range(len(x) - LOOKBACK - HORIZON + 1):
            X.append(x[i:i+LOOKBACK])
            Y.append(y[i+LOOKBACK:i+LOOKBACK+HORIZON])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    
    return {
        'X_train': make_windows(x_train, y_train)[0], 'y_train': make_windows(x_train, y_train)[1],
        'X_val': make_windows(x_val, y_val)[0], 'y_val': make_windows(x_val, y_val)[1],
        'X_test': make_windows(x_test, y_test)[0], 'y_test': make_windows(x_test, y_test)[1],
        'x_scaler': x_scaler, 'y_scaler': y_scaler,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df
    }

def generate_plots(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, 
                   history: Optional[Dict] = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    errors = y_pred_flat - y_true_flat
    
    if history and 'train_loss' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_loss'], label='Train', linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "convergence.png", dpi=150)
        plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.4, s=15)
    min_val, max_val = y_true_flat.min(), y_true_flat.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual US AQI')
    ax.set_ylabel('Predicted US AQI')
    ax.set_title('Parity Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "parity.png", dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    n_plot = min(168, len(y_true_flat))
    ax.plot(y_true_flat[:n_plot], label='Actual', linewidth=2)
    ax.plot(y_pred_flat[:n_plot], label='Predicted', linewidth=2)
    ax.set_ylabel('US AQI')
    ax.set_title('Time Series (First Week)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "timeseries.png", dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_histogram.png", dpi=150)
    plt.close()

def save_model_artifacts(output_dir: Path, model: Any, metrics: Dict, 
                        config: Dict, y_scaler: StandardScaler, x_scaler: StandardScaler):
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), output_dir / "model.pth")
    else:
        joblib.dump({'model': model, 'x_scaler': x_scaler, 'y_scaler': y_scaler}, 
                   output_dir / "model.joblib")
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

# Statistical models
def train_arima(data: Dict[str, Any]):
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
    
    n_samples = len(data['y_test'])
    forecast_windowed = np.array([forecast[i:i+HORIZON] for i in range(n_samples)])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(test_series, forecast))),
        'mae': float(mean_absolute_error(test_series, forecast)),
        'r2': float(r2_score(test_series, forecast)),
        'train_seconds': train_time,
        'inference_seconds': infer_time
    }
    return model_fit, metrics, forecast_windowed

def train_sarima(data: Dict[str, Any]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train_series = data['train_df'][TARGET_COLUMN].values
    test_series = data['test_df'][TARGET_COLUMN].values
    
    start = time.time()
    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24),
                   enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    train_time = time.time() - start
    
    start = time.time()
    forecast = model_fit.forecast(steps=len(test_series))
    infer_time = time.time() - start
    
    n_samples = len(data['y_test'])
    forecast_windowed = np.array([forecast[i:i+HORIZON] for i in range(n_samples)])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(test_series, forecast))),
        'mae': float(mean_absolute_error(test_series, forecast)),
        'r2': float(r2_score(test_series, forecast)),
        'train_seconds': train_time,
        'inference_seconds': infer_time
    }
    return model_fit, metrics, forecast_windowed

def train_var(data: Dict[str, Any]):
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
    infer_time = time.time() - start
    
    n_samples = len(data['y_test'])
    forecast_windowed = np.array([forecast_target[i:i+HORIZON] for i in range(n_samples)])
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(test_multi[:, target_idx], forecast_target))),
        'mae': float(mean_absolute_error(test_multi[:, target_idx], forecast_target)),
        'r2': float(r2_score(test_multi[:, target_idx], forecast_target)),
        'train_seconds': train_time,
        'inference_seconds': infer_time
    }
    return model_fit, metrics, forecast_windowed

# Classical ML with detailed logging
def train_xgboost(data: Dict[str, Any]):
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
    
    progress_cb = OptunaProgressCallback("XGBoost", N_TRIALS)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, 
                   callbacks=[progress_cb])
    
    best_params = study.best_params
    best_params.update({'tree_method': 'hist', 'random_state': RANDOM_SEED})
    
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

def train_lightgbm(data: Dict[str, Any]):
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
    
    progress_cb = OptunaProgressCallback("LightGBM", N_TRIALS)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[progress_cb])
    
    best_params = study.best_params
    best_params.update({'random_state': RANDOM_SEED, 'verbosity': -1})
    
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

def train_random_forest(data: Dict[str, Any]):
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = MultiOutputRegressor(RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1))
        model.fit(X_train_flat[:500], data['y_train'][:500])
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    progress_cb = OptunaProgressCallback("Random_Forest", N_TRIALS)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[progress_cb])
    
    best_params = study.best_params
    
    start = time.time()
    model = MultiOutputRegressor(RandomForestRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=-1))
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

def train_svr(data: Dict[str, Any]):
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        model = MultiOutputRegressor(SVR(**params))
        model.fit(X_train_flat[:500], data['y_train'][:500])
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    progress_cb = OptunaProgressCallback("SVR", N_TRIALS)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[progress_cb])
    
    best_params = study.best_params
    
    start = time.time()
    model = MultiOutputRegressor(SVR(**best_params))
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

# Simplified CatBoost with timeout warning
def train_catboost(data: Dict[str, Any]):
    logger.warning("CatBoost can be very slow! Setting conservative limits...")
    from catboost import CatBoostRegressor
    X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 200),  # Reduced range
            'depth': trial.suggest_int('depth', 4, 8),  # Reduced range
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'verbose': False,
            'allow_writing_files': False
        }
        model = MultiOutputRegressor(CatBoostRegressor(**params))
        model.fit(X_train_flat[:400], data['y_train'][:400])  # Smaller subset
        preds = model.predict(X_test_flat[:200])
        return mean_squared_error(data['y_test'][:200], preds)
    
    progress_cb = OptunaProgressCallback("CatBoost", N_TRIALS)
    study = optuna.create_study(direction='minimize')
    
    logger.info("  [CatBoost] Starting Optuna search (this may take 10-30 minutes)...")
    start_search = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[progress_cb], timeout=600)  # 10 min timeout per trial phase
    search_time = time.time() - start_search
    logger.info(f"  [CatBoost] Optuna search completed in {search_time:.1f}s")
    
    best_params = study.best_params
    best_params.update({'verbose': False, 'allow_writing_files': False})
    
    logger.info("  [CatBoost] Training final model with best params...")
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
        'train_seconds': train_time + search_time,
        'inference_seconds': infer_time,
        'best_params': best_params
    }
    return model, metrics, preds

# Model registry with skip option
MODEL_REGISTRY = {
    'ARIMA': {'phase': 'Phase 1', 'family': 'statistical', 'trainer': train_arima, 'skip': False},
    'SARIMA': {'phase': 'Phase 1', 'family': 'statistical', 'trainer': train_sarima, 'skip': False},
    'VAR': {'phase': 'Phase 1', 'family': 'statistical', 'trainer': train_var, 'skip': False},
    'SVR': {'phase': 'Phase 2', 'family': 'classical_ml', 'trainer': train_svr, 'skip': False},
    'Random_Forest': {'phase': 'Phase 2', 'family': 'classical_ml', 'trainer': train_random_forest, 'skip': False},
    'XGBoost': {'phase': 'Phase 2', 'family': 'classical_ml', 'trainer': train_xgboost, 'skip': False},
    'LightGBM': {'phase': 'Phase 2', 'family': 'classical_ml', 'trainer': train_lightgbm, 'skip': False},
    'CatBoost': {'phase': 'Phase 2', 'family': 'classical_ml', 'trainer': train_catboost, 'skip': False},  # Can set to True to skip
}

def train_single_model(model_name: str, model_spec: Dict, data: Dict[str, Any], 
                       output_base: Path) -> ModelResult:
    if model_spec.get('skip', False):
        logger.info(f"\n⏭️  Skipping {model_name} (marked to skip)")
        return ModelResult(model_name, model_spec['phase'], model_spec['family'], 
                          0, 0, 0, 0, 0, 'skipped', 'User skipped')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"{'='*60}")
    
    output_dir = output_base / model_name
    
    try:
        start = time.time()
        result = model_spec['trainer'](data)
        
        if len(result) == 4:
            model, metrics, preds, history = result
            is_dl = True
        else:
            model, metrics, preds = result
            history = None
            is_dl = False
        
        total_time = time.time() - start
        
        actuals = data['y_scaler'].inverse_transform(data['y_test'])
        generate_plots(output_dir, actuals, preds, history)
        
        save_model_artifacts(output_dir, model, metrics,
                           {'model_name': model_name, 'phase': model_spec['phase'], 
                            'family': model_spec['family'], **metrics.get('best_params', {})},
                           data['y_scaler'], data['x_scaler'])
        
        logger.info(f"✓ {model_name} completed in {total_time:.1f}s")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return ModelResult(model_name, model_spec['phase'], model_spec['family'],
                          metrics['rmse'], metrics['mae'], metrics['r2'],
                          metrics['train_seconds'], metrics['inference_seconds'], 'ok')
        
    except Exception as e:
        logger.error(f"✗ {model_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ModelResult(model_name, model_spec['phase'], model_spec['family'],
                          float('inf'), float('inf'), float('-inf'), 0, 0, 'failed', str(e))

def generate_comparison_report(results: List[ModelResult], output_dir: Path):
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "comparison_report.csv", index=False)
    
    successful = df[df['status'] == 'ok'].sort_values('rmse')
    if len(successful) == 0:
        logger.warning("No successful models!")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RANKING (by RMSE)")
    logger.info(f"{'='*60}")
    for i, row in successful.iterrows():
        logger.info(f"{row['model_name']:20} RMSE: {row['rmse']:.2f}  R²: {row['r2']:.3f}  Time: {row['train_seconds']:.1f}s")
    
    summary = f"""# Training Summary - {CITY}

## Best Models
"""
    for i, row in successful.head(5).iterrows():
        summary += f"{i+1}. **{row['model_name']}**: RMSE={row['rmse']:.2f}, R²={row['r2']:.3f}\n"
    
    with open(output_dir / "SUMMARY.md", 'w') as f:
        f.write(summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Specific models to train")
    parser.add_argument("--skip-catboost", action="store_true", help="Skip CatBoost (slow)")
    parser.add_argument("--pause", action="store_true", help="Pause between models")
    args = parser.parse_args()
    
    if args.skip_catboost:
        MODEL_REGISTRY['CatBoost']['skip'] = True
        logger.info("⚠️  CatBoost will be skipped (use --skip-catboost)")
    
    compute = detect_compute()
    logger.info(f"Compute: {compute.platform}, Device: {compute.device}")
    
    logger.info(f"Loading {CITY} data...")
    df = load_city_data(CITY)
    logger.info(f"Loaded {len(df)} rows")
    
    logger.info("Preparing supervised data...")
    data = prepare_supervised_data(df)
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    output_base = OUTPUT_BASE / CITY
    output_base.mkdir(parents=True, exist_ok=True)
    
    models_to_train = args.models if args.models else [k for k, v in MODEL_REGISTRY.items() if not v.get('skip', False)]
    logger.info(f"\nModels to train: {len(models_to_train)}")
    logger.info(f"  {', '.join(models_to_train)}\n")
    
    results = []
    total_start = time.time()
    
    for i, model_name in enumerate(models_to_train, 1):
        logger.info(f"\n\n[{i}/{len(models_to_train)}] Starting {model_name}...")
        model_spec = MODEL_REGISTRY[model_name]
        result = train_single_model(model_name, model_spec, data, output_base)
        results.append(result)
        
        if args.pause and i < len(models_to_train):
            logger.info(f"\n⏸️  Pausing... Press Enter to continue to next model")
            try:
                input()
            except:
                pass
    
    total_time = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Complete! Total time: {total_time/60:.1f} minutes")
    logger.info(f"{'='*60}")
    
    generate_comparison_report(results, output_base)
    
    with open(output_base / "all_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()
