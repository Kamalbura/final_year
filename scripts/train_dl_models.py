#!/usr/bin/env python3
"""
Deep Learning Training Pipeline - Hyderabad Dataset
Models: RNN, LSTM, GRU, BiLSTM, CNN-LSTM, CNN-GRU, BiLSTM-Attention, Transformer, Informer, Autoformer, TFT, ST-GCN
Optimized for RTX 2050 (4GB VRAM)
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
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

# Configuration - OPTIMIZED FOR RTX 2050 4GB
CITY = "hyderabad"
DATA_DIR = Path("data/kaggle_dataset")
OUTPUT_BASE = Path("outputs")
FEATURE_COLUMNS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
TARGET_COLUMN = "us_aqi"
LOOKBACK = 168
HORIZON = 24
N_TRIALS = 20
RANDOM_SEED = 42

# RTX 2050 Optimized Settings
BATCH_SIZE = 32  # Reduced for 4GB VRAM
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
log_format = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(OUTPUT_BASE / "training_dl_pipeline.log")
])
logger = logging.getLogger(__name__)

logger.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
            logger.info(f"  [{self.model_name}] Trial {self.trial_count}/{self.n_trials}: NEW BEST = {self.best_value:.4f}")
        else:
            if self.trial_count % 5 == 0:
                logger.info(f"  [{self.model_name}] Trial {self.trial_count}/{self.n_trials}: best={self.best_value:.4f}")

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
                   history: Dict[str, List[float]]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    errors = y_pred_flat - y_true_flat
    
    # 1. Convergence plot (DL models only)
    if history and 'train_loss' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val Loss', linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "convergence.png", dpi=150)
        plt.close()
        logger.info(f"  Saved convergence plot")
    
    # 2. Parity plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.4, s=15, color='steelblue')
    min_val, max_val = y_true_flat.min(), y_true_flat.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual US AQI', fontsize=12)
    ax.set_ylabel('Predicted US AQI', fontsize=12)
    ax.set_title('Parity Plot: Predicted vs Actual', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "parity.png", dpi=150)
    plt.close()
    
    # 3. Timeseries plot
    fig, ax = plt.subplots(figsize=(14, 6))
    n_plot = min(168, len(y_true_flat))
    ax.plot(y_true_flat[:n_plot], label='Actual', linewidth=2, color='steelblue')
    ax.plot(y_pred_flat[:n_plot], label='Predicted', linewidth=2, color='darkorange')
    ax.set_ylabel('US AQI', fontsize=12)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_title('Time Series Forecast (First Week)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "timeseries.png", dpi=150)
    plt.close()
    
    # 4. Error histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_histogram.png", dpi=150)
    plt.close()
    
    logger.info(f"  Saved all plots to {plots_dir}")

def save_model_artifacts(output_dir: Path, model: nn.Module, metrics: Dict, 
                        config: Dict, y_scaler: StandardScaler, x_scaler: StandardScaler):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    joblib.dump(x_scaler, output_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")

# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class SequenceModel(nn.Module):
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
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.distill = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1)
        self.transformer = TransformerForecaster(input_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.distill(x.transpose(1, 2)).transpose(1, 2)
        return self.transformer(x)

class AutoformerLite(nn.Module):
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.transformer = TransformerForecaster(input_dim, horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return self.transformer(seasonal + trend)

class TFTLite(nn.Module):
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

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dl_model(model_class: type, model_name: str, model_kwargs: Dict,
                   data: Dict[str, Any]) -> Tuple[nn.Module, Dict, np.ndarray, Dict]:
    """Train a deep learning model with Optuna"""
    
    logger.info(f"  Preparing data tensors...")
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    
    input_dim = data['X_train'].shape[-1]
    
    # Optuna hyperparameter search
    logger.info(f"  Starting Optuna hyperparameter search ({N_TRIALS} trials)...")
    
    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        num_layers = trial.suggest_int('num_layers', 1, 3) if 'num_layers' in model_kwargs else 2
        
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=HORIZON,
            num_layers=num_layers,
            dropout=dropout,
            **{k: v for k, v in model_kwargs.items() if k != 'num_layers'}
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Quick training for hyperparameter search
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        model.train()
        for _ in range(5):
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE))
            val_loss = criterion(val_pred, y_val.to(DEVICE)).item()
        
        return val_loss
    
    progress_cb = OptunaProgressCallback(model_name, N_TRIALS)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[progress_cb])
    
    best_params = study.best_params
    logger.info(f"  Best params: hidden_dim={best_params.get('hidden_dim', 64)}, "
                f"lr={best_params.get('lr', 1e-3):.6f}, dropout={best_params.get('dropout', 0.2):.2f}")
    
    # Full training with best params
    logger.info(f"  Training final model with {EPOCHS} epochs...")
    model = model_class(
        input_dim=input_dim,
        hidden_dim=best_params.get('hidden_dim', 64),
        horizon=HORIZON,
        num_layers=best_params.get('num_layers', 2),
        dropout=best_params.get('dropout', 0.2),
        **model_kwargs
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=best_params.get('lr', 1e-3), weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    history = {'train_loss': [], 'val_loss': []}
    
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        train_loss = np.mean(epoch_losses)
        history['train_loss'].append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE))
            val_loss = criterion(val_pred, y_val.to(DEVICE)).item()
            history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                logger.info(f"    GPU Memory: {mem:.2f} GB")
    
    train_time = time.time() - start
    
    # Inference
    model.eval()
    start = time.time()
    with torch.no_grad():
        test_pred_scaled = model(X_test.to(DEVICE)).cpu().numpy()
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
        'epochs': EPOCHS
    }
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, metrics, preds, history

# ============================================================================
# MODEL REGISTRY - DL ONLY
# ============================================================================

MODEL_REGISTRY = {
    # Phase 3: Standard DL Sequence Models
    'RNN': {
        'phase': 'Phase 3: Standard DL Sequence Models',
        'trainer': lambda d: train_dl_model(SequenceModel, 'RNN', {'model_type': 'RNN'}, d)
    },
    'LSTM': {
        'phase': 'Phase 3: Standard DL Sequence Models',
        'trainer': lambda d: train_dl_model(SequenceModel, 'LSTM', {'model_type': 'LSTM'}, d)
    },
    'GRU': {
        'phase': 'Phase 3: Standard DL Sequence Models',
        'trainer': lambda d: train_dl_model(SequenceModel, 'GRU', {'model_type': 'GRU'}, d)
    },
    'BiLSTM': {
        'phase': 'Phase 3: Standard DL Sequence Models',
        'trainer': lambda d: train_dl_model(SequenceModel, 'BiLSTM', {'model_type': 'BiLSTM'}, d)
    },
    
    # Phase 4: Hybrid and Attention
    'CNN-LSTM': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(CNNRNN, 'CNN-LSTM', {'rnn_type': 'lstm'}, d)
    },
    'CNN-GRU': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(CNNRNN, 'CNN-GRU', {'rnn_type': 'gru'}, d)
    },
    'BiLSTM_Attention': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(BiLSTMAttention, 'BiLSTM_Attention', {}, d)
    },
    'Transformer': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(TransformerForecaster, 'Transformer', {}, d)
    },
    'Informer': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(InformerLite, 'Informer', {}, d)
    },
    'Autoformer': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(AutoformerLite, 'Autoformer', {}, d)
    },
    'TFT': {
        'phase': 'Phase 4: Hybrid and Attention',
        'trainer': lambda d: train_dl_model(TFTLite, 'TFT', {}, d)
    },
    
    # Phase 5: Spatio-Temporal
    'ST-GCN': {
        'phase': 'Phase 5: Spatio-Temporal Models',
        'trainer': lambda d: train_dl_model(STGCNLite, 'ST-GCN', {}, d)
    },
}

def train_single_model(model_name: str, model_spec: Dict, data: Dict[str, Any], 
                       output_base: Path) -> ModelResult:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"Phase: {model_spec['phase']}")
    logger.info(f"{'='*60}")
    
    output_dir = output_base / model_name
    
    try:
        start = time.time()
        model, metrics, preds, history = model_spec['trainer'](data)
        total_time = time.time() - start
        
        actuals = data['y_scaler'].inverse_transform(data['y_test'])
        generate_plots(output_dir, actuals, preds, history)
        
        save_model_artifacts(output_dir, model, metrics,
                           {'model_name': model_name, 'phase': model_spec['phase'], 
                            **metrics.get('best_params', {})},
                           data['y_scaler'], data['x_scaler'])
        
        logger.info(f"✓ {model_name} completed in {total_time:.1f}s")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return ModelResult(model_name, model_spec['phase'], 'deep_learning',
                          metrics['rmse'], metrics['mae'], metrics['r2'],
                          metrics['train_seconds'], metrics['inference_seconds'], 'ok')
        
    except Exception as e:
        logger.error(f"✗ {model_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ModelResult(model_name, model_spec['phase'], 'deep_learning',
                          float('inf'), float('inf'), float('-inf'), 0, 0, 'failed', str(e))

def generate_comparison_report(results: List[ModelResult], output_dir: Path):
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "comparison_report_dl.csv", index=False)
    
    successful = df[df['status'] == 'ok'].sort_values('rmse')
    if len(successful) == 0:
        logger.warning("No successful DL models!")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("DEEP LEARNING MODEL RANKING (by RMSE)")
    logger.info(f"{'='*60}")
    for i, row in successful.iterrows():
        logger.info(f"{row['model_name']:20} RMSE: {row['rmse']:.2f}  R²: {row['r2']:.3f}  Time: {row['train_seconds']:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Pipeline")
    parser.add_argument("--models", nargs="+", help="Specific DL models to train")
    parser.add_argument("--skip", nargs="+", help="Models to skip")
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info("DEEP LEARNING TRAINING PIPELINE")
    logger.info(f"City: {CITY}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Models: {len(MODEL_REGISTRY)} DL models")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Loading {CITY} data...")
    df = load_city_data(CITY)
    logger.info(f"Loaded {len(df)} rows")
    
    logger.info("Preparing supervised data...")
    data = prepare_supervised_data(df)
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    output_base = OUTPUT_BASE / CITY
    output_base.mkdir(parents=True, exist_ok=True)
    
    models_to_train = args.models if args.models else list(MODEL_REGISTRY.keys())
    if args.skip:
        models_to_train = [m for m in models_to_train if m not in args.skip]
    
    logger.info(f"\nDL Models to train: {len(models_to_train)}")
    logger.info(f"  {', '.join(models_to_train)}\n")
    
    results = []
    total_start = time.time()
    
    for i, model_name in enumerate(models_to_train, 1):
        logger.info(f"\n\n[{i}/{len(models_to_train)}] Starting {model_name}...")
        model_spec = MODEL_REGISTRY[model_name]
        result = train_single_model(model_name, model_spec, data, output_base)
        results.append(result)
        
        # Clear memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    total_time = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Complete! Total time: {total_time/60:.1f} minutes")
    logger.info(f"{'='*60}")
    
    generate_comparison_report(results, output_base)
    
    with open(output_base / "all_results_dl.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()
