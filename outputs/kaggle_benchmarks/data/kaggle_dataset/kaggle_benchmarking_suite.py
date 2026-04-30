from __future__ import annotations

import argparse
import importlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import joblib
except Exception:  # pragma: no cover - Kaggle normally has joblib via sklearn.
    joblib = None


FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "ozone",
]
TARGET_COLUMN = "us_aqi"
TRAINING_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
VAR_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
TARGET_CITIES = ("delhi", "hyderabad", "bengaluru")


@dataclass(frozen=True)
class ModelSpec:
    phase: str
    name: str
    family: str
    trainer: str


@dataclass
class BenchmarkRow:
    city: str
    phase: str
    model: str
    family: str
    status: str
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    train_seconds: float | None = None
    inference_seconds: float | None = None
    reason: str = ""


MODEL_SPECS = [
    ModelSpec("Phase 1: Statistical Baselines", "ARIMA", "statistical", "arima"),
    ModelSpec("Phase 1: Statistical Baselines", "SARIMA", "statistical", "sarima"),
    ModelSpec("Phase 1: Statistical Baselines", "VAR", "statistical", "var"),
    ModelSpec("Phase 2: Classical ML Ensembles", "SVR", "classical_ml", "svr"),
    ModelSpec("Phase 2: Classical ML Ensembles", "Random Forest", "classical_ml", "random_forest"),
    ModelSpec("Phase 2: Classical ML Ensembles", "XGBoost", "classical_ml", "xgboost"),
    ModelSpec("Phase 2: Classical ML Ensembles", "LightGBM", "classical_ml", "lightgbm"),
    ModelSpec("Phase 2: Classical ML Ensembles", "CatBoost", "classical_ml", "catboost"),
    ModelSpec("Phase 3: Standard DL Sequence Models", "RNN", "deep_learning", "rnn"),
    ModelSpec("Phase 3: Standard DL Sequence Models", "LSTM", "deep_learning", "lstm"),
    ModelSpec("Phase 3: Standard DL Sequence Models", "GRU", "deep_learning", "gru"),
    ModelSpec("Phase 3: Standard DL Sequence Models", "Bi-LSTM", "deep_learning", "bilstm"),
    ModelSpec("Phase 4: Hybrid and Attention", "CNN-LSTM", "deep_learning", "cnn_lstm"),
    ModelSpec("Phase 4: Hybrid and Attention", "CNN-GRU", "deep_learning", "cnn_gru"),
    ModelSpec("Phase 4: Hybrid and Attention", "Bi-LSTM + Attention", "deep_learning", "bilstm_attention"),
    ModelSpec("Phase 4: Hybrid and Attention", "Transformer", "deep_learning", "transformer"),
    ModelSpec("Phase 4: Hybrid and Attention", "Informer", "deep_learning", "informer_lite"),
    ModelSpec("Phase 4: Hybrid and Attention", "Autoformer", "deep_learning", "autoformer_lite"),
    ModelSpec("Phase 4: Hybrid and Attention", "Temporal Fusion Transformer", "deep_learning", "tft_lite"),
    ModelSpec("Phase 5: Spatio-Temporal Models", "ST-GCN", "deep_learning", "stgcn_lite"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle GPU benchmark suite for 3-city AQI forecasting.")
    parser.add_argument("--data-dir", default="data/kaggle_dataset", help="Directory with clean city CSV files.")
    parser.add_argument("--output-dir", default="outputs/kaggle_benchmarks", help="Directory for artifacts and summaries.")
    parser.add_argument("--lookback", type=int, default=168, help="History window in hours.")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours.")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs for deep learning models.")
    parser.add_argument("--batch-size", type=int, default=64, help="Deep learning batch size.")
    parser.add_argument("--max-train-windows", type=int, default=1600, help="Cap for training windows per city/model.")
    parser.add_argument("--max-test-windows", type=int, default=400, help="Cap for test windows per city/model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    return parser.parse_args()


def optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        return exc


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def city_file(data_dir: Path, city: str) -> Path:
    clean_path = data_dir / f"clean_{city}_aq_1y.csv"
    raw_path = data_dir / f"{city}_aq_1y.csv"
    return clean_path if clean_path.exists() else raw_path


def load_city_frame(data_dir: Path, city: str) -> pd.DataFrame:
    path = city_file(data_dir, city)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset for {city}: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in ["timestamp", *TRAINING_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = frame[["timestamp", *TRAINING_COLUMNS]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    frame = frame.resample("1h").asfreq()

    for column in TRAINING_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].interpolate(limit=6, limit_direction="both")
        frame[column] = frame[column].ffill(limit=24).bfill(limit=24)
        frame[column] = frame[column].fillna(frame[column].median())

    if frame[TRAINING_COLUMNS].isna().any().any():
        raise ValueError(f"{city} still has missing values after imputation")
    return frame.reset_index()


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:val_end].copy(), frame.iloc[val_end:].copy()


def make_windows(features: np.ndarray, target: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    limit = len(features) - lookback - horizon + 1
    if limit <= 0:
        raise ValueError(f"Not enough rows for lookback={lookback}, horizon={horizon}")
    x_values = np.empty((limit, lookback, features.shape[1]), dtype=np.float32)
    y_values = np.empty((limit, horizon), dtype=np.float32)
    for index in range(limit):
        x_values[index] = features[index : index + lookback]
        y_values[index] = target[index + lookback : index + lookback + horizon]
    return x_values, y_values


def limit_windows(x_values: np.ndarray, y_values: np.ndarray, max_count: int) -> tuple[np.ndarray, np.ndarray]:
    if max_count <= 0 or len(x_values) <= max_count:
        return x_values, y_values
    indices = np.linspace(0, len(x_values) - 1, max_count, dtype=int)
    return x_values[indices], y_values[indices]


def prepare_supervised(
    frame: pd.DataFrame,
    lookback: int,
    horizon: int,
    max_train_windows: int,
    max_test_windows: int,
) -> dict[str, object]:
    train_df, _, test_df = split_frame(frame)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x_test_raw = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_raw = train_df[[TARGET_COLUMN]].to_numpy(dtype=np.float32)
    y_test_raw = test_df[[TARGET_COLUMN]].to_numpy(dtype=np.float32)

    x_train_scaled = x_scaler.fit_transform(x_train_raw)
    x_test_scaled = x_scaler.transform(x_test_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw).reshape(-1)
    y_test_scaled = y_scaler.transform(y_test_raw).reshape(-1)

    x_train, y_train = make_windows(x_train_scaled, y_train_scaled, lookback, horizon)
    x_test, y_test = make_windows(x_test_scaled, y_test_scaled, lookback, horizon)
    x_train, y_train = limit_windows(x_train, y_train, max_train_windows)
    x_test, y_test = limit_windows(x_test, y_test, max_test_windows)

    latest_window = x_scaler.transform(frame[FEATURE_COLUMNS].tail(lookback).to_numpy(dtype=np.float32))
    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "latest_window": latest_window.reshape(1, lookback, len(FEATURE_COLUMNS)),
    }


def inverse_targets(y_scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    return y_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(flat_true, flat_pred))),
        "mae": float(mean_absolute_error(flat_true, flat_pred)),
        "r2": float(r2_score(flat_true, flat_pred)) if len(flat_true) > 1 else float("nan"),
    }


class SequenceRNN(nn.Module):
    def __init__(self, rnn_type: str, input_dim: int, hidden_dim: int, horizon: int, bidirectional: bool = False):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_dim * (2 if bidirectional else 1), horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :])


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
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.rnn(conv_out)
        return self.head(output[:, -1, :])


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.score = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Linear(hidden_dim * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        weights = torch.softmax(self.score(output), dim=1)
        context = torch.sum(output * weights, dim=1)
        return self.head(context)


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


class TransformerLite(nn.Module):
    def __init__(self, input_dim: int, horizon: int, model_dim: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos = PositionalEncoding(model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(model_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.pos(self.proj(x)))
        return self.head(encoded[:, -1, :])


class InformerLite(nn.Module):
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.distill = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1)
        self.transformer = TransformerLite(input_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.distill(x.transpose(1, 2)).transpose(1, 2)
        return self.transformer(x)


class AutoformerLite(nn.Module):
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.transformer = TransformerLite(input_dim, horizon)

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
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        attended, _ = self.attn(output, output, output)
        gated = attended[:, -1, :] * self.gate(attended[:, -1, :])
        return self.head(gated)


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
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x.transpose(1, 2)).mean(dim=-1)
        mixed = temporal + torch.relu(self.graph_gate(temporal))
        return self.head(mixed)


def build_dl_model(trainer: str, input_dim: int, horizon: int) -> nn.Module:
    hidden = 48
    if trainer == "rnn":
        return SequenceRNN("rnn", input_dim, hidden, horizon)
    if trainer == "lstm":
        return SequenceRNN("lstm", input_dim, hidden, horizon)
    if trainer == "gru":
        return SequenceRNN("gru", input_dim, hidden, horizon)
    if trainer == "bilstm":
        return SequenceRNN("lstm", input_dim, hidden, horizon, bidirectional=True)
    if trainer == "cnn_lstm":
        return CNNRNN("lstm", input_dim, hidden, horizon)
    if trainer == "cnn_gru":
        return CNNRNN("gru", input_dim, hidden, horizon)
    if trainer == "bilstm_attention":
        return BiLSTMAttention(input_dim, hidden, horizon)
    if trainer == "transformer":
        return TransformerLite(input_dim, horizon)
    if trainer == "informer_lite":
        return InformerLite(input_dim, horizon)
    if trainer == "autoformer_lite":
        return AutoformerLite(input_dim, horizon)
    if trainer == "tft_lite":
        return TFTLite(input_dim, hidden, horizon)
    if trainer == "stgcn_lite":
        return STGCNLite(input_dim, hidden, horizon)
    raise ValueError(f"Unsupported deep learning trainer: {trainer}")


def train_dl(
    spec: ModelSpec,
    supervised: dict[str, object],
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[BenchmarkRow, np.ndarray, nn.Module]:
    x_train = supervised["x_train"]
    y_train = supervised["y_train"]
    x_test = supervised["x_test"]
    y_test = supervised["y_test"]
    y_scaler = supervised["y_scaler"]
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_scaler, StandardScaler)

    model = build_dl_model(spec.trainer, x_train.shape[-1], y_train.shape[-1]).to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    start_train = time.time()
    model.train()
    for _ in range(epochs):
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            optimizer.step()
    train_seconds = time.time() - start_train

    start_infer = time.time()
    pred_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch = torch.tensor(x_test[start : start + batch_size], dtype=torch.float32, device=device)
            pred_batches.append(model(batch).cpu().numpy())
        latest = torch.tensor(supervised["latest_window"], dtype=torch.float32, device=device)
        future_scaled = model(latest).cpu().numpy().reshape(-1)
    inference_seconds = time.time() - start_infer

    pred_scaled = np.concatenate(pred_batches, axis=0)
    y_true = inverse_targets(y_scaler, y_test)
    y_pred = inverse_targets(y_scaler, pred_scaled)
    future_pred = inverse_targets(y_scaler, future_scaled.reshape(1, -1)).reshape(-1)
    metrics = metric_dict(y_true, y_pred)
    row = BenchmarkRow(
        city="",
        phase=spec.phase,
        model=spec.name,
        family=spec.family,
        status="ok",
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        **metrics,
    )
    return row, future_pred, model


def train_classical(
    spec: ModelSpec,
    supervised: dict[str, object],
    output_path: Path,
) -> tuple[BenchmarkRow, np.ndarray, object]:
    x_train = supervised["x_train"]
    y_train = supervised["y_train"]
    x_test = supervised["x_test"]
    y_test = supervised["y_test"]
    y_scaler = supervised["y_scaler"]
    latest_window = supervised["latest_window"]
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_scaler, StandardScaler)
    assert isinstance(latest_window, np.ndarray)

    if spec.trainer == "svr":
        estimator = SVR(C=10.0, gamma="scale", epsilon=0.05)
    elif spec.trainer == "random_forest":
        estimator = RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1, min_samples_leaf=2)
    elif spec.trainer == "xgboost":
        module = optional_import("xgboost")
        if isinstance(module, Exception):
            raise ImportError(f"xgboost unavailable: {module}")
        estimator = module.XGBRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    elif spec.trainer == "lightgbm":
        module = optional_import("lightgbm")
        if isinstance(module, Exception):
            raise ImportError(f"lightgbm unavailable: {module}")
        estimator = module.LGBMRegressor(
            n_estimators=260,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    elif spec.trainer == "catboost":
        module = optional_import("catboost")
        if isinstance(module, Exception):
            raise ImportError(f"catboost unavailable: {module}")
        estimator = module.CatBoostRegressor(iterations=220, depth=5, learning_rate=0.05, loss_function="RMSE", verbose=False)
    else:
        raise ValueError(f"Unsupported classical trainer: {spec.trainer}")

    model = MultiOutputRegressor(estimator)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    latest_flat = latest_window.reshape(1, -1)

    start_train = time.time()
    model.fit(x_train_flat, y_train)
    train_seconds = time.time() - start_train

    start_infer = time.time()
    pred_scaled = model.predict(x_test_flat)
    future_scaled = model.predict(latest_flat).reshape(-1)
    inference_seconds = time.time() - start_infer

    y_true = inverse_targets(y_scaler, y_test)
    y_pred = inverse_targets(y_scaler, pred_scaled)
    future_pred = inverse_targets(y_scaler, future_scaled.reshape(1, -1)).reshape(-1)
    metrics = metric_dict(y_true, y_pred)

    if joblib is not None:
        joblib.dump(
            {
                "model": model,
                "x_scaler": supervised["x_scaler"],
                "y_scaler": y_scaler,
                "features": list(FEATURE_COLUMNS),
                "target": TARGET_COLUMN,
            },
            output_path,
        )

    row = BenchmarkRow(
        city="",
        phase=spec.phase,
        model=spec.name,
        family=spec.family,
        status="ok",
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        **metrics,
    )
    return row, future_pred, model


def train_statistical(spec: ModelSpec, frame: pd.DataFrame, horizon: int) -> tuple[BenchmarkRow, np.ndarray]:
    train_df, _, test_df = split_frame(frame)
    train_target = train_df[TARGET_COLUMN].astype(float).to_numpy()
    test_target = test_df[TARGET_COLUMN].astype(float).to_numpy()
    if len(test_target) == 0:
        raise ValueError("No test rows available for statistical evaluation")

    start_train = time.time()
    if spec.trainer == "arima":
        module = optional_import("statsmodels.tsa.arima.model")
        if isinstance(module, Exception):
            raise ImportError(f"statsmodels unavailable: {module}")
        model = module.ARIMA(train_target, order=(2, 1, 2)).fit()
        train_seconds = time.time() - start_train
        start_infer = time.time()
        pred = np.asarray(model.forecast(steps=len(test_target)), dtype=float)
        future = np.asarray(model.forecast(steps=horizon), dtype=float)
    elif spec.trainer == "sarima":
        module = optional_import("statsmodels.tsa.statespace.sarimax")
        if isinstance(module, Exception):
            raise ImportError(f"statsmodels unavailable: {module}")
        model = module.SARIMAX(
            train_target,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        train_seconds = time.time() - start_train
        start_infer = time.time()
        pred = np.asarray(model.forecast(steps=len(test_target)), dtype=float)
        future = np.asarray(model.forecast(steps=horizon), dtype=float)
    elif spec.trainer == "var":
        module = optional_import("statsmodels.tsa.api")
        if isinstance(module, Exception):
            raise ImportError(f"statsmodels unavailable: {module}")
        train_values = train_df[VAR_COLUMNS].astype(float).to_numpy()
        model = module.VAR(train_values).fit(maxlags=24, ic="aic")
        train_seconds = time.time() - start_train
        start_infer = time.time()
        history = train_values[-model.k_ar :]
        forecast = model.forecast(history, steps=len(test_target))
        pred = forecast[:, VAR_COLUMNS.index(TARGET_COLUMN)]
        future = model.forecast(frame[VAR_COLUMNS].astype(float).to_numpy()[-model.k_ar :], steps=horizon)[
            :, VAR_COLUMNS.index(TARGET_COLUMN)
        ]
    else:
        raise ValueError(f"Unsupported statistical trainer: {spec.trainer}")
    inference_seconds = time.time() - start_infer

    metrics = metric_dict(test_target.reshape(1, -1), pred.reshape(1, -1))
    return (
        BenchmarkRow(
            city="",
            phase=spec.phase,
            model=spec.name,
            family=spec.family,
            status="ok",
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
            **metrics,
        ),
        future,
    )


def confidence_from_rmse(rmse: float | None) -> float:
    if rmse is None or not math.isfinite(rmse):
        return 0.0
    return float(max(0.05, min(0.99, 1.0 / (1.0 + rmse / 100.0))))


def safe_model_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")


def forecast_rows(city: str, city_frame: pd.DataFrame, model_name: str, confidence: float, forecast: np.ndarray) -> list[dict[str, object]]:
    last_ts = pd.Timestamp(city_frame["timestamp"].max()).tz_convert(timezone.utc)
    forecast_timestamp = pd.Timestamp.utcnow()
    rows = []
    for index, predicted_aqi in enumerate(forecast, start=1):
        horizon_timestamp = last_ts + pd.Timedelta(hours=index)
        rows.append(
            {
                "forecast_id": f"{city}-{safe_model_name(model_name)}-{forecast_timestamp.strftime('%Y%m%dT%H%M%SZ')}-{index:03d}",
                "city_slug": city,
                "city_name": city.title(),
                "forecast_timestamp": forecast_timestamp.isoformat(),
                "horizon_timestamp": horizon_timestamp.isoformat(),
                "horizon_hours": index,
                "predicted_us_aqi": float(predicted_aqi),
                "model_type": model_name,
                "model_version": "kaggle-benchmark-v1",
                "confidence": confidence,
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_city(city: str, frame: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> tuple[list[BenchmarkRow], dict[str, object], list[dict[str, object]]]:
    output_dir = Path(args.output_dir)
    city_dir = output_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    supervised = prepare_supervised(frame, args.lookback, args.horizon, args.max_train_windows, args.max_test_windows)
    rows: list[BenchmarkRow] = []
    futures: dict[str, np.ndarray] = {}

    for spec in MODEL_SPECS:
        print(f"[{city}] Training {spec.name}...")
        try:
            if spec.family == "statistical":
                row, future = train_statistical(spec, frame, args.horizon)
            elif spec.family == "classical_ml":
                artifact_path = city_dir / f"{safe_model_name(spec.name)}.joblib"
                row, future, _ = train_classical(spec, supervised, artifact_path)
            elif spec.family == "deep_learning":
                row, future, model = train_dl(spec, supervised, device, args.epochs, args.batch_size)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": spec.name,
                        "trainer": spec.trainer,
                        "features": list(FEATURE_COLUMNS),
                        "lookback": args.lookback,
                        "horizon": args.horizon,
                        "target": TARGET_COLUMN,
                    },
                    city_dir / f"{safe_model_name(spec.name)}.pt",
                )
            else:
                raise ValueError(f"Unknown family: {spec.family}")
            row.city = city
            rows.append(row)
            futures[spec.name] = future
        except Exception as exc:  # noqa: BLE001 - benchmark matrix should continue.
            rows.append(
                BenchmarkRow(
                    city=city,
                    phase=spec.phase,
                    model=spec.name,
                    family=spec.family,
                    status="skipped",
                    reason=str(exc),
                )
            )
            print(f"[{city}] Skipped {spec.name}: {exc}")

    successful = [row for row in rows if row.status == "ok" and row.rmse is not None and math.isfinite(row.rmse)]
    if not successful:
        best = {"city": city, "status": "no_successful_model"}
        return rows, best, []

    best_row = min(successful, key=lambda row: row.rmse or float("inf"))
    best = {
        "city": city,
        "model": best_row.model,
        "phase": best_row.phase,
        "rmse": best_row.rmse,
        "mae": best_row.mae,
        "r2": best_row.r2,
        "confidence": confidence_from_rmse(best_row.rmse),
        "artifact_dir": str(city_dir),
    }
    rows_for_db = forecast_rows(city, frame, best_row.model, best["confidence"], futures[best_row.model])
    return rows, best, rows_for_db


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    set_seed(args.seed)
    device = infer_device(args.device)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[BenchmarkRow] = []
    best_models: list[dict[str, object]] = []
    all_forecasts: list[dict[str, object]] = []

    for city in TARGET_CITIES:
        frame = load_city_frame(data_dir, city)
        rows, best, forecasts = run_city(city, frame, args, device)
        all_rows.extend(rows)
        best_models.append(best)
        all_forecasts.extend(forecasts)

    summary = pd.DataFrame([asdict(row) for row in all_rows])
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    forecasts_path = output_dir / "forecast_rows.csv"
    pd.DataFrame(all_forecasts).to_csv(forecasts_path, index=False)

    results_payload = {
        "config": {
            "cities": TARGET_CITIES,
            "features": list(FEATURE_COLUMNS),
            "target": TARGET_COLUMN,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "epochs": args.epochs,
            "device": str(device),
        },
        "models": [asdict(spec) for spec in MODEL_SPECS],
        "best_models": best_models,
        "rows": [asdict(row) for row in all_rows],
        "summary_csv": str(summary_path),
        "forecast_rows_csv": str(forecasts_path),
    }
    write_json(output_dir / "benchmarks_results.json", results_payload)

    # Also place canonical copies in the dataset directory for easy Kaggle output download.
    summary.to_csv(data_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(all_forecasts).to_csv(data_dir / "forecast_rows.csv", index=False)
    write_json(data_dir / "benchmarks_results.json", results_payload)

    print(json.dumps({"summary_csv": str(summary_path), "forecast_rows_csv": str(forecasts_path)}, indent=2))


if __name__ == "__main__":
    main()
