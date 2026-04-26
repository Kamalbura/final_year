from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformers import TransformerForecaster
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 7-day AQI forecasting models for a city dataset.")
    parser.add_argument("--city-csv", required=True, help="Path to city CSV with us_aqi and feature columns")
    parser.add_argument("--output-dir", default="outputs/7day", help="Directory for model artifacts")
    parser.add_argument("--lookback", type=int, default=168, help="Input history window in hours")
    parser.add_argument("--horizon", type=int, default=168, help="Forecast horizon in hours (168 = 7 days)")
    parser.add_argument("--epochs", type=int, default=8, help="Transformer training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


def load_city_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ["timestamp", *FEATURE_COLUMNS]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = df[required_cols].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    frame = frame.dropna(subset=FEATURE_COLUMNS)
    if frame.empty:
        raise ValueError("No valid rows after cleaning the city dataset")
    return frame.reset_index(drop=True)


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def window_xy(x_values: np.ndarray, y_values: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    limit = len(x_values) - lookback - horizon + 1
    if limit <= 0:
        raise ValueError(f"Not enough rows for lookback={lookback}, horizon={horizon}")
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for i in range(limit):
        x_list.append(x_values[i : i + lookback])
        y_list.append(y_values[i + lookback : i + lookback + horizon])
    return np.array(x_list), np.array(y_list)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def train_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
) -> TransformerForecaster:
    device = torch.device("cpu")
    model = TransformerForecaster(input_dim=x_train.shape[-1], horizon=horizon, output_dim=1)
    model.to(device)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(y_train.shape[0], -1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).reshape(xb.shape[0], -1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _ = model(torch.tensor(x_val[: min(len(x_val), 32)], dtype=torch.float32).to(device))
    return model


def infer_transformer(model: TransformerForecaster, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return pred.squeeze(-1)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    city_frame = load_city_frame(args.city_csv)
    city_name = str(city_frame.iloc[0].get("city", Path(args.city_csv).stem))

    train_df, val_df, test_df = split_frame(city_frame)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_raw = train_df[FEATURE_COLUMNS].to_numpy()
    x_val_raw = val_df[FEATURE_COLUMNS].to_numpy()
    x_test_raw = test_df[FEATURE_COLUMNS].to_numpy()

    y_train_raw = train_df[["us_aqi"]].to_numpy()
    y_val_raw = val_df[["us_aqi"]].to_numpy()
    y_test_raw = test_df[["us_aqi"]].to_numpy()

    x_train_scaled = x_scaler.fit_transform(x_train_raw)
    x_val_scaled = x_scaler.transform(x_val_raw)
    x_test_scaled = x_scaler.transform(x_test_raw)

    y_train_scaled = y_scaler.fit_transform(y_train_raw).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val_raw).reshape(-1)
    y_test_scaled = y_scaler.transform(y_test_raw).reshape(-1)

    x_train, y_train = window_xy(x_train_scaled, y_train_scaled, args.lookback, args.horizon)
    x_val, y_val = window_xy(x_val_scaled, y_val_scaled, args.lookback, args.horizon)
    x_test, y_test = window_xy(x_test_scaled, y_test_scaled, args.lookback, args.horizon)

    y_train_2d = y_train
    y_val_2d = y_val
    y_test_2d = y_test

    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=240, random_state=args.seed, n_jobs=-1))
    rf.fit(x_train.reshape(x_train.shape[0], -1), y_train_2d)

    rf_val_pred_scaled = rf.predict(x_val.reshape(x_val.shape[0], -1))
    rf_test_pred_scaled = rf.predict(x_test.reshape(x_test.shape[0], -1))

    transformer = train_transformer(
        x_train=x_train,
        y_train=y_train_2d,
        x_val=x_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
    )
    tf_val_pred_scaled = infer_transformer(transformer, x_val)
    tf_test_pred_scaled = infer_transformer(transformer, x_test)

    def inv_targets(arr: np.ndarray) -> np.ndarray:
        return y_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)

    y_val_true = inv_targets(y_val_2d)
    y_test_true = inv_targets(y_test_2d)

    rf_val_pred = inv_targets(rf_val_pred_scaled)
    rf_test_pred = inv_targets(rf_test_pred_scaled)
    tf_val_pred = inv_targets(tf_val_pred_scaled)
    tf_test_pred = inv_targets(tf_test_pred_scaled)

    metrics = {
        "random_forest": {
            "val_rmse": rmse(y_val_true, rf_val_pred),
            "val_mae": mae(y_val_true, rf_val_pred),
            "test_rmse": rmse(y_test_true, rf_test_pred),
            "test_mae": mae(y_test_true, rf_test_pred),
        },
        "transformer": {
            "val_rmse": rmse(y_val_true, tf_val_pred),
            "val_mae": mae(y_val_true, tf_val_pred),
            "test_rmse": rmse(y_test_true, tf_test_pred),
            "test_mae": mae(y_test_true, tf_test_pred),
        },
    }

    best_model = min(metrics.items(), key=lambda kv: kv[1]["val_rmse"])[0]

    output_dir = Path(args.output_dir) / city_name.lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, output_dir / "random_forest.pkl")
    torch.save(transformer.state_dict(), output_dir / "transformer.pth")
    joblib.dump(x_scaler, output_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")

    metadata = {
        "city": city_name,
        "features": FEATURE_COLUMNS,
        "target": "us_aqi",
        "lookback": args.lookback,
        "horizon": args.horizon,
        "best_model": best_model,
        "metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"model": "random_forest", **metrics["random_forest"]},
            {"model": "transformer", **metrics["transformer"]},
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)

    print(json.dumps({"city": city_name, "best_model": best_model, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
