from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformers import TransformerForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate next-7-day AQI forecast from trained city model.")
    parser.add_argument("--city-csv", required=True, help="City CSV path")
    parser.add_argument("--model-dir", required=True, help="Directory produced by train_city_7day.py")
    parser.add_argument("--output", default="outputs/7day/latest_forecast.csv", help="Forecast CSV output path")
    parser.add_argument("--trend-output", default="outputs/7day/latest_trend.json", help="Trend JSON output path")
    return parser.parse_args()


def summarize_trend(values: np.ndarray) -> dict:
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0
    mean_val = float(np.mean(values))
    max_val = float(np.max(values))

    if slope > 0.15:
        direction = "rising"
    elif slope < -0.15:
        direction = "falling"
    else:
        direction = "stable"

    if max_val >= 200:
        risk = "very_unhealthy"
    elif max_val >= 150:
        risk = "unhealthy"
    elif max_val >= 100:
        risk = "moderate"
    else:
        risk = "good_to_satisfactory"

    return {
        "trend_direction": direction,
        "slope_per_hour": slope,
        "mean_forecast_aqi": mean_val,
        "max_forecast_aqi": max_val,
        "risk_band": risk,
    }


def load_city_frame(path: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["timestamp", *features]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"City CSV missing columns: {missing}")
    frame = df[needed].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", *features]).sort_values("timestamp")
    if frame.empty:
        raise ValueError("No valid rows in city data")
    return frame


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    features = list(metadata["features"])
    lookback = int(metadata["lookback"])
    horizon = int(metadata["horizon"])
    best_model = str(metadata["best_model"])

    frame = load_city_frame(args.city_csv, features)
    if len(frame) < lookback:
        raise ValueError(f"Need at least {lookback} rows for forecasting")

    x_scaler = joblib.load(model_dir / "x_scaler.pkl")
    y_scaler = joblib.load(model_dir / "y_scaler.pkl")

    latest_window = frame[features].tail(lookback).to_numpy()
    latest_scaled = x_scaler.transform(latest_window).reshape(1, lookback, len(features))

    if best_model == "random_forest":
        rf = joblib.load(model_dir / "random_forest.pkl")
        pred_scaled = rf.predict(latest_scaled.reshape(1, -1)).reshape(-1)
    elif best_model == "transformer":
        model = TransformerForecaster(input_dim=len(features), horizon=horizon, output_dim=1)
        state = torch.load(model_dir / "transformer.pth", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(latest_scaled, dtype=torch.float32)).numpy().reshape(-1)
    else:
        raise ValueError(f"Unsupported model: {best_model}")

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

    last_ts = frame["timestamp"].iloc[-1]
    forecast_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="H", tz="UTC")

    forecast_df = pd.DataFrame({"timestamp": forecast_index, "predicted_us_aqi": pred})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)

    trend = summarize_trend(pred)
    trend["model"] = best_model
    trend["city"] = str(frame.iloc[-1].get("city", Path(args.city_csv).stem))
    trend["forecast_start_utc"] = str(forecast_index.min())
    trend["forecast_end_utc"] = str(forecast_index.max())

    trend_path = Path(args.trend_output)
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    trend_path.write_text(json.dumps(trend, indent=2), encoding="utf-8")

    print(json.dumps({"forecast_csv": str(output_path), "trend_json": str(trend_path), "model": best_model}, indent=2))


if __name__ == "__main__":
    main()
