from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor forecast drift and retrain when threshold is exceeded.")
    parser.add_argument("--city-csv", required=True, help="City CSV with actual us_aqi")
    parser.add_argument("--forecast-csv", required=True, help="Latest forecast CSV output")
    parser.add_argument("--model-dir", required=True, help="Model output directory")
    parser.add_argument("--drift-threshold", type=float, default=15.0, help="MAPE threshold percentage")
    parser.add_argument("--train-command", default="", help="Custom retraining command")
    parser.add_argument("--forecast-command", default="", help="Custom forecast refresh command")
    return parser.parse_args()


def compute_mape(actual: pd.Series, predicted: pd.Series) -> float:
    denom = actual.abs().replace(0, pd.NA).dropna()
    aligned = predicted.loc[denom.index]
    if denom.empty:
        return 0.0
    return float(((aligned - denom).abs() / denom).mean() * 100)


def main() -> None:
    args = parse_args()

    actual_df = pd.read_csv(args.city_csv, usecols=["timestamp", "us_aqi"])
    pred_df = pd.read_csv(args.forecast_csv, usecols=["timestamp", "predicted_us_aqi"])

    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"], utc=True, errors="coerce")
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True, errors="coerce")

    merged = pd.merge(pred_df, actual_df, on="timestamp", how="inner").dropna()
    if merged.empty:
        print("No overlapping timestamps yet; skipping drift check.")
        return

    mape = compute_mape(merged["us_aqi"], merged["predicted_us_aqi"])
    print(f"Current drift (MAPE): {mape:.2f}%")

    if mape <= args.drift_threshold:
        print("Drift is within threshold. No retraining triggered.")
        return

    print("Drift exceeded threshold. Triggering retraining...")

    if args.train_command:
        train_cmd = args.train_command
    else:
        train_cmd = (
            f"python scripts/train_city_7day.py --city-csv \"{args.city_csv}\" "
            f"--output-dir \"{Path(args.model_dir).parent}\""
        )

    train_result = subprocess.run(train_cmd, shell=True, check=False)
    if train_result.returncode != 0:
        raise SystemExit("Retraining command failed")

    if args.forecast_command:
        forecast_cmd = args.forecast_command
    else:
        forecast_cmd = (
            f"python scripts/forecast_city_7day.py --city-csv \"{args.city_csv}\" "
            f"--model-dir \"{args.model_dir}\" --output \"{args.forecast_csv}\""
        )

    forecast_result = subprocess.run(forecast_cmd, shell=True, check=False)
    if forecast_result.returncode != 0:
        raise SystemExit("Forecast refresh command failed")

    print("Retraining and forecast refresh completed.")


if __name__ == "__main__":
    main()
