from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


INPUT_DIR = Path("/kaggle/input/indian-air-quality-3-city-benchmark-1y")
WORK_DATA_DIR = Path("/kaggle/working/data/kaggle_dataset")
OUTPUT_DIR = Path("/kaggle/working/outputs/kaggle_benchmarks")

WORK_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Preparing Kaggle working dataset...")
for item in INPUT_DIR.iterdir():
    if item.is_file():
        shutil.copy2(item, WORK_DATA_DIR / item.name)

print("Installing optional benchmark dependencies...")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "statsmodels",
        "xgboost",
        "lightgbm",
        "catboost",
    ],
    check=False,
)

cmd = [
    sys.executable,
    str(WORK_DATA_DIR / "kaggle_benchmarking_suite.py"),
    "--data-dir",
    str(WORK_DATA_DIR),
    "--output-dir",
    str(OUTPUT_DIR),
    "--lookback",
    "168",
    "--horizon",
    "24",
    "--epochs",
    "6",
    "--batch-size",
    "64",
    "--max-train-windows",
    "1600",
    "--max-test-windows",
    "400",
    "--device",
    "auto",
]
print("Running AQI benchmark:", " ".join(cmd))
subprocess.run(cmd, check=True)

summary = pd.read_csv(OUTPUT_DIR / "benchmark_summary.csv")
ok_summary = summary[summary["status"] == "ok"].sort_values("rmse")
print(ok_summary.head(20).to_string(index=False))
