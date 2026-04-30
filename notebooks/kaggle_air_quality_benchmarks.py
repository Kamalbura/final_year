# Kaggle GPU notebook entrypoint for the final_year AQI benchmark suite.
#
# Expected attached dataset:
#   kamalbura/indian-air-quality-3-city-benchmark-1y
#
# The dataset bundle should contain:
#   clean_delhi_aq_1y.csv
#   clean_hyderabad_aq_1y.csv
#   clean_bengaluru_aq_1y.csv
#   kaggle_benchmarking_suite.py

from pathlib import Path
import shutil
import subprocess
import sys


INPUT_DIR = Path("/kaggle/input/indian-air-quality-3-city-benchmark-1y")
WORK_DATA_DIR = Path("/kaggle/working/data/kaggle_dataset")
OUTPUT_DIR = Path("/kaggle/working/outputs/kaggle_benchmarks")

WORK_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for item in INPUT_DIR.iterdir():
    destination = WORK_DATA_DIR / item.name
    if item.is_file():
        shutil.copy2(item, destination)

print("Working data files:")
for item in sorted(WORK_DATA_DIR.iterdir()):
    print(" -", item)


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
    "168",
    "--epochs",
    "8",
    "--batch-size",
    "64",
    "--max-train-windows",
    "1200",
    "--max-test-windows",
    "300",
    "--device",
    "auto",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)


print("\nBenchmark artifacts:")
for item in sorted(OUTPUT_DIR.glob("*")):
    print(" -", item)

print("\nTop benchmark rows:")
import pandas as pd

summary = pd.read_csv(OUTPUT_DIR / "benchmark_summary.csv")
print(summary[summary["status"] == "ok"].sort_values("rmse").head(20).to_string(index=False))
