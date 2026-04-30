from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


INPUT_ROOT = Path("/kaggle/input")
EXPECTED_INPUT_DIR = INPUT_ROOT / "indian-air-quality-3-city-benchmark-1y"
WORK_DATA_DIR = Path("/kaggle/working/data/kaggle_dataset")
OUTPUT_DIR = Path("/kaggle/working/outputs/kaggle_benchmarks")

def resolve_input_dir() -> Path:
    print("Available Kaggle input directories:")
    if INPUT_ROOT.exists():
        for candidate in sorted(INPUT_ROOT.iterdir()):
            print(f"  - {candidate}")
    else:
        print("  <missing /kaggle/input>")

    candidates = [EXPECTED_INPUT_DIR]
    if INPUT_ROOT.exists():
        candidates.extend(path for path in INPUT_ROOT.iterdir() if path.is_dir())

    for candidate in candidates:
        if (candidate / "kaggle_benchmarking_suite.py").exists():
            return candidate

    if INPUT_ROOT.exists():
        matches = sorted(INPUT_ROOT.rglob("kaggle_benchmarking_suite.py"))
        for match in matches:
            print(f"  found benchmark script at {match}")
        if matches:
            return matches[0].parent

    raise FileNotFoundError(
        "Could not find attached benchmark dataset with kaggle_benchmarking_suite.py under /kaggle/input"
    )


WORK_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_dir = resolve_input_dir()
print(f"Using Kaggle input directory: {input_dir}")
for item in input_dir.iterdir():
    if item.is_file():
        shutil.copy2(item, WORK_DATA_DIR / item.name)

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
        "catboost"
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
    "auto"
]
subprocess.run(cmd, check=True)

summary = pd.read_csv(OUTPUT_DIR / "benchmark_summary.csv")
print(summary[summary["status"] == "ok"].sort_values("rmse").head(20).to_string(index=False))
