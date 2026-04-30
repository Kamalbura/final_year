import sys
import os
import time
from pathlib import Path
import pandas as pd
import torch

# Add current directory to sys.path
sys.path.append(os.getcwd())
import scripts.kaggle_benchmarking_suite as suite

print("--- DEBUG TRAINING START ---")
DATA_DIR = Path("data/kaggle_dataset")

# 1. Load data
city = "delhi"
print(f"Loading {city}...")
frame = suite.load_city_frame(DATA_DIR, city)
print(f"Frame loaded: {len(frame)} rows")

# 2. Prepare supervised
print("Preparing supervised...")
supervised = suite.prepare_supervised(
    frame, 
    lookback=168, 
    horizon=24, 
    max_train_windows=50, # VERY SMALL
    max_test_windows=10
)
print("Supervised ready.")

# 3. Train one DL model for 1 epoch
spec = suite.ModelSpec("Phase 3", "LSTM", "deep_learning", "lstm")
device = suite.infer_device("auto")
print(f"Training {spec.name} on {device}...")

row, future, model = suite.train_dl(spec, supervised, device, epochs=1, batch_size=32)
print(f"DONE: RMSE={row.rmse:.4f}")
print("--- DEBUG TRAINING END ---")
