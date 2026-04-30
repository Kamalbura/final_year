import sys
import os
import time
from pathlib import Path
import pandas as pd
import torch
from dataclasses import asdict

# Add current directory to sys.path
sys.path.append(os.getcwd())
import scripts.kaggle_benchmarking_suite as suite

def run_exhaustive_local():
    print("--- [LOCAL GPU TRAINING] Starting exhaustive benchmark ---")
    DATA_DIR = Path("data/kaggle_dataset")
    OUTPUT_DIR = Path("outputs/local_exhaustive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = suite.infer_device("auto")
    print(f"Device: {device}")
    
    all_results = []
    
    for city in ["delhi", "hyderabad", "bengaluru"]:
        print(f"\n[{city.upper()}] Loading data...")
        try:
            frame = suite.load_city_frame(DATA_DIR, city)
            supervised = suite.prepare_supervised(
                frame, 
                lookback=168, 
                horizon=24, 
                max_train_windows=1200, 
                max_test_windows=300
            )
            
            for spec in suite.MODEL_SPECS:
                print(f"  > Model: {spec.name} ({spec.family})...", end=" ", flush=True)
                start = time.time()
                try:
                    if spec.family == "statistical":
                        row, _ = suite.train_statistical(spec, frame, 24)
                    elif spec.family == "classical_ml":
                        artifact = OUTPUT_DIR / f"{city}_{suite.safe_model_name(spec.name)}.joblib"
                        row, _, _ = suite.train_classical(spec, supervised, artifact)
                    else:
                        row, _, _ = suite.train_dl(spec, supervised, device, epochs=5, batch_size=64)
                    
                    row.city = city
                    duration = time.time() - start
                    print(f"DONE ({duration:.1f}s) | RMSE: {row.rmse:.4f}")
                    all_results.append(row)
                except Exception as e:
                    print(f"FAILED: {e}")
                    
            # Save partial progress
            pd.DataFrame([asdict(r) for r in all_results]).to_csv(OUTPUT_DIR / "partial_summary.csv", index=False)
            
        except Exception as e:
            print(f"Error loading {city}: {e}")

    print("\n--- Exhaustive Local Training Complete ---")

if __name__ == "__main__":
    run_exhaustive_local()
