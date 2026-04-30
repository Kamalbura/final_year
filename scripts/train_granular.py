import sys
import os
import time
from pathlib import Path
import pandas as pd
import torch

# Add current directory to sys.path to import local modules
sys.path.append(os.getcwd())
import scripts.kaggle_benchmarking_suite as suite

def run_granular_training():
    print("--- Starting Granular Training Session ---")
    DATA_DIR = Path("data/kaggle_dataset")
    OUTPUT_DIR = Path("outputs/granular_logs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cities = ["delhi", "hyderabad", "bengaluru"]
    
    # We'll pick one representative model from each family to keep it manageable but detailed
    target_specs = [
        suite.ModelSpec("Phase 2", "Random Forest", "classical_ml", "random_forest"),
        suite.ModelSpec("Phase 3", "LSTM", "deep_learning", "lstm"),
        suite.ModelSpec("Phase 4", "Transformer", "deep_learning", "transformer")
    ]
    
    device = suite.infer_device("auto")
    print(f"Target Device: {device}")
    
    for city in cities:
        print(f"\n[CITY: {city.upper()}] Loading data...")
        try:
            frame = suite.load_city_frame(DATA_DIR, city)
            supervised = suite.prepare_supervised(
                frame, 
                lookback=168, 
                horizon=24, 
                max_train_windows=800, 
                max_test_windows=200
            )
            
            for spec in target_specs:
                print(f"\n>>> Starting training for Model: {spec.name}")
                start_time = time.time()
                
                if spec.family == "classical_ml":
                    artifact_path = OUTPUT_DIR / f"{city}_{suite.safe_model_name(spec.name)}.joblib"
                    row, future, _ = suite.train_classical(spec, supervised, artifact_path)
                elif spec.family == "deep_learning":
                    # Reduced epochs for quicker log feedback
                    row, future, model = suite.train_dl(spec, supervised, device, epochs=3, batch_size=64)
                
                duration = time.time() - start_time
                print(f"DONE: {spec.name} in {duration:.2f}s")
                print(f"LOG: RMSE={row.rmse:.4f}, MAE={row.mae:.4f}, R2={row.r2:.4f}")
                
        except Exception as e:
            print(f"ERROR processing {city}: {e}")

if __name__ == "__main__":
    run_granular_training()
