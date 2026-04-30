import os
import sys
import time
import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

# Add root to path
sys.path.append(os.getcwd())
import scripts.kaggle_benchmarking_suite as suite

# Best Practice: Structured Logging
LOG_DIR = Path("logs/training")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_single_model(city, model_name, data_dir, output_dir, epochs=10):
    """
    Methodically trains a single model for a single city.
    Following best practices: Memory clearing, checkpointing, and detailed logging.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Resource Management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU Memory Cleared. Using: {torch.cuda.get_device_name(0)}")

    # 2. Find Spec
    spec = next((s for s in suite.MODEL_SPECS if s.name == model_name), None)
    if not spec:
        logger.error(f"Model spec not found for: {model_name}")
        return

    logger.info(f"--- STARTING: {model_name} for {city.upper()} ---")
    
    try:
        # 3. Data Loading
        frame = suite.load_city_frame(Path(data_dir), city)
        # Using larger window counts for local training best practices
        supervised = suite.prepare_supervised(
            frame, 
            lookback=168, 
            horizon=24, 
            max_train_windows=2000, 
            max_test_windows=500
        )
        
        device = suite.infer_device("auto")
        start_time = time.time()

        # 4. Training
        if spec.family == "statistical":
            row, _ = suite.train_statistical(spec, frame, 24)
        elif spec.family == "classical_ml":
            artifact_path = output_dir / f"{city}_{suite.safe_model_name(spec.name)}.joblib"
            row, _, _ = suite.train_classical(spec, supervised, artifact_path)
        else:
            row, _, model = suite.train_dl(spec, supervised, device, epochs=epochs, batch_size=64)
            # Checkpointing deep learning models
            ckpt_path = output_dir / f"{city}_{suite.safe_model_name(spec.name)}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        row.city = city
        duration = time.time() - start_time
        
        # 5. Persistence of Results
        result_file = output_dir / f"result_{city}_{suite.safe_model_name(model_name)}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(row), f, indent=4)
            
        logger.info(f"COMPLETED: {model_name} in {duration:.1f}s | RMSE: {row.rmse:.4f} | R2: {row.r2:.4f}")
        return row

    except Exception as e:
        logger.exception(f"FAILED: {model_name} for {city}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="data/kaggle_dataset")
    parser.add_argument("--output-dir", default="outputs/local_refined")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    train_single_model(args.city, args.model, args.data_dir, args.output_dir, args.epochs)
