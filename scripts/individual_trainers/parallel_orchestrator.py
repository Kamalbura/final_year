import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CITIES = ["delhi", "hyderabad", "bengaluru"]
TRAINERS = [
    "scripts/individual_trainers/statistical_trainers.py",
    "scripts/individual_trainers/ensemble_trainers.py",
    "scripts/individual_trainers/sequence_trainers.py",
    "scripts/individual_trainers/advanced_trainers.py",
    "scripts/individual_trainers/tft_trainer.py",
    "scripts/individual_trainers/spatiotemporal_trainers.py",
    "scripts/individual_trainers/rf_trainer.py",
]

def run_trainer(trainer_script, city, trials=5):
    print(f"Starting {trainer_script} for {city}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)
    
    # Handle multi-model scripts
    if "statistical_trainers.py" in trainer_script:
        models = ["ARIMA", "SARIMA", "VAR"]
    elif "ensemble_trainers.py" in trainer_script:
        models = ["XGBoost", "LightGBM", "CatBoost", "SVR"]
    elif "sequence_trainers.py" in trainer_script:
        models = ["RNN", "LSTM", "GRU", "BiLSTM"]
    elif "advanced_trainers.py" in trainer_script:
        models = ["CNN-LSTM", "CNN-GRU", "BiLSTM-Attention"]
    else:
        models = [None]

    for model in models:
        cmd = [
            "conda", "run", "-n", "dl-env", 
            "python", trainer_script, 
            "--city", city, 
            "--trials", str(trials)
        ]
        if model:
            cmd.extend(["--model", model])
            print(f"  Training sub-model: {model}")
            
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"Error in {trainer_script} ({model}) for {city}:\n{result.stderr}")
    
    return 0

def main():
    tasks = []
    for city in CITIES:
        for trainer in TRAINERS:
            tasks.append((trainer, city))
    
    print(f"Orchestrating {len(tasks)} training tasks in parallel...")
    start_time = time.time()
    
    # Using 3 workers to handle the 3 cities in parallel
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_trainer, trainer, city) for trainer, city in tasks]
        results = [f.result() for f in futures]
        
    duration = time.time() - start_time
    print(f"All tasks completed in {duration:.2f} seconds")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
