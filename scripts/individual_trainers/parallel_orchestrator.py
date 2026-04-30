import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CITIES = ["delhi", "hyderabad", "bengaluru"]
TRAINERS = [
    "scripts/individual_trainers/lstm_trainer.py",
    "scripts/individual_trainers/rf_trainer.py",
    "scripts/individual_trainers/transformer_trainer.py",
]

def run_trainer(trainer_script, city, trials=5):
    print(f"Starting {trainer_script} for {city}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)
    cmd = [
        "conda", "run", "-n", "dl-env", 
        "python", trainer_script, 
        "--city", city, 
        "--trials", str(trials)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print(f"Finished {trainer_script} for {city}")
    else:
        print(f"Error in {trainer_script} for {city}:\n{result.stderr}")
    return result.returncode

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
