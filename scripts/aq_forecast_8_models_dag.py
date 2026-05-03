"""
Airflow DAG: PM2.5 Forecast using all 8 production models
Runs sequentially every hour: each model 10 seconds apart
Model order: TIER 1 (best) -> TIER 2 -> TIER 3

DAG schedule: minute 15 each hour (after data ingestion completes)
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

REPO_ROOT = Path("/home/bura/projects/final_year")
PREDICT_SCRIPT = REPO_ROOT / "deployment_models" / "predict_server.py"
VENV_PYTHON = "/home/bura/venvs/ml/bin/python3"

# 8 models in priority order (Tier 1 -> Tier 2 -> Tier 3)
MODEL_ORDER = [
    "xgboost",       # Tier 1 - Best overall
    "lightgbm",      # Tier 1
    "random_forest", # Tier 1
    "cnn_lstm",      # Tier 2 - Best DL
    "gru",           # Tier 2
    "transformer",   # Tier 2 - Best DL accuracy
    "svr",           # Tier 3
    "bilstm",        # Tier 3
]

MODEL_NAMES = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest",
    "cnn_lstm": "CNN-LSTM", "gru": "GRU", "transformer": "Transformer",
    "svr": "SVR", "bilstm": "BiLSTM"
}

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="aq_forecast_8_models",
    description="PM2.5 24h forecast using 8 production models (sequential, 10s apart)",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    schedule="15 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["aq", "pi", "forecast", "ml"],
) as dag:
    
    prev_task = None
    
    for i, model_key in enumerate(MODEL_ORDER):
        model_name = MODEL_NAMES[model_key]
        
        # Schedule: 15:00, 15:10, 15:20, 15:30, 15:40, 15:50, 16:00, 16:10
        delay_seconds = 15 + i * 10  # 15, 25, 35, 45, 55, 65, 75, 85 sec after the minute
        
        task = BashOperator(
            task_id=f"predict_{model_key}",
            bash_command=f"sleep {delay_seconds} && {VENV_PYTHON} {PREDICT_SCRIPT} {model_key}",
            cwd=str(REPO_ROOT),
            execution_timeout=timedelta(seconds=120),
        )
        
        if prev_task:
            prev_task >> task
        prev_task = task
