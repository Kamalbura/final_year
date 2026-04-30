"""
Airflow DAG for AQI Forecasting
Generates and loads 24-hour ahead forecasts hourly for Delhi, Hyderabad, and Bengaluru
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


# Paths
REPO_ROOT = Path("/home/bura/projects/final_year")
VENV_PYTHON = "/home/bura/venvs/ml/bin/python3"
FORECAST_GENERATOR = REPO_ROOT / "deployment_models" / "pi_forecast_generator.py"
FORECAST_LOADER = REPO_ROOT / "deployment_models" / "pi_forecast_loader.py"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _log_forecast_start():
    """Log forecast generation start"""
    print(f"Starting AQI forecast generation at {datetime.now(timezone.utc)}")


def _log_forecast_complete():
    """Log forecast generation complete"""
    print(f"Completed AQI forecast generation at {datetime.now(timezone.utc)}")


with DAG(
    dag_id="aq_forecast_hourly",
    description="Generate and load 24-hour ahead AQI forecasts for 3 cities",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 4, 27, tzinfo=timezone.utc),
    schedule="15 * * * *",  # Run at 15 minutes past every hour (after ingestion)
    catchup=False,
    max_active_runs=1,
    tags=["aq", "pi", "forecast", "ml"],
) as dag:
    
    log_start = PythonOperator(
        task_id="log_forecast_start",
        python_callable=_log_forecast_start,
    )
    
    generate_forecasts = BashOperator(
        task_id="generate_forecasts",
        bash_command=f"source /home/bura/venvs/ml/bin/activate && {VENV_PYTHON} {FORECAST_GENERATOR}",
        cwd=str(REPO_ROOT),
    )
    
    load_forecasts = BashOperator(
        task_id="load_forecasts_to_db",
        bash_command=f"source /home/bura/venvs/ml/bin/activate && {VENV_PYTHON} {FORECAST_LOADER}",
        cwd=str(REPO_ROOT),
    )
    
    log_complete = PythonOperator(
        task_id="log_forecast_complete",
        python_callable=_log_forecast_complete,
    )
    
    # Task dependencies
    log_start >> generate_forecasts >> load_forecasts >> log_complete
