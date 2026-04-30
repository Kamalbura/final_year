from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path


REPO_ROOT = Path("/opt/final_year")
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airflow import DAG  # type: ignore[import-untyped]
from airflow.operators.python import PythonOperator  # type: ignore[import-untyped]

from src.data.cities import ALL_MAJOR_CITIES, City, dag_id_for_city
from src.ingestion.india_aq import IngestionSettings, connect, run_incremental_cycle_for_city
from scripts.migrate_aggregates import refresh_materialized_views


DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _dag_id_for_city(city: City) -> str:
    return dag_id_for_city(city)


def _description_for_city(city: City) -> str:
    return f"City-isolated incremental air-quality ingestion for {city.name}, {city.country}"


def _run_city(city_slug: str, pipeline_name: str) -> dict[str, object]:
    settings = IngestionSettings.from_env()
    settings.archive_root.mkdir(parents=True, exist_ok=True)
    return run_incremental_cycle_for_city(settings, city_slug, pipeline_name=pipeline_name)


def _refresh_warehouse_views() -> None:
    settings = IngestionSettings.from_env()
    connection = connect(settings.dsn)
    try:
        refresh_materialized_views(connection)
    finally:
        connection.close()


def _build_city_dag(city: City) -> DAG:
    dag_id = _dag_id_for_city(city)

    with DAG(
        dag_id=dag_id,
        description=_description_for_city(city),
        default_args=DEFAULT_ARGS,
        start_date=datetime(2026, 4, 25, tzinfo=timezone.utc),
        schedule="0 * * * *",
        catchup=False,
        max_active_runs=1,
        tags=["aq", "pi", "city", city.country_code.lower(), city.slug],
    ) as dag:
        PythonOperator(
            task_id=f"ingest_{city.slug}",
            python_callable=_run_city,
            op_kwargs={"city_slug": city.slug, "pipeline_name": dag_id},
        )

    return dag


for _city in ALL_MAJOR_CITIES:
    globals()[_dag_id_for_city(_city)] = _build_city_dag(_city)


with DAG(
    dag_id="aq_refresh_warehouse_views_hourly",
    description="Refresh materialized warehouse views after city ingestion DAGs run.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 4, 25, tzinfo=timezone.utc),
    schedule="10 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["aq", "pi", "warehouse", "refresh"],
) as _warehouse_refresh_dag:
    PythonOperator(
        task_id="refresh_materialized_views",
        python_callable=_refresh_warehouse_views,
    )

globals()["aq_refresh_warehouse_views_hourly"] = _warehouse_refresh_dag
