from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.india_aq import IngestionSettings, connect, ensure_schema


FORECAST_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aq.forecasts (
    forecast_id TEXT PRIMARY KEY,
    city_slug TEXT NOT NULL REFERENCES aq.cities(city_slug) ON DELETE CASCADE,
    city_name TEXT NOT NULL,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    horizon_timestamp TIMESTAMPTZ NOT NULL,
    horizon_hours INT NOT NULL,
    predicted_us_aqi DOUBLE PRECISION NOT NULL,
    predicted_pm2_5 DOUBLE PRECISION,
    predicted_pm10 DOUBLE PRECISION,
    model_type TEXT NOT NULL,
    model_version TEXT,
    confidence DOUBLE PRECISION,
    actual_us_aqi DOUBLE PRECISION,
    actual_pm2_5 DOUBLE PRECISION,
    actual_pm10 DOUBLE PRECISION,
    prediction_error DOUBLE PRECISION,
    absolute_error DOUBLE PRECISION,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecasts_city_horizon
    ON aq.forecasts (city_slug, horizon_timestamp DESC);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Kaggle benchmark forecast rows into aq.forecasts.")
    parser.add_argument("--csv", default="outputs/kaggle_benchmarks/forecast_rows.csv", help="Kaggle forecast_rows.csv path")
    parser.add_argument("--dsn", default=None, help="Optional PostgreSQL DSN override")
    return parser.parse_args()


def ensure_forecast_table(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(FORECAST_TABLE_SQL)
    connection.commit()


def load_forecasts(csv_path: Path, dsn: str | None = None) -> int:
    from psycopg2.extras import execute_values  # type: ignore[import-untyped]

    settings = IngestionSettings.from_env()
    if dsn:
        settings = IngestionSettings(
            dsn=dsn,
            archive_root=settings.archive_root,
            initial_lookback_hours=settings.initial_lookback_hours,
            overlap_hours=settings.overlap_hours,
            timeout_seconds=settings.timeout_seconds,
            retries=settings.retries,
            source=settings.source,
            schema_name=settings.schema_name,
        )

    frame = pd.read_csv(csv_path)
    required = [
        "forecast_id",
        "city_slug",
        "city_name",
        "forecast_timestamp",
        "horizon_timestamp",
        "horizon_hours",
        "predicted_us_aqi",
        "model_type",
        "confidence",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast CSV missing required columns: {missing}")

    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"], utc=True, errors="raise")
    frame["horizon_timestamp"] = pd.to_datetime(frame["horizon_timestamp"], utc=True, errors="raise")

    rows = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        rows.append(
            (
                payload["forecast_id"],
                payload["city_slug"],
                payload["city_name"],
                payload["forecast_timestamp"].to_pydatetime(),
                payload["horizon_timestamp"].to_pydatetime(),
                int(payload["horizon_hours"]),
                float(payload["predicted_us_aqi"]),
                payload.get("predicted_pm2_5") if pd.notna(payload.get("predicted_pm2_5")) else None,
                payload.get("predicted_pm10") if pd.notna(payload.get("predicted_pm10")) else None,
                payload["model_type"],
                payload.get("model_version") if pd.notna(payload.get("model_version")) else None,
                float(payload["confidence"]) if pd.notna(payload["confidence"]) else None,
            )
        )

    connection = connect(settings.dsn)
    try:
        ensure_schema(connection, settings.schema_name)
        ensure_forecast_table(connection)
        with connection.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO aq.forecasts (
                    forecast_id,
                    city_slug,
                    city_name,
                    forecast_timestamp,
                    horizon_timestamp,
                    horizon_hours,
                    predicted_us_aqi,
                    predicted_pm2_5,
                    predicted_pm10,
                    model_type,
                    model_version,
                    confidence
                ) VALUES %s
                ON CONFLICT (forecast_id) DO UPDATE SET
                    predicted_us_aqi = EXCLUDED.predicted_us_aqi,
                    predicted_pm2_5 = EXCLUDED.predicted_pm2_5,
                    predicted_pm10 = EXCLUDED.predicted_pm10,
                    model_type = EXCLUDED.model_type,
                    model_version = EXCLUDED.model_version,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
                """,
                rows,
            )
        connection.commit()
        return len(rows)
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    loaded = load_forecasts(Path(args.csv), args.dsn)
    print({"loaded_forecast_rows": loaded, "source": args.csv})


if __name__ == "__main__":
    main()
