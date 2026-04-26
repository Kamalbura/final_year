from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.india_aq import IngestionSettings, bootstrap_csv_to_postgres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap India AQ PostgreSQL schema from archived CSV data.")
    parser.add_argument(
        "--csv-path",
        default="data/india_aq_1y/india_major_cities_aq_1y_combined.csv",
        help="Combined historical CSV to load into PostgreSQL.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN. Defaults to AQ_DATABASE_DSN or the Airflow Postgres service.",
    )
    parser.add_argument(
        "--archive-root",
        default="data/india_aq_1y/archive",
        help="Directory for bootstrap archive snapshots.",
    )
    parser.add_argument(
        "--schema-name",
        default="aq",
        help="Database schema name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = IngestionSettings.from_env()
    if args.dsn:
        settings = IngestionSettings(
            dsn=args.dsn,
            archive_root=Path(args.archive_root),
            initial_lookback_hours=settings.initial_lookback_hours,
            overlap_hours=settings.overlap_hours,
            timeout_seconds=settings.timeout_seconds,
            retries=settings.retries,
            source=settings.source,
            schema_name=args.schema_name,
        )
    else:
        settings = IngestionSettings(
            dsn=settings.dsn,
            archive_root=Path(args.archive_root),
            initial_lookback_hours=settings.initial_lookback_hours,
            overlap_hours=settings.overlap_hours,
            timeout_seconds=settings.timeout_seconds,
            retries=settings.retries,
            source=settings.source,
            schema_name=args.schema_name,
        )

    result = bootstrap_csv_to_postgres(Path(args.csv_path), settings)
    print(result)


if __name__ == "__main__":
    main()
