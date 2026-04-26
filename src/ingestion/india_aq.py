from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd


class PgConnection(Protocol):
    def cursor(self) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...

from src.data.cities import City, INDIA_MAJOR_CITIES, city_by_slug

OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_FIELDS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]
DEFAULT_SOURCE = "open-meteo"
SCHEMA_NAME = "aq"
DEFAULT_INITIAL_LOOKBACK_HOURS = 24
DEFAULT_OVERLAP_HOURS = 6
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRIES = 3
DEFAULT_INITIAL_WINDOW_DAYS = 365


@dataclass(frozen=True)
class IngestionSettings:
    dsn: str
    archive_root: Path
    initial_lookback_hours: int = DEFAULT_INITIAL_LOOKBACK_HOURS
    overlap_hours: int = DEFAULT_OVERLAP_HOURS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_RETRIES
    source: str = DEFAULT_SOURCE
    schema_name: str = SCHEMA_NAME

    @classmethod
    def from_env(cls) -> "IngestionSettings":
        import os

        dsn = os.getenv("AQ_DATABASE_DSN", "postgresql://airflow:airflow@postgres:5432/airflow")
        archive_root = Path(os.getenv("AQ_ARCHIVE_ROOT", "/opt/final_year/data/india_aq_1y/archive"))
        initial_lookback_hours = int(os.getenv("AQ_INITIAL_LOOKBACK_HOURS", str(DEFAULT_INITIAL_LOOKBACK_HOURS)))
        overlap_hours = int(os.getenv("AQ_OVERLAP_HOURS", str(DEFAULT_OVERLAP_HOURS)))
        timeout_seconds = int(os.getenv("AQ_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        retries = int(os.getenv("AQ_RETRIES", str(DEFAULT_RETRIES)))
        source = os.getenv("AQ_SOURCE", DEFAULT_SOURCE)
        schema_name = os.getenv("AQ_SCHEMA_NAME", SCHEMA_NAME)
        return cls(
            dsn=dsn,
            archive_root=archive_root,
            initial_lookback_hours=initial_lookback_hours,
            overlap_hours=overlap_hours,
            timeout_seconds=timeout_seconds,
            retries=retries,
            source=source,
            schema_name=schema_name,
        )


SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.cities (
    city_slug TEXT PRIMARY KEY,
    city_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.city_watermarks (
    city_slug TEXT PRIMARY KEY REFERENCES {SCHEMA_NAME}.cities(city_slug) ON DELETE CASCADE,
    last_observed_at TIMESTAMPTZ,
    overlap_hours INTEGER NOT NULL DEFAULT {DEFAULT_OVERLAP_HOURS},
    source TEXT NOT NULL DEFAULT '{DEFAULT_SOURCE}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.ingestion_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    cities_total INTEGER NOT NULL DEFAULT 0,
    cities_succeeded INTEGER NOT NULL DEFAULT 0,
    cities_failed INTEGER NOT NULL DEFAULT 0,
    rows_fetched INTEGER NOT NULL DEFAULT 0,
    rows_upserted INTEGER NOT NULL DEFAULT 0,
    archive_root TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.observations (
    city_slug TEXT NOT NULL REFERENCES {SCHEMA_NAME}.cities(city_slug) ON DELETE CASCADE,
    city_name TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    pm2_5 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    carbon_monoxide DOUBLE PRECISION,
    nitrogen_dioxide DOUBLE PRECISION,
    sulphur_dioxide DOUBLE PRECISION,
    ozone DOUBLE PRECISION,
    us_aqi DOUBLE PRECISION,
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    run_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.ingestion_runs(run_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (city_slug, observed_at, source)
);

CREATE INDEX IF NOT EXISTS idx_observations_city_time_desc
    ON {SCHEMA_NAME}.observations (city_slug, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_time_desc
    ON {SCHEMA_NAME}.observations (observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_run_id
    ON {SCHEMA_NAME}.observations (run_id);
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def resolve_dates(start_date_raw: str | None, end_date_raw: str | None) -> tuple[str, str]:
    end_dt = date.fromisoformat(end_date_raw) if end_date_raw else date.today()
    start_dt = date.fromisoformat(start_date_raw) if start_date_raw else end_dt - timedelta(days=365)
    if start_dt > end_dt:
        raise ValueError("start-date must be earlier than or equal to end-date")
    return start_dt.isoformat(), end_dt.isoformat()


def build_params(city: City, start_date: str, end_date: str) -> dict[str, str | float]:
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": ",".join(HOURLY_FIELDS),
    }


def _coerce_numeric(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _coerce_timestamp(value: Any) -> datetime:
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.to_pydatetime()
    if isinstance(timestamp, datetime):
        return timestamp.astimezone(timezone.utc)
    raise ValueError(f"Unable to coerce timestamp value: {value!r}")


def fetch_city(city: City, start_date: str, end_date: str, timeout: int, retries: int = DEFAULT_RETRIES) -> pd.DataFrame:
    import requests
    from requests import RequestException

    params = build_params(city, start_date, end_date)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            hourly = payload.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                raise ValueError(f"No hourly data returned for {city.name}")

            data = {"timestamp": times}
            for field in HOURLY_FIELDS:
                values = hourly.get(field, [None] * len(times))
                if len(values) != len(times):
                    raise ValueError(
                        f"Length mismatch for {city.name}: field={field}, time={len(times)}, values={len(values)}"
                    )
                data[field] = values

            df = pd.DataFrame(data)
            df.insert(0, "city", city.name)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            nat_count = int(df["timestamp"].isna().sum())
            if nat_count > 0:
                raise ValueError(f"Invalid timestamps for {city.name}: NaT rows={nat_count}")
            return df
        except RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            continue
        except ValueError:
            raise
    raise RuntimeError(f"Failed to download {city.name} after {retries} attempts: {last_error}")


def save_city_csv(df: pd.DataFrame, output_dir: Path, city_name: str) -> Path:
    file_name = city_name.lower().replace(" ", "_") + "_aq_1y.csv"
    output_path = output_dir / file_name
    df.to_csv(output_path, index=False)
    return output_path


def download_all_cities(
    cities: Iterable[City], start_date: str, end_date: str, output_dir: Path, timeout: int
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for city in cities:
        print(f"Downloading {city.name} ({start_date} to {end_date})...")
        try:
            city_df = fetch_city(city, start_date, end_date, timeout=timeout)
            csv_path = save_city_csv(city_df, output_dir, city.name)
            frames.append(city_df)
            print(f"Saved: {csv_path} | rows={len(city_df)}")
        except (RuntimeError, ValueError) as exc:
            failures.append(f"{city.name}: {exc}")
            print(f"Failed: {city.name} | {exc}")

    if not frames:
        failure_report = "\n".join(failures) if failures else "unknown failure"
        raise RuntimeError(f"No city data was downloaded successfully.\n{failure_report}")

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "india_major_cities_aq_1y_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined dataset: {combined_path} | rows={len(combined)}")

    if failures:
        failed_path = output_dir / "failed_cities.log"
        failed_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Some cities failed. See: {failed_path}")

    return combined


def normalize_observation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["city", "timestamp", *HOURLY_FIELDS]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame missing required columns: {missing}")

    normalized = frame[required_columns].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="raise")
    if int(normalized["timestamp"].isna().sum()) > 0:
        raise ValueError("Frame contains invalid timestamps")
    return normalized


def incremental_window(
    last_observed_at: datetime | None,
    *,
    run_end: datetime,
    overlap_hours: int,
    initial_lookback_hours: int = DEFAULT_INITIAL_LOOKBACK_HOURS,
) -> tuple[datetime, datetime]:
    if last_observed_at is None:
        start_at = run_end - timedelta(hours=initial_lookback_hours)
    else:
        start_at = last_observed_at - timedelta(hours=overlap_hours)

    if start_at >= run_end:
        start_at = run_end - timedelta(hours=1)
    return start_at.astimezone(timezone.utc), run_end.astimezone(timezone.utc)


def build_archive_path(archive_root: Path, city: City, run_id: str, run_end: datetime) -> Path:
    run_end = run_end.astimezone(timezone.utc)
    return archive_root / city.slug / run_end.strftime("%Y/%m/%d") / f"{run_id}.csv.gz"


def archive_city_frame(
    frame: pd.DataFrame,
    *,
    archive_root: Path,
    city: City,
    run_id: str,
    run_end: datetime,
) -> Path:
    archive_path = build_archive_path(archive_root, city, run_id, run_end)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_frame = frame.copy()
    archive_frame["timestamp"] = pd.to_datetime(archive_frame["timestamp"], utc=True, errors="raise")
    archive_frame.to_csv(archive_path, index=False, compression="gzip")
    return archive_path


def connect(dsn: str) -> PgConnection:
    import psycopg2  # type: ignore[import-untyped]

    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def ensure_schema(connection: PgConnection, schema_name: str = SCHEMA_NAME) -> None:
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_SQL.replace(SCHEMA_NAME, schema_name))
    connection.commit()


def seed_cities(connection: PgConnection, cities: Iterable[City], schema_name: str = SCHEMA_NAME) -> None:
    from psycopg2.extras import execute_values  # type: ignore[import-untyped]

    rows = [(city.slug, city.name, city.latitude, city.longitude) for city in cities]
    with connection.cursor() as cursor:
        execute_values(
            cursor,
            f"""
            INSERT INTO {schema_name}.cities (city_slug, city_name, latitude, longitude)
            VALUES %s
            ON CONFLICT (city_slug) DO UPDATE SET
                city_name = EXCLUDED.city_name,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                active = TRUE,
                updated_at = NOW()
            """,
            rows,
        )
    connection.commit()


def load_watermarks(connection: PgConnection, schema_name: str = SCHEMA_NAME) -> dict[str, datetime | None]:
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT city_slug, last_observed_at FROM {schema_name}.city_watermarks")
        return {slug: last_observed_at for slug, last_observed_at in cursor.fetchall()}


def upsert_observations(
    connection: PgConnection,
    *,
    city: City,
    frame: pd.DataFrame,
    run_id: str,
    source: str,
    ingested_at: datetime,
    schema_name: str = SCHEMA_NAME,
) -> int:
    from psycopg2.extras import execute_values  # type: ignore[import-untyped]

    normalized = normalize_observation_frame(frame)
    if normalized.empty:
        return 0

    rows: list[tuple[Any, ...]] = []
    for row in normalized.itertuples(index=False, name=None):
        _, timestamp, *metrics = row
        rows.append(
            (
                city.slug,
                city.name,
                _coerce_timestamp(timestamp),
                *(_coerce_numeric(metric) for metric in metrics),
                source,
                ingested_at.astimezone(timezone.utc),
                run_id,
            )
        )

    with connection.cursor() as cursor:
        execute_values(
            cursor,
            f"""
            INSERT INTO {schema_name}.observations (
                city_slug,
                city_name,
                observed_at,
                pm2_5,
                pm10,
                carbon_monoxide,
                nitrogen_dioxide,
                sulphur_dioxide,
                ozone,
                us_aqi,
                source,
                ingested_at,
                run_id
            ) VALUES %s
            ON CONFLICT (city_slug, observed_at, source) DO UPDATE SET
                city_name = EXCLUDED.city_name,
                pm2_5 = EXCLUDED.pm2_5,
                pm10 = EXCLUDED.pm10,
                carbon_monoxide = EXCLUDED.carbon_monoxide,
                nitrogen_dioxide = EXCLUDED.nitrogen_dioxide,
                sulphur_dioxide = EXCLUDED.sulphur_dioxide,
                ozone = EXCLUDED.ozone,
                us_aqi = EXCLUDED.us_aqi,
                ingested_at = EXCLUDED.ingested_at,
                run_id = EXCLUDED.run_id,
                updated_at = NOW()
            """,
            rows,
            page_size=1000,
        )
    connection.commit()
    return len(rows)


def update_watermark(
    connection: PgConnection,
    *,
    city: City,
    observed_at: datetime,
    overlap_hours: int,
    source: str,
    schema_name: str = SCHEMA_NAME,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {schema_name}.city_watermarks (city_slug, last_observed_at, overlap_hours, source, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (city_slug) DO UPDATE SET
                last_observed_at = EXCLUDED.last_observed_at,
                overlap_hours = EXCLUDED.overlap_hours,
                source = EXCLUDED.source,
                updated_at = NOW()
            """,
            (city.slug, observed_at.astimezone(timezone.utc), overlap_hours, source),
        )
    connection.commit()


def record_ingestion_run(
    connection: PgConnection,
    *,
    run_id: str,
    pipeline_name: str,
    started_at: datetime,
    finished_at: datetime | None,
    status: str,
    cities_total: int,
    cities_succeeded: int,
    cities_failed: int,
    rows_fetched: int,
    rows_upserted: int,
    archive_root: Path,
    details: dict[str, Any] | None = None,
    schema_name: str = SCHEMA_NAME,
) -> None:
    payload = json.dumps(details or {}, default=str)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {schema_name}.ingestion_runs (
                run_id,
                pipeline_name,
                started_at,
                finished_at,
                status,
                cities_total,
                cities_succeeded,
                cities_failed,
                rows_fetched,
                rows_upserted,
                archive_root,
                details,
                updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                pipeline_name = EXCLUDED.pipeline_name,
                started_at = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at,
                status = EXCLUDED.status,
                cities_total = EXCLUDED.cities_total,
                cities_succeeded = EXCLUDED.cities_succeeded,
                cities_failed = EXCLUDED.cities_failed,
                rows_fetched = EXCLUDED.rows_fetched,
                rows_upserted = EXCLUDED.rows_upserted,
                archive_root = EXCLUDED.archive_root,
                details = EXCLUDED.details,
                updated_at = NOW()
            """,
            (
                run_id,
                pipeline_name,
                started_at.astimezone(timezone.utc),
                finished_at.astimezone(timezone.utc) if finished_at else None,
                status,
                cities_total,
                cities_succeeded,
                cities_failed,
                rows_fetched,
                rows_upserted,
                str(archive_root),
                payload,
            ),
        )
    connection.commit()


def run_incremental_cycle(settings: IngestionSettings, cities: Iterable[City] = INDIA_MAJOR_CITIES) -> dict[str, Any]:
    return run_incremental_cycle_for_cities(settings, cities)


def run_incremental_cycle_for_cities(
    settings: IngestionSettings,
    cities: Iterable[City],
    *,
    pipeline_name: str = "aq_postgres_incremental_6h",
) -> dict[str, Any]:
    city_list = tuple(cities)
    run_id = datetime.now(timezone.utc).strftime("aq-%Y%m%dT%H%M%SZ")
    started_at = datetime.now(timezone.utc)
    run_end = started_at

    settings.archive_root.mkdir(parents=True, exist_ok=True)
    connection = connect(settings.dsn)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_rows_fetched = 0
    total_rows_upserted = 0

    try:
        ensure_schema(connection, settings.schema_name)
        seed_cities(connection, city_list, settings.schema_name)
        watermarks = load_watermarks(connection, settings.schema_name)

        for city in city_list:
            last_observed_at = watermarks.get(city.slug)
            window_start, window_end = incremental_window(
                last_observed_at,
                run_end=run_end,
                overlap_hours=settings.overlap_hours,
                initial_lookback_hours=settings.initial_lookback_hours,
            )
            start_date = window_start.date().isoformat()
            end_date = run_end.date().isoformat()

            try:
                city_frame = fetch_city(city, start_date, end_date, timeout=settings.timeout_seconds, retries=settings.retries)
                total_rows_fetched += len(city_frame)
                archive_path = archive_city_frame(
                    city_frame,
                    archive_root=settings.archive_root,
                    city=city,
                    run_id=run_id,
                    run_end=run_end,
                )
                upserted_rows = upsert_observations(
                    connection,
                    city=city,
                    frame=city_frame,
                    run_id=run_id,
                    source=settings.source,
                    ingested_at=started_at,
                    schema_name=settings.schema_name,
                )
                total_rows_upserted += upserted_rows
                city_latest = pd.to_datetime(city_frame["timestamp"], utc=True).max().to_pydatetime()
                update_watermark(
                    connection,
                    city=city,
                    observed_at=city_latest,
                    overlap_hours=settings.overlap_hours,
                    source=settings.source,
                    schema_name=settings.schema_name,
                )
                results.append(
                    {
                        "city": city.name,
                        "status": "success",
                        "rows_fetched": len(city_frame),
                        "rows_upserted": upserted_rows,
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                        "archive_path": str(archive_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                connection.rollback()
                failures.append(
                    {
                        "city": city.name,
                        "status": "failed",
                        "error": str(exc),
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                    }
                )

        status = "success" if not failures else "partial_failure"
        record_ingestion_run(
            connection,
            run_id=run_id,
            pipeline_name=pipeline_name,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            status=status,
            cities_total=len(city_list),
            cities_succeeded=len(results),
            cities_failed=len(failures),
            rows_fetched=total_rows_fetched,
            rows_upserted=total_rows_upserted,
            archive_root=settings.archive_root,
            details={"results": results, "failures": failures},
            schema_name=settings.schema_name,
        )

        if failures:
            raise RuntimeError(f"Incremental ingestion completed with failures: {json.dumps(failures, default=str)}")

        return {
            "run_id": run_id,
            "status": status,
            "cities_succeeded": len(results),
            "cities_failed": len(failures),
            "rows_fetched": total_rows_fetched,
            "rows_upserted": total_rows_upserted,
            "archive_root": str(settings.archive_root),
            "results": results,
        }
    finally:
        connection.close()


def run_incremental_cycle_for_city(
    settings: IngestionSettings,
    city_slug: str,
    *,
    pipeline_name: str | None = None,
) -> dict[str, Any]:
    city = city_by_slug()[city_slug]
    return run_incremental_cycle_for_cities(
        settings,
        (city,),
        pipeline_name=pipeline_name or f"aq_{city.slug}_incremental_6h",
    )


def bootstrap_csv_to_postgres(
    csv_path: Path,
    settings: IngestionSettings,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("aq-bootstrap-%Y%m%dT%H%M%SZ")
    started_at = datetime.now(timezone.utc)
    source_name = source or settings.source
    connection = connect(settings.dsn)
    results: list[dict[str, Any]] = []

    try:
        ensure_schema(connection, settings.schema_name)
        seed_cities(connection, INDIA_MAJOR_CITIES, settings.schema_name)

        frame = pd.read_csv(csv_path)
        frame = normalize_observation_frame(frame)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame["city"] = frame["city"].astype(str)

        for city in INDIA_MAJOR_CITIES:
            city_frame = frame[frame["city"] == city.name].copy()
            if city_frame.empty:
                continue
            archive_path = archive_city_frame(
                city_frame,
                archive_root=settings.archive_root,
                city=city,
                run_id=run_id,
                run_end=started_at,
            )
            upserted_rows = upsert_observations(
                connection,
                city=city,
                frame=city_frame,
                run_id=run_id,
                source=source_name,
                ingested_at=started_at,
                schema_name=settings.schema_name,
            )
            latest = pd.to_datetime(city_frame["timestamp"], utc=True).max().to_pydatetime()
            update_watermark(
                connection,
                city=city,
                observed_at=latest,
                overlap_hours=settings.overlap_hours,
                source=source_name,
                schema_name=settings.schema_name,
            )
            results.append(
                {
                    "city": city.name,
                    "rows": len(city_frame),
                    "rows_upserted": upserted_rows,
                    "archive_path": str(archive_path),
                }
            )

        record_ingestion_run(
            connection,
            run_id=run_id,
            pipeline_name="aq_bootstrap_csv_to_postgres",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            status="success",
            cities_total=len(INDIA_MAJOR_CITIES),
            cities_succeeded=len(results),
            cities_failed=0,
            rows_fetched=len(frame),
            rows_upserted=sum(result["rows_upserted"] for result in results),
            archive_root=settings.archive_root,
            details={"results": results, "source_csv": str(csv_path)},
            schema_name=settings.schema_name,
        )

        return {
            "run_id": run_id,
            "status": "success",
            "cities_succeeded": len(results),
            "rows_fetched": len(frame),
            "rows_upserted": sum(result["rows_upserted"] for result in results),
            "results": results,
        }
    finally:
        connection.close()
