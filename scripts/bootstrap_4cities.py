#!/usr/bin/env python3
"""Bootstrap historical CSV data into PostgreSQL for 4 target cities."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.data.cities import city_by_slug, INDIA_MAJOR_CITIES
from src.ingestion.india_aq import (
    connect, ensure_schema, seed_cities,
    upsert_observations, update_watermark, record_ingestion_run,
    IngestionSettings, normalize_observation_frame,
)
from datetime import datetime, timezone

TARGET_CITIES = ["hyderabad", "delhi", "mumbai", "bengaluru"]
DATA_DIR = REPO_ROOT / "data" / "india_aq_1y"

def main():
    dsn = sys.argv[1] if len(sys.argv) > 1 else "postgresql://airflow:airflow@localhost:5432/airflow"
    conn = connect(dsn)
    
    try:
        ensure_schema(conn)
        seed_cities(conn, INDIA_MAJOR_CITIES)
        print("Schema + cities seeded")

        city_map = city_by_slug()
        run_id = datetime.now(timezone.utc).strftime("aq-bootstrap-%Y%m%dT%H%M%SZ")
        started_at = datetime.now(timezone.utc)
        total_upserted = 0
        results = []

        # Record ingestion run FIRST (FK constraint requires it before observations)
        record_ingestion_run(
            conn, run_id=run_id, pipeline_name="bootstrap_csv_4cities",
            started_at=started_at, finished_at=None,
            status="running", cities_total=len(TARGET_CITIES),
            cities_succeeded=0, cities_failed=0,
            rows_fetched=0, rows_upserted=0,
            archive_root=Path("/tmp/bootstrap"),
            details={},
        )
        print(f"Ingestion run {run_id} created")

        for slug in TARGET_CITIES:
            city = city_map[slug]
            csv_path = DATA_DIR / f"{slug}_aq_1y.csv"
            if not csv_path.exists():
                print(f"SKIP {slug}: {csv_path} not found")
                continue

            df = pd.read_csv(csv_path)
            print(f"Loading {slug}: {len(df)} rows from {csv_path.name}")

            # Ensure column mapping
            if "city" not in df.columns and "City" in df.columns:
                df.rename(columns={"City": "city"}, inplace=True)
            if "city" not in df.columns:
                df.insert(0, "city", city.name)
            if "timestamp" not in df.columns and "Timestamp" in df.columns:
                df.rename(columns={"Timestamp": "timestamp"}, inplace=True)

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])

            try:
                norm = normalize_observation_frame(df)
            except ValueError as e:
                print(f"  Normalization error: {e}")
                # Try renaming columns
                col_map = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl in ("pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                              "sulphur_dioxide", "ozone", "us_aqi", "city", "timestamp"):
                        col_map[c] = cl
                df.rename(columns=col_map, inplace=True)
                norm = normalize_observation_frame(df)

            n = upsert_observations(
                conn, city=city, frame=df, run_id=run_id,
                source="open-meteo-csv", ingested_at=started_at,
            )
            total_upserted += n
            
            latest = df["timestamp"].max()
            if pd.notna(latest):
                update_watermark(conn, city=city, observed_at=latest.to_pydatetime(),
                                overlap_hours=6, source="open-meteo-csv")

            results.append({"city": slug, "rows": len(df), "upserted": n})
            print(f"  {slug}: {n} rows upserted")

        record_ingestion_run(
            conn, run_id=run_id, pipeline_name="bootstrap_csv_4cities",
            started_at=started_at, finished_at=datetime.now(timezone.utc),
            status="success", cities_total=len(TARGET_CITIES),
            cities_succeeded=len(results), cities_failed=0,
            rows_fetched=sum(r["rows"] for r in results),
            rows_upserted=total_upserted,
            archive_root=Path("/tmp/bootstrap"),
            details={"results": results},
        )
        print(f"\nDone! Total upserted: {total_upserted}")
        
        # Refresh materialized views
        cursor = conn.cursor()
        for view in ["aq.latest_observations", "aq.hourly_aggregates",
                      "aq.daily_aggregates", "aq.monthly_aggregates"]:
            try:
                cursor.execute(f"REFRESH MATERIALIZED VIEW {view}")
                conn.commit()
                print(f"Refreshed {view}")
            except Exception as e:
                conn.rollback()
                print(f"Error refreshing {view}: {e}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
