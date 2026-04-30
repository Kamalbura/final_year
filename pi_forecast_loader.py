#!/usr/bin/env python3
"""
Forecast Loader for Raspberry Pi Deployment
Loads forecasts from CSV into PostgreSQL aq.forecasts table
"""
import sys
import pandas as pd
import psycopg2
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict


PROJECT_ROOT = Path("/home/bura/projects/final_year")
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"


def load_forecasts_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load forecasts from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Convert timestamp columns to datetime
    df['forecast_timestamp'] = pd.to_datetime(df['forecast_timestamp'])
    df['horizon_timestamp'] = pd.to_datetime(df['horizon_timestamp'])
    
    return df


def insert_forecasts_to_db(forecasts_df: pd.DataFrame) -> int:
    """
    Insert forecasts into PostgreSQL aq.forecasts table
    
    Returns:
        Number of rows inserted
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="airflow",
        user="airflow",
        password="airflow"
    )
    
    cursor = conn.cursor()
    
    insert_query = """
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
        confidence,
        created_at,
        updated_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (forecast_id) DO UPDATE SET
        predicted_us_aqi = EXCLUDED.predicted_us_aqi,
        predicted_pm2_5 = EXCLUDED.predicted_pm2_5,
        predicted_pm10 = EXCLUDED.predicted_pm10,
        model_type = EXCLUDED.model_type,
        model_version = EXCLUDED.model_version,
        confidence = EXCLUDED.confidence,
        updated_at = NOW()
    """
    
    inserted_count = 0
    
    for _, row in forecasts_df.iterrows():
        try:
            cursor.execute(insert_query, (
                row['forecast_id'],
                row['city_slug'],
                row['city_name'],
                row['forecast_timestamp'],
                row['horizon_timestamp'],
                row['horizon_hours'],
                row['predicted_us_aqi'],
                row.get('predicted_pm2_5'),
                row.get('predicted_pm10'),
                row['model_type'],
                row.get('model_version', '1.0'),
                row.get('confidence'),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
            inserted_count += 1
        except Exception as e:
            print(f"Error inserting forecast {row['forecast_id']}: {e}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return inserted_count


def verify_forecasts_in_db(city_slug: str = None) -> int:
    """
    Verify forecasts in database
    
    Args:
        city_slug: Optional city filter
    
    Returns:
        Number of forecasts in database
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="airflow",
        user="airflow",
        password="airflow"
    )
    
    cursor = conn.cursor()
    
    if city_slug:
        query = """
        SELECT COUNT(*) FROM aq.forecasts 
        WHERE city_slug = %s 
        AND forecast_timestamp >= NOW() - INTERVAL '1 hour'
        """
        cursor.execute(query, (city_slug,))
    else:
        query = """
        SELECT COUNT(*) FROM aq.forecasts 
        WHERE forecast_timestamp >= NOW() - INTERVAL '1 hour'
        """
        cursor.execute(query)
    
    count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    return count


def main():
    """Main forecast loading pipeline"""
    print(f"Starting forecast loading at {datetime.now(timezone.utc)}")
    
    # Load latest forecasts CSV
    latest_csv = FORECAST_DIR / "forecasts_latest.csv"
    
    if not latest_csv.exists():
        print(f"✗ Forecast file not found: {latest_csv}")
        print("Run pi_forecast_generator.py first to generate forecasts")
        return 1
    
    print(f"Loading forecasts from {latest_csv}")
    forecasts_df = load_forecasts_from_csv(latest_csv)
    print(f"Loaded {len(forecasts_df)} forecasts from CSV")
    
    # Insert into database
    print("Inserting forecasts into PostgreSQL...")
    inserted_count = insert_forecasts_to_db(forecasts_df)
    print(f"✓ Inserted {inserted_count} forecasts into aq.forecasts table")
    
    # Verify
    print("\nVerifying forecasts in database:")
    for city in ['delhi', 'hyderabad', 'bengaluru']:
        count = verify_forecasts_in_db(city)
        print(f"  {city}: {count} recent forecasts")
    
    total_count = verify_forecasts_in_db()
    print(f"  Total: {total_count} recent forecasts")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
