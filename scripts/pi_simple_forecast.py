#!/usr/bin/env python3
"""
Simple Forecast Generator - Uses persistence and trend-based forecasting
Since trained models require features not available in current database
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import psycopg2

PROJECT_ROOT = Path("/home/bura/projects/final_year")

CITIES = {
    "delhi": "Delhi",
    "hyderabad": "Hyderabad",
    "bengaluru": "Bengaluru",
}


def fetch_recent_data(city_slug: str, hours: int = 48) -> pd.DataFrame:
    """Fetch recent observations"""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="airflow",
        user="airflow",
        password="airflow"
    )
    
    query = f"""
    SELECT observed_at, us_aqi, pm2_5
    FROM aq.observations
    WHERE city_slug = %s
    AND observed_at >= NOW() - INTERVAL '{hours} hours'
    ORDER BY observed_at ASC
    """
    
    df = pd.read_sql_query(query, conn, params=(city_slug,))
    conn.close()
    return df


def generate_simple_forecast(city_slug: str, horizon_hours: int = 24):
    """Generate forecast using persistence + trend"""
    df = fetch_recent_data(city_slug, hours=48)
    
    if len(df) < 24:
        raise ValueError(f"Insufficient data: only {len(df)} observations")
    
    # Get recent trend (last 24 hours)
    recent = df.tail(24)
    current_aqi = recent['us_aqi'].iloc[-1]
    trend = recent['us_aqi'].diff().mean()  # Average hourly change
    
    # Generate forecasts
    forecasts = []
    forecast_timestamp = datetime.now(timezone.utc)
    
    for h in range(1, horizon_hours + 1):
        # Simple persistence + trend model
        predicted_aqi = current_aqi + (trend * h)
        
        # Add some realistic bounds
        predicted_aqi = max(0, min(500, predicted_aqi))
        
        horizon_timestamp = forecast_timestamp + pd.Timedelta(hours=h)
        forecast_id = f"{city_slug}_{forecast_timestamp.strftime('%Y%m%d%H%M%S')}_h{h:02d}"
        
        forecasts.append({
            "forecast_id": forecast_id,
            "city_slug": city_slug,
            "city_name": CITIES[city_slug],
            "forecast_timestamp": forecast_timestamp.isoformat(),
            "horizon_timestamp": horizon_timestamp.isoformat(),
            "horizon_hours": h,
            "predicted_us_aqi": float(predicted_aqi),
            "predicted_pm2_5": None,
            "predicted_pm10": None,
            "model_type": "Persistence+Trend",
            "model_version": "1.0-simple",
            "confidence": 0.7,  # Moderate confidence for simple model
        })
    
    return forecasts


def save_to_csv(forecasts, output_path):
    """Save forecasts to CSV"""
    df = pd.DataFrame(forecasts)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(forecasts)} forecasts to {output_path}")


def main():
    print(f"Starting simple forecast generation at {datetime.now(timezone.utc)}")
    
    all_forecasts = []
    
    for city_slug in CITIES.keys():
        try:
            print(f"\nGenerating forecasts for {city_slug}...")
            forecasts = generate_simple_forecast(city_slug, horizon_hours=24)
            all_forecasts.extend(forecasts)
            print(f"✓ Generated {len(forecasts)} forecasts for {city_slug}")
        except Exception as e:
            print(f"✗ Error for {city_slug}: {e}")
            continue
    
    if all_forecasts:
        output_dir = PROJECT_ROOT / "data" / "forecasts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"forecasts_{timestamp}.csv"
        save_to_csv(all_forecasts, output_path)
        
        latest_path = output_dir / "forecasts_latest.csv"
        save_to_csv(all_forecasts, latest_path)
        
        print(f"\n✓ Total: {len(all_forecasts)} forecasts")
        return 0
    else:
        print("\n✗ No forecasts generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
