#!/usr/bin/env python3
"""
Forecast Generator for Raspberry Pi Deployment
Generates 24-hour ahead AQI forecasts for Delhi, Hyderabad, and Bengaluru
"""
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = Path("/home/bura/projects/final_year")
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment_models"
VENV_PYTHON = "/home/bura/venvs/ml/bin/python3"

# City configurations
CITIES = {
    "delhi": {"model": "lightgbm", "display_name": "Delhi"},
    "hyderabad": {"model": "xgboost", "display_name": "Hyderabad"},
    "bengaluru": {"model": "catboost", "display_name": "Bengaluru"},
}


def load_model(city_slug: str) -> Tuple[object, Dict]:
    """Load the trained model and metadata for a city"""
    city_dir = DEPLOYMENT_DIR / city_slug
    model_config = CITIES[city_slug]
    
    # Load model
    model_path = city_dir / f"{model_config['model']}.joblib"
    model_dict = joblib.load(model_path)
    
    # Extract the actual model from the dictionary
    if isinstance(model_dict, dict) and 'model' in model_dict:
        model = model_dict['model']
        x_scaler = model_dict.get('x_scaler')
        y_scaler = model_dict.get('y_scaler')
    else:
        model = model_dict
        x_scaler = None
        y_scaler = None
    
    # Load metadata
    metadata_path = city_dir / "model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata, x_scaler, y_scaler


def fetch_recent_observations(city_slug: str, lookback_hours: int = 168) -> pd.DataFrame:
    """
    Fetch recent observations from PostgreSQL for feature engineering
    
    Args:
        city_slug: City identifier (delhi, hyderabad, bengaluru)
        lookback_hours: Number of hours to look back (default 168 = 7 days)
    
    Returns:
        DataFrame with recent observations
    """
    import psycopg2
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="airflow",
        user="airflow",
        password="airflow"
    )
    
    query = f"""
    SELECT 
        observed_at,
        us_aqi,
        pm2_5
    FROM aq.observations
    WHERE city_slug = %s
    AND observed_at >= NOW() - INTERVAL '{lookback_hours} hours'
    ORDER BY observed_at ASC
    """
    
    df = pd.read_sql_query(query, conn, params=(city_slug,))
    conn.close()
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw observations
    This should match the feature engineering used during training
    """
    # Convert observed_at to datetime
    df['observed_at'] = pd.to_datetime(df['observed_at'])
    
    # Time-based features
    df['hour'] = df['observed_at'].dt.hour
    df['day_of_week'] = df['observed_at'].dt.dayofweek
    df['month'] = df['observed_at'].dt.month
    
    # Lag features (1h, 3h, 6h, 12h, 24h)
    for col in ['us_aqi', 'pm2_5']:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_3h'] = df[col].shift(3)
            df[f'{col}_lag_6h'] = df[col].shift(6)
            df[f'{col}_lag_12h'] = df[col].shift(12)
            df[f'{col}_lag_24h'] = df[col].shift(24)
    
    # Rolling statistics (3h, 6h, 12h, 24h windows)
    for col in ['us_aqi', 'pm2_5']:
        if col in df.columns:
            df[f'{col}_rolling_mean_3h'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_rolling_std_3h'] = df[col].rolling(window=3, min_periods=1).std()
            df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
            df[f'{col}_rolling_mean_12h'] = df[col].rolling(window=12, min_periods=1).mean()
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
    
    # Drop rows with NaN values from lag/rolling features
    df = df.dropna()
    
    return df


def generate_forecast(city_slug: str, horizon_hours: int = 24) -> List[Dict]:
    """
    Generate forecast for a city
    
    Args:
        city_slug: City identifier
        horizon_hours: Forecast horizon in hours (default 24)
    
    Returns:
        List of forecast dictionaries
    """
    # Load model and metadata
    model, metadata, x_scaler, y_scaler = load_model(city_slug)
    
    # Fetch recent observations
    df = fetch_recent_observations(city_slug, lookback_hours=168)
    
    if len(df) < 24:
        raise ValueError(f"Insufficient data for {city_slug}: only {len(df)} observations")
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Get the most recent observation for forecasting
    latest_features = df_features.iloc[-1:].copy()
    
    # Prepare features for prediction (exclude timestamp and target columns)
    feature_cols = [col for col in latest_features.columns 
                   if col not in ['observed_at', 'us_aqi', 'pm2_5']]
    X = latest_features[feature_cols]
    
    # Scale features if scaler is available
    if x_scaler is not None:
        X = x_scaler.transform(X)
    
    # Generate predictions for each hour in the horizon
    forecasts = []
    forecast_timestamp = datetime.now(timezone.utc)
    
    for h in range(1, horizon_hours + 1):
        # Predict
        predicted = model.predict(X)[0]
        
        # Inverse scale if scaler is available
        if y_scaler is not None:
            predicted = y_scaler.inverse_transform(predicted.reshape(1, -1))[0]
        
        # Extract AQI prediction (first output if multi-output)
        predicted_aqi = predicted[0] if isinstance(predicted, (list, np.ndarray)) else predicted
        
        # Create forecast record
        horizon_timestamp = forecast_timestamp + pd.Timedelta(hours=h)
        forecast_id = f"{city_slug}_{forecast_timestamp.strftime('%Y%m%d%H%M%S')}_h{h:02d}"
        
        forecast = {
            "forecast_id": forecast_id,
            "city_slug": city_slug,
            "city_name": CITIES[city_slug]["display_name"],
            "forecast_timestamp": forecast_timestamp.isoformat(),
            "horizon_timestamp": horizon_timestamp.isoformat(),
            "horizon_hours": h,
            "predicted_us_aqi": float(predicted_aqi),
            "predicted_pm2_5": None,  # Can be added if model predicts PM2.5
            "predicted_pm10": None,   # Can be added if model predicts PM10
            "model_type": metadata["model_type"],
            "model_version": metadata.get("model_version", "1.0"),
            "confidence": None,  # Can be added if model provides confidence
        }
        
        forecasts.append(forecast)
    
    return forecasts


def save_forecasts_to_csv(forecasts: List[Dict], output_path: Path):
    """Save forecasts to CSV file"""
    df = pd.DataFrame(forecasts)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(forecasts)} forecasts to {output_path}")


def main():
    """Main forecast generation pipeline"""
    print(f"Starting forecast generation at {datetime.now(timezone.utc)}")
    
    all_forecasts = []
    
    for city_slug in CITIES.keys():
        try:
            print(f"\nGenerating forecasts for {city_slug}...")
            forecasts = generate_forecast(city_slug, horizon_hours=24)
            all_forecasts.extend(forecasts)
            print(f"✓ Generated {len(forecasts)} forecasts for {city_slug}")
        except Exception as e:
            print(f"✗ Error generating forecasts for {city_slug}: {e}")
            continue
    
    if all_forecasts:
        # Save to CSV
        output_dir = PROJECT_ROOT / "data" / "forecasts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"forecasts_{timestamp}.csv"
        
        save_forecasts_to_csv(all_forecasts, output_path)
        
        # Also save as latest.csv for easy access
        latest_path = output_dir / "forecasts_latest.csv"
        save_forecasts_to_csv(all_forecasts, latest_path)
        
        print(f"\n✓ Total forecasts generated: {len(all_forecasts)}")
        print(f"✓ Output: {output_path}")
        return 0
    else:
        print("\n✗ No forecasts generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
