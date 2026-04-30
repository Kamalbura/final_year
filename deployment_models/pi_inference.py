"""
Raspberry Pi Inference Script for AQI Forecasting
Loads best models per city and generates 24-hour forecasts
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

# Model paths
MODELS_DIR = Path(__file__).parent
CITIES = ["delhi", "hyderabad", "bengaluru"]
FEATURES = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
LOOKBACK_HOURS = 168
HORIZON_HOURS = 24


def load_model(city: str):
    """Load model and metadata for a city."""
    city_dir = MODELS_DIR / city
    metadata_path = city_dir / "model_metadata.json"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    model_path = city_dir / metadata["model_file"]
    model_data = joblib.load(model_path)
    
    return {
        "model": model_data["model"],
        "x_scaler": model_data["x_scaler"],
        "y_scaler": model_data["y_scaler"],
        "metadata": metadata
    }


def prepare_input_window(observations_df: pd.DataFrame, lookback_hours: int):
    """
    Prepare input window from latest observations.
    
    Args:
        observations_df: DataFrame with columns [timestamp, pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, ozone]
        lookback_hours: Number of hours to look back
    
    Returns:
        Scaled input array of shape (1, lookback_hours * n_features)
    """
    # Sort by timestamp and take last lookback_hours rows
    df = observations_df.sort_values("timestamp").tail(lookback_hours)
    
    if len(df) < lookback_hours:
        raise ValueError(f"Not enough data: need {lookback_hours} hours, got {len(df)}")
    
    # Extract features
    features = df[FEATURES].values.astype(np.float32)
    
    return features


def forecast_city(city: str, observations_df: pd.DataFrame):
    """
    Generate 24-hour forecast for a city.
    
    Args:
        city: City name (delhi, hyderabad, bengaluru)
        observations_df: Recent observations DataFrame
    
    Returns:
        List of forecast dictionaries
    """
    print(f"Loading model for {city}...")
    model_data = load_model(city)
    
    print(f"Preparing input window ({LOOKBACK_HOURS} hours)...")
    features = prepare_input_window(observations_df, LOOKBACK_HOURS)
    
    # Scale features
    features_scaled = model_data["x_scaler"].transform(features)
    
    # Flatten for classical ML models (they expect 2D: [n_samples, n_features])
    features_flat = features_scaled.reshape(1, -1)
    
    print(f"Running inference with {model_data['metadata']['model_name']}...")
    predictions_scaled = model_data["model"].predict(features_flat)
    
    # Inverse transform predictions
    predictions = model_data["y_scaler"].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    # Build forecast rows
    last_timestamp = pd.Timestamp(observations_df["timestamp"].max())
    forecast_timestamp = pd.Timestamp.utcnow()
    
    forecasts = []
    for hour_offset in range(1, HORIZON_HOURS + 1):
        horizon_timestamp = last_timestamp + pd.Timedelta(hours=hour_offset)
        forecasts.append({
            "forecast_id": f"{city}-{model_data['metadata']['model_name'].lower()}-{forecast_timestamp.strftime('%Y%m%dT%H%M%SZ')}-{hour_offset:03d}",
            "city_slug": city,
            "city_name": city.title(),
            "forecast_timestamp": forecast_timestamp.isoformat(),
            "horizon_timestamp": horizon_timestamp.isoformat(),
            "horizon_hours": hour_offset,
            "predicted_us_aqi": float(predictions[hour_offset - 1]),
            "model_type": model_data["metadata"]["model_name"],
            "model_version": "production-v1",
            "confidence": float(1.0 / (1.0 + model_data["metadata"]["metrics"]["rmse"] / 100.0))
        })
    
    return forecasts


def main():
    """Main inference loop for all cities."""
    print("=" * 70)
    print("Raspberry Pi AQI Forecasting - Production Inference")
    print("=" * 70)
    
    # Example: Load observations from PostgreSQL
    # In production, replace this with actual DB query
    # from src.ingestion.india_aq import IngestionSettings, connect
    # settings = IngestionSettings.from_env()
    # conn = connect(settings.dsn)
    # observations = pd.read_sql("SELECT * FROM aq.observations WHERE city_id = ... ORDER BY timestamp DESC LIMIT 168", conn)
    
    # For testing, use dummy data
    print("\n⚠️  Using dummy data for testing. Replace with actual DB query in production.")
    
    for city in CITIES:
        print(f"\n{'='*70}")
        print(f"City: {city.upper()}")
        print(f"{'='*70}")
        
        # Create dummy observations (replace with real data)
        timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=LOOKBACK_HOURS, freq="1h")
        dummy_observations = pd.DataFrame({
            "timestamp": timestamps,
            "pm2_5": np.random.uniform(10, 100, LOOKBACK_HOURS),
            "pm10": np.random.uniform(20, 150, LOOKBACK_HOURS),
            "carbon_monoxide": np.random.uniform(200, 1000, LOOKBACK_HOURS),
            "nitrogen_dioxide": np.random.uniform(5, 50, LOOKBACK_HOURS),
            "ozone": np.random.uniform(10, 100, LOOKBACK_HOURS)
        })
        
        try:
            forecasts = forecast_city(city, dummy_observations)
            print(f"✓ Generated {len(forecasts)} forecast points")
            print(f"  AQI range: [{min(f['predicted_us_aqi'] for f in forecasts):.1f}, {max(f['predicted_us_aqi'] for f in forecasts):.1f}]")
            
            # Save to CSV (in production, insert into aq.forecasts table)
            forecast_df = pd.DataFrame(forecasts)
            output_path = MODELS_DIR / f"{city}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            forecast_df.to_csv(output_path, index=False)
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*70}")
    print("Inference complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
