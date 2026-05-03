#!/usr/bin/env python3
"""
Proper PM2.5 Forecast Generator for Raspberry Pi
Uses trained PyTorch models (Transformer, GRU, CNN-LSTM) with 168h lookback window
"""
import sys
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path("/home/bura/projects/final_year")
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment_models" / "hyderabad"

# Model configurations - matching our trained architectures
MODEL_CONFIGS = {
    "transformer": {
        "name": "Transformer",
        "path": DEPLOYMENT_DIR / "transformers" / "Transformer" / "model.pth",
        "family": "Transformer",
    },
    "cnn_lstm": {
        "name": "CNN-LSTM",
        "path": DEPLOYMENT_DIR / "dl_hybrid" / "CNN-LSTM" / "model.pth",
        "family": "CNN-RNN Hybrid",
    },
    "gru": {
        "name": "GRU",
        "path": DEPLOYMENT_DIR / "dl_sequence" / "GRU" / "model.pth",
        "family": "RNN Sequence",
    },
}

# Feature order matching training: [pm2_5, pm10, co, no2, so2, o3, aqi]
FEATURE_COLS = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'us_aqi']
LOOKBACK_HOURS = 168  # 7 days
FORECAST_HORIZON = 24  # 24 hours

# Model definitions (matching our training architectures)
class GRUForecaster(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, output_dim=24, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, output_dim=24, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim=7, d_model=64, nhead=2, num_layers=2, output_dim=24, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

def fetch_recent_observations(lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Fetch recent observations from PostgreSQL for feature engineering"""
    import psycopg2
    
    conn = psycopg2.connect(
        host="localhost", port=5432, database="airflow",
        user="airflow", password="airflow"
    )
    
    query = """
    SELECT observed_at, pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, 
           sulphur_dioxide, ozone, us_aqi
    FROM aq.observations
    WHERE city_slug = %s
      AND observed_at >= NOW() - INTERVAL '%s hours'
      AND observed_at <= NOW()
    ORDER BY observed_at ASC
    """
    
    df = pd.read_sql_query(query, conn, params=('hyderabad', str(lookback_hours + 24)))
    conn.close()
    
    if len(df) < LOOKBACK_HOURS:
        raise ValueError(f"Insufficient data: only {len(df)} observations, need {LOOKBACK_HOURS}")
    
    return df

def preprocess_data(df: pd.DataFrame, x_scaler, y_scaler) -> torch.Tensor:
    """Preprocess to match training pipeline: 168h window -> normalized -> tensor"""
    # Ensure we have exactly 168 rows
    df = df.tail(LOOKBACK_HOURS).reset_index(drop=True)
    
    # Extract features in correct order
    X = df[FEATURE_COLS].values.astype(np.float32)
    
    # Handle missing values with interpolation
    X = pd.DataFrame(X).interpolate(method='linear', limit_direction='both').values
    
    # Apply MinMax scaling from training
    X_scaled = x_scaler.transform(X)
    
    # Convert to tensor: (1, 168, 7)
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
    
    return X_tensor

def load_model_and_scalers(model_config: Dict):
    """Load PyTorch model and scalers from deployment directory"""
    model_dir = model_config['path'].parent
    
    # Load scalers
    with open(model_dir / 'x_scaler.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open(model_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
    
    # Load metrics to get architecture params
    metrics_path = model_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        best_params = metrics.get('best_params', {})
    else:
        best_params = {}
    
    # Create model based on family
    family = model_config['family']
    if family == 'Transformer':
        model = TransformerForecaster(
            input_dim=7,
            d_model=best_params.get('model_dim', 64),
            nhead=best_params.get('heads', 2),
            num_layers=best_params.get('layers', 2),
            output_dim=FORECAST_HORIZON,
            dropout=best_params.get('dropout', 0.1)
        )
    elif family == 'CNN-RNN Hybrid':
        model = GRUForecaster(
            input_dim=7, hidden_dim=best_params.get('hidden_dim', 128),
            num_layers=2, output_dim=FORECAST_HORIZON,
            dropout=best_params.get('dropout', 0.2)
        )
    else:  # RNN Sequence / GRU
        model = GRUForecaster(
            input_dim=7, hidden_dim=best_params.get('hidden_dim', 128),
            num_layers=2, output_dim=FORECAST_HORIZON,
            dropout=best_params.get('dropout', 0.2)
        )
    
    # Load weights
    model.load_state_dict(torch.load(model_config['path'], map_location='cpu'))
    model.eval()
    
    return model, x_scaler, y_scaler

def generate_forecast(model_config: Dict) -> Optional[List[Dict]]:
    """Generate 24-hour PM2.5 forecast for Hyderabad using the specified model"""
    try:
        model, x_scaler, y_scaler = load_model_and_scalers(model_config)
        
        # Fetch recent data
        df = fetch_recent_observations()
        
        # Preprocess
        X_tensor = preprocess_data(df, x_scaler, y_scaler)
        
        # Predict
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).squeeze(0).numpy()
        
        # Inverse scale predictions
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Build forecast records
        now = datetime.now(timezone.utc)
        forecasts = []
        
        for h, pm25_val in enumerate(y_pred, 1):
            forecast = {
                "city_slug": "hyderabad",
                "city_name": "Hyderabad",
                "model_name": model_config['name'],
                "model_family": model_config['family'],
                "forecast_timestamp": now.isoformat(),
                "horizon_hours": h,
                "horizon_timestamp": (now + pd.Timedelta(hours=h)).isoformat(),
                "predicted_pm2_5": round(float(pm25_val), 1),
                "predicted_pm10": None,
            }
            forecasts.append(forecast)
        
        return forecasts
    
    except Exception as e:
        print(f"Error generating forecast for {model_config['name']}: {e}", file=sys.stderr)
        return None

def main():
    """Main forecast generation"""
    print(f"=== PM2.5 Forecast Generation ===")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"City: Hyderabad")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print()
    
    all_forecasts = []
    for model_key, config in MODEL_CONFIGS.items():
        print(f"Generating forecast using {config['name']}...")
        forecasts = generate_forecast(config)
        if forecasts:
            all_forecasts.append(forecasts)
            print(f"  OK: {len(forecasts)} hour forecast generated")
            # Print sample
            print(f"  t+1h PM2.5: {forecasts[0]['predicted_pm2_5']:.1f} ug/m3")
            print(f"  t+24h PM2.5: {forecasts[-1]['predicted_pm2_5']:.1f} ug/m3")
        else:
            print(f"  FAILED")
    
    # Save to JSON
    output_dir = PROJECT_ROOT / "data" / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "city": "hyderabad",
        "forecasts": {}
    }
    
    for model_config, forecasts in zip(MODEL_CONFIGS.values(), all_forecasts):
        if forecasts:
            output["forecasts"][model_config["name"]] = {
                "model_family": model_config["family"],
                "values": forecasts
            }
    
    # Save latest
    latest_path = output_dir / "hyderabad_forecast_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {latest_path}")
    print(f"Total models with forecasts: {len([f for f in all_forecasts if f])}")
    
    return 0 if any(all_forecasts) else 1

if __name__ == "__main__":
    sys.exit(main())
