#!/usr/bin/env python3
"""
Unified Prediction Engine for Raspberry Pi
Supports 8 model types across 3 cities
Each model runs independently, triggered by its own Airflow DAG

Models by tier:
  TIER 1 (Best):  XGBoost (joblib), LightGBM (joblib), Random Forest (joblib)
  TIER 2 (DL):    Transformer (ONNX/torch), CNN-LSTM (ONNX), GRU (ONNX)
  TIER 3 (Good):  SVR (joblib), BiLSTM (ONNX)
"""
import sys, os, json, pickle, time, joblib
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

PROJECT_ROOT = Path("/home/bura/projects/final_year")
DEPLOY_DIR = PROJECT_ROOT / "deployment_models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Cities to predict for
CITIES = ["hyderabad", "delhi", "bengaluru"]
CITY_NAMES = {"hyderabad": "Hyderabad", "delhi": "Delhi", "bengaluru": "Bengaluru"}

# Features for tree models (lag-based)
POLLUTANTS = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'us_aqi']
LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]
HORIZON = 24

# Model registry - all 8 models
MODEL_REGISTRY = {
    "xgboost":       {"tier": 1, "family": "Tree Ensemble", "backend": "joblib", "name": "XGBoost"},
    "lightgbm":      {"tier": 1, "family": "Tree Ensemble", "backend": "joblib", "name": "LightGBM"},
    "random_forest": {"tier": 1, "family": "Tree Ensemble", "backend": "joblib", "name": "Random Forest"},
    "svr":           {"tier": 3, "family": "Support Vector", "backend": "joblib", "name": "SVR"},
    "cnn_lstm":      {"tier": 2, "family": "DL Hybrid",     "backend": "onnx",   "name": "CNN-LSTM"},
    "gru":           {"tier": 2, "family": "DL Sequence",   "backend": "onnx",   "name": "GRU"},
    "transformer":   {"tier": 2, "family": "Transformer",   "backend": "torch",  "name": "Transformer"},
    "bilstm":        {"tier": 3, "family": "DL Sequence",   "backend": "onnx",   "name": "BiLSTM"},
}

def fetch_city_data(city_slug: str) -> pd.DataFrame:
    """Fetch last 48h of observations from PostgreSQL"""
    conn = psycopg2.connect(host="localhost", port=5432, database="airflow", user="airflow", password="airflow")
    query = """
        SELECT observed_at, pm2_5, pm10, carbon_monoxide, nitrogen_dioxide,
               sulphur_dioxide, ozone, us_aqi
        FROM aq.observations
        WHERE city_slug = %s AND observed_at >= NOW() - INTERVAL '72 hours'
        ORDER BY observed_at DESC
    """
    df = pd.read_sql_query(query, conn, params=(city_slug,))
    conn.close()
    if len(df) < 48:
        raise ValueError(f"Insufficient data for {city_slug}: {len(df)} rows")
    return df.sort_values("observed_at").reset_index(drop=True)

def engineer_features_tree(df: pd.DataFrame):
    """Build simple lag features matching training (no scaling)"""
    df = df.copy()
    for p in POLLUTANTS:
        df[p] = df[p].interpolate(method='linear', limit_direction='both')
    
    for p in POLLUTANTS:
        for lag in [1, 3, 6, 12, 24]:
            df[f'{p}_l{lag}'] = df[p].shift(lag)
    
    return df.dropna()

def predict_joblib(model_key: str, city_slug: str, scaler=None):
    """Run prediction using joblib tree model - RAW features, NO scaling"""
    df = fetch_city_data(city_slug)
    df_feat = engineer_features_tree(df)
    latest = df_feat.iloc[-1:]
    
    # Load feature names and use available ones
    feat_path = DEPLOY_DIR / "hyderabad" / f"tree_{city_slug}" / "feature_names.pkl"
    if feat_path.exists():
        feature_names = pickle.load(open(feat_path, 'rb'))
        available = [f for f in feature_names if f in latest.columns]
        X = latest[available].values.astype(np.float32)
    else:
        X = latest.values.astype(np.float32)
    
    X = np.nan_to_num(X, nan=0.0)
    
    # Find model
    model_name_map = {'xgboost': 'XGBoost_raw', 'lightgbm': 'LightGBM_raw'}
    base_name = model_name_map.get(model_key.lower(), f"{model_key.title()}_raw")
    model_dir = DEPLOY_DIR / "hyderabad" / f"tree_{city_slug}"
    model_path = model_dir / f"{base_name}.joblib"
    
    if not model_path.exists():
        for c in model_dir.glob("*.joblib"):
            if model_key.lower().replace('_','') in c.name.lower():
                model_path = c; break
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(str(model_path))
    y_pred = model.predict(X)
    return y_pred.flatten()

def predict_onnx(model_key: str, city_slug: str, scaler):
    """Run prediction using ONNX model"""
    import onnxruntime as ort
    
    df = fetch_city_data(city_slug)
    # Get last 168h of raw data
    df = df.tail(168)
    
    X = df[POLLUTANTS].values.astype(np.float32)
    X = pd.DataFrame(X).interpolate(method='linear', limit_direction='both').values
    
    # Scale
    from sklearn.preprocessing import MinMaxScaler
    scl = MinMaxScaler()
    X_scaled = scl.fit_transform(X)
    X_tensor = X_scaled[np.newaxis, :, :]
    
    # Find ONNX model
    onnx_path = DEPLOY_DIR / "hyderabad" / "onnx" / f"{model_key}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    out = sess.run(None, {'input': X_tensor.astype(np.float32)})
    y_pred = out[0].flatten()
    
    return y_pred

def predict_torch(model_key: str, city_slug: str, scaler):
    """Run prediction using TorchScript model"""
    import torch
    
    df = fetch_city_data(city_slug)
    df = df.tail(168)
    
    X = df[POLLUTANTS].values.astype(np.float32)
    X = pd.DataFrame(X).interpolate(method='linear', limit_direction='both').values
    
    from sklearn.preprocessing import MinMaxScaler
    scl = MinMaxScaler()
    X_scaled = scl.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
    
    # Find TorchScript model
    ts_path = DEPLOY_DIR / "hyderabad" / "onnx" / f"{model_key}.pt"
    if not ts_path.exists():
        raise FileNotFoundError(f"TorchScript not found: {ts_path}")
    
    model = torch.jit.load(str(ts_path))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy().flatten()
    
    return y_pred

def run_prediction(model_key: str) -> bool:
    """Run prediction for a single model across all 3 cities"""
    model_info = MODEL_REGISTRY.get(model_key)
    if not model_info:
        print(f"Unknown model: {model_key}")
        return False
    
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Predicting {model_info['name']}...")
    
    all_predictions = {}
    
    for city in CITIES:
        try:
            if model_info['backend'] == 'joblib':
                preds = predict_joblib(model_key, city, None)
            elif model_info['backend'] == 'onnx':
                preds = predict_onnx(model_key, city, None)
            elif model_info['backend'] == 'torch':
                preds = predict_torch(model_key, city, None)
            else:
                raise ValueError(f"Unknown backend: {model_info['backend']}")
            
            # Build forecast records
            now = datetime.now(timezone.utc)
            forecasts = []
            for h, val in enumerate(preds[:HORIZON], 1):
                forecasts.append({
                    "hour": h,
                    "timestamp": (now + pd.Timedelta(hours=h)).isoformat(),
                    "pm2_5": round(float(val), 1)
                })
            
            all_predictions[city] = {
                "city_name": CITY_NAMES[city],
                "forecasts": forecasts
            }
            print(f"  {city}: t+1h={forecasts[0]['pm2_5']:.1f} ug/m3")
            
        except Exception as e:
            print(f"  {city}: FAILED - {e}")
            all_predictions[city] = {"city_name": CITY_NAMES[city], "forecasts": [], "error": str(e)}
    
    # Save predictions
    output = {
        "model": model_key,
        "model_name": model_info["name"],
        "model_family": model_info["family"],
        "tier": model_info["tier"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": all_predictions
    }
    
    out_path = PREDICTIONS_DIR / f"{model_key}_latest.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved: {out_path}")
    return True

if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    success = run_prediction(model_key)
    sys.exit(0 if success else 1)
