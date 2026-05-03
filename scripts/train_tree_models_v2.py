"""Train tree-based ML models + SVR for all 3 cities (Delhi, Hyderabad, Bengaluru)
   Uses efficient feature engineering: lag features + rolling stats, not raw 168h window."""
import pandas as pd
import numpy as np
import json
import pickle
import joblib
import time
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\burak\ptojects\final_year\data\kaggle_dataset")
OUT_DIR = Path(r"C:\Users\burak\ptojects\final_year\outputs\hyderabad")
POLLUTANTS = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'us_aqi']
TARGET = 'pm2_5'
HORIZON = 24
SPLIT = (0.70, 0.15, 0.15)
SEED = 42
LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

CITIES = {'delhi': 'clean_delhi_aq_1y.csv', 'hyderabad': 'clean_hyderabad_aq_1y.csv', 'bengaluru': 'clean_bengaluru_aq_1y.csv'}

def engineer_features(df):
    """Build time-series features from raw pollutant data"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    for p in POLLUTANTS:
        for lag in LAGS:
            df[f'{p}_lag_{lag}h'] = df[p].shift(lag)
        for w in ROLLING_WINDOWS:
            df[f'{p}_roll_{w}h'] = df[p].rolling(w, min_periods=1).mean()
    
    # Cross-pollutant interactions
    df['pm25_pm10_ratio'] = df['pm2_5'] / df['pm10'].replace(0, np.nan)
    df['co_no2_ratio'] = df['carbon_monoxide'] / df['nitrogen_dioxide'].replace(0, np.nan)
    
    # Derived: 24h change
    for p in POLLUTANTS:
        df[f'{p}_delta_24h'] = df[p] - df[p].shift(24)
    
    df = df.dropna()
    return df

def load_and_prepare(city_slug):
    csv_path = DATA_DIR / CITIES[city_slug]
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    if 'city' in df.columns:
        df = df.drop(columns=['city'])
    
    # Interpolate missing pollutant values
    for p in POLLUTANTS:
        df[p] = df[p].interpolate(method='linear', limit_direction='both')
    
    # Engineer features
    df_feat = engineer_features(df)
    
    # Target: next 24h of PM2.5
    y_cols = [f'pm2_5_h{h}' for h in range(1, HORIZON + 1)]
    for h in range(1, HORIZON + 1):
        df_feat[f'pm2_5_h{h}'] = df_feat['pm2_5'].shift(-h)
    
    df_feat = df_feat.dropna()
    
    # Feature columns (exclude timestamp, target cols)
    feature_cols = [c for c in df_feat.columns
                    if c not in ['timestamp'] + y_cols + POLLUTANTS
                    and 'pm2_5_h' not in c
                    and 'target' not in c]
    
    X = df_feat[feature_cols].values.astype(np.float32)
    y = df_feat[y_cols].values.astype(np.float32)
    
    # Chronological split - NO SCALING for tree models
    n = len(X)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    return X_train, y_train, X_test, y_test, features

def main():
    for city_slug in CITIES:
        print(f"\n{'='*50}")
        print(f"  {city_slug.upper()}")
        print(f"{'='*50}")
        
        X_train, y_train, X_test, y_test, y_scaler, unscaled_y, features = load_and_prepare(city_slug)
        print(f"  Features: {len(features)}, Train: {X_train.shape}, Test: {y_test.shape}")
        
        city_dir = OUT_DIR / f"tree_{city_slug}"
        city_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Models to train
        model_defs = {
            'XGBoost': ('Tree', xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=SEED, n_jobs=-1)),
            'LightGBM': ('Tree', MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=SEED, n_jobs=-1, verbose=-1))),
            'Random_Forest': ('Tree', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1)),
            'SVR': ('Tree', MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma=0.01, epsilon=0.1))),
        }
        
        for name, (family, model) in model_defs.items():
            print(f"    {name:18s}", end=' ', flush=True)
            t0 = time.time()
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred = y_scaler.inverse_transform(y_pred)
                
                rmse = float(np.sqrt(np.mean((unscaled_y - y_pred) ** 2)))
                mae = float(np.mean(np.abs(unscaled_y - y_pred)))
                r2 = float(1 - np.sum((unscaled_y - y_pred) ** 2) / np.sum((unscaled_y - np.mean(unscaled_y)) ** 2))
                elapsed = time.time() - t0
                
                # Save model and both scalers
                joblib.dump(model, city_dir / f'{name}.joblib')
                joblib.dump(scl_X, city_dir / 'x_scaler.pkl')
                joblib.dump(scl_y, city_dir / 'y_scaler.pkl')
                joblib.dump({'feature_names': features, 'x_scaler': scl_X, 'y_scaler': scl_y}, city_dir / f'{name}_bundle.pkl')
                
                print(f'RMSE={rmse:.2f} R²={r2:.3f} [{elapsed:.1f}s]')
                results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'train_seconds': round(elapsed,1), 'family': family}
                
            except Exception as e:
                print(f'FAIL: {e}')
        
        # Save results and feature names
        with open(city_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        with open(city_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(features, f)
        joblib.dump(y_scaler, city_dir / 'y_scaler.pkl')
    
    print(f"\nDone! Models in {OUT_DIR}/tree_*/")

if __name__ == '__main__':
    main()
