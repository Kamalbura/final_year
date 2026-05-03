"""Train tree-based ML models + SVR for all 3 cities (Delhi, Hyderabad, Bengaluru)"""
import pandas as pd
import numpy as np
import json
import pickle
import joblib
import time
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\burak\ptojects\final_year\data\kaggle_dataset")
OUT_DIR = Path(r"C:\Users\burak\ptojects\final_year\outputs\hyderabad")
FEATURES = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'us_aqi']
TARGET = 'pm2_5'
LOOKBACK = 168  # 7 days
HORIZON = 24
SPLIT = (0.70, 0.15, 0.15)
SEED = 42

CITIES = {
    'delhi': 'clean_delhi_aq_1y.csv',
    'hyderabad': 'clean_hyderabad_aq_1y.csv',
    'bengaluru': 'clean_bengaluru_aq_1y.csv',
}

from sklearn.multioutput import MultiOutputRegressor

MODELS = {
    'XGBoost': lambda: xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
    'LightGBM': lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1),
    'CatBoost': lambda: None,
    'Random_Forest': lambda: RandomForestRegressor(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1),
    'SVR': lambda: MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma=0.01, epsilon=0.1)),
}

def load_city_data(city_slug):
    """Load and preprocess data for a city"""
    csv_path = DATA_DIR / CITIES[city_slug]
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    if 'city' in df.columns:
        df = df.drop(columns=['city'])
    
    # Handle missing by interpolation
    for col in FEATURES:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    X_raw = df[FEATURES].values.astype(np.float32)
    y_raw = df[TARGET].values.astype(np.float32)
    
    # Create sliding windows: 168h input -> 24h output (7*168 = 1176 features)
    samples = []
    targets = []
    for i in range(len(X_raw) - LOOKBACK - HORIZON):
        samples.append(X_raw[i:i + LOOKBACK].flatten())
        targets.append(y_raw[i + LOOKBACK:i + LOOKBACK + HORIZON])
    X_win = np.array(samples)
    y_win = np.array(targets)
    
    # Chronological split
    n = len(X_win)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    
    # Scale features and targets
    scl_X = MinMaxScaler()
    scl_y = MinMaxScaler()
    
    X_train = scl_X.fit_transform(X_win[:n_train])
    y_train = scl_y.fit_transform(y_win[:n_train])
    X_val = scl_X.transform(X_win[n_train:n_train + n_val])
    y_val = scl_y.transform(y_win[n_train:n_train + n_val])
    X_test = scl_X.transform(X_win[n_train + n_val:])
    y_test = scl_y.transform(y_win[n_train + n_val:])
    unscaled_y_test = y_win[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scl_y, unscaled_y_test

def evaluate(model, X_test, y_test, y_scaler, unscaled_y):
    """Evaluate model and return metrics"""
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    # RMSE per sample (average over all hours)
    rmse = np.sqrt(np.mean((unscaled_y - y_pred) ** 2))
    mae = np.mean(np.abs(unscaled_y - y_pred))
    r2 = 1 - np.sum((unscaled_y - y_pred) ** 2) / np.sum((unscaled_y - np.mean(unscaled_y)) ** 2)
    
    return float(rmse), float(mae), float(r2)

def train_city_models(city_slug):
    """Train all models for one city"""
    print(f"\n{'='*60}")
    print(f"Training models for {city_slug.upper()}")
    print(f"{'='*60}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler, unscaled_y = load_city_data(city_slug)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    results = {}
    city_dir = OUT_DIR / f"tree_{city_slug}"
    city_dir.mkdir(parents=True, exist_ok=True)
    
    for name, factory in MODELS.items():
        print(f"\n  [{name}]", end=" ")
        t0 = time.time()
        
        try:
            if name == 'CatBoost':
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1,
                                          random_seed=SEED, verbose=0, thread_count=-1)
            else:
                model = factory()
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            rmse, mae, r2 = evaluate(model, X_test, y_test, y_scaler, unscaled_y)
            
            # Save
            model_path = city_dir / f"{name}.joblib"
            scaler_path = city_dir / f"{name}_y_scaler.pkl"
            
            joblib.dump(model, model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(y_scaler, f)
            
            elapsed = time.time() - t0
            print(f"RMSE={rmse:.2f} MAE={mae:.2f} R²={r2:.3f} [{elapsed:.1f}s]")
            
            results[name] = {
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 3),
                'train_seconds': round(elapsed, 1),
                'model_path': str(model_path)
            }
            
        except ImportError as e:
            print(f"SKIP - {e}")
            results[name] = {'rmse': None, 'mae': None, 'r2': None, 'train_seconds': 0, 'model_path': None}
        except Exception as e:
            print(f"FAILED - {e}")
            results[name] = {'rmse': None, 'mae': None, 'r2': None, 'train_seconds': 0, 'model_path': None}
    
    # Save results
    with open(city_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    all_results = {}
    for city_slug in CITIES:
        all_results[city_slug] = train_city_models(city_slug)
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - Tree Model Training Complete")
    print(f"{'='*80}")
    
    for city in CITIES:
        print(f"\n{'-'*40}")
        print(f"  {city.upper()}")
        print(f"{'-'*40}")
        for model_name, metrics in all_results[city].items():
            if metrics['rmse'] is not None:
                print(f"    {model_name:18s} RMSE={metrics['rmse']:6.2f}  R²={metrics['r2']:7.3f}  {metrics['train_seconds']:6.1f}s")
    
    print(f"\nModels saved to: {OUT_DIR}/tree_*/")

if __name__ == '__main__':
    main()
