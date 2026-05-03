"""Quick XGBoost trainer - no scaling, just raw features"""
import pandas as pd, numpy as np, joblib, pickle, time
from pathlib import Path
import xgboost as xgb

DATA = Path(r"C:\Users\burak\ptojects\final_year\data\kaggle_dataset")
OUT = Path(r"C:\Users\burak\ptojects\final_year\outputs\hyderabad")
P = ['pm2_5','pm10','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide','ozone','us_aqi']

for city in ['hyderabad']:
    print(f'Loading {city}...')
    df = pd.read_csv(DATA / f'clean_{city}_aq_1y.csv').sort_values('timestamp')
    for c in ['city','timestamp']:
        if c in df.columns: df = df.drop(columns=[c])
    for p in P: df[p] = df[p].interpolate('linear', limit_direction='both')
    
    for p in P:
        for lag in [1,3,6,12,24]:
            df[f'{p}_l{lag}'] = df[p].shift(lag)
    
    for h in range(1,25):
        df[f't{h}'] = df['pm2_5'].shift(-h)
    
    df = df.dropna()
    feat = [c for c in df.columns if not c.startswith('t') and c not in P]
    X = df[feat].values.astype(np.float32)
    Y = df[[f't{h}' for h in range(1,25)]].values.astype(np.float32)
    
    n = len(X); nt = int(n*0.7); ns = int(n*0.15)
    print(f'Training {nt} samples, {X.shape[1]} features...')
    
    t0 = time.time()
    m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    m.fit(X[:nt], Y[:nt])
    yp = m.predict(X[nt+ns:])
    
    rmse = float(np.sqrt(np.mean((Y[nt+ns:] - yp)**2)))
    r2 = float(1 - np.sum((Y[nt+ns:]-yp)**2) / np.sum((Y[nt+ns:]-np.mean(Y[nt+ns:]))**2))
    
    print(f'XGBoost: RMSE={rmse:.2f} R2={r2:.3f} [{time.time()-t0:.0f}s]')
    
    o = OUT / f'tree_{city}'; o.mkdir(exist_ok=True)
    jb = o / 'XGBoost_raw.joblib'
    joblib.dump(m, jb)
    pickle.dump(feat, open(o / 'feature_names.pkl','wb'))
    print(f'Saved: {jb} ({jb.stat().st_size//1024}KB)')

print('Done')
