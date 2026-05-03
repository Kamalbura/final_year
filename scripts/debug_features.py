import pickle, pandas as pd, psycopg2, numpy as np
from pathlib import Path

feat_path = Path('/home/bura/projects/final_year/deployment_models/hyderabad/tree_hyderabad/feature_names.pkl')
feat_names = pickle.load(open(feat_path, 'rb'))
print(f'Expected {len(feat_names)} features')

conn = psycopg2.connect(host='localhost', port=5432, database='airflow', user='airflow', password='airflow')
df = pd.read_sql_query(
    'SELECT observed_at, pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, us_aqi FROM aq.observations WHERE city_slug=%s AND observed_at >= NOW() - INTERVAL %s ORDER BY observed_at DESC',
    conn, params=('hyderabad', '72 hours'))
conn.close()
df = df.sort_values('observed_at').reset_index(drop=True)

P = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'us_aqi']
for p in P:
    df[p] = df[p].interpolate(method='linear', limit_direction='both')

df['hour'] = pd.to_datetime(df['observed_at']).dt.hour
df['day_of_week'] = pd.to_datetime(df['observed_at']).dt.dayofweek
df['month'] = pd.to_datetime(df['observed_at']).dt.month

for p in P:
    for lag in [1,2,3,6,12,24]:
        df[f'{p}_lag_{lag}h'] = df[p].shift(lag)
    for w in [3,6,12,24]:
        df[f'{p}_roll_{w}h'] = df[p].rolling(w, min_periods=1).mean()

df['pm25_pm10_ratio'] = df['pm2_5'] / df['pm10'].replace(0, np.nan)
df['co_no2_ratio'] = df['carbon_monoxide'] / df['nitrogen_dioxide'].replace(0, np.nan)
for p in P:
    df[f'{p}_delta_24h'] = df[p] - df[p].shift(24)

df = df.dropna()
latest = df.iloc[-1:]
available = [f for f in feat_names if f in latest.columns]
missing = set(feat_names) - set(available)
print(f'Available: {len(available)}, Missing: {missing}')
print(f'Latest cols: {len(latest.columns)}')
