import pandas as pd
import numpy as np
import os

FILE_NAME = 'feeds.csv'
print(f"--- [Phase 1] Loading data from '{FILE_NAME}' ---")

if not os.path.exists(FILE_NAME):
    print(f"Error: '{FILE_NAME}' not found in working directory: {os.getcwd()}\nPlease ensure the file is available and retry.")
    exit(2)

# attempt to read excel first, then csv
last_err = None
try:
    df = pd.read_excel(FILE_NAME)
    print(f"Loaded with pd.read_excel. Shape: {df.shape}")
except Exception as e_excel:
    last_err = e_excel
    try:
        df = pd.read_csv(FILE_NAME)
        print(f"Loaded with pd.read_csv. Shape: {df.shape}")
    except Exception as e_csv:
        print('Failed to read file with both read_excel and read_csv')
        print('read_excel error:', repr(last_err))
        print('read_csv error:', repr(e_csv))
        raise

print('\n--- [Phase 2] Initial Data Inspection ---\n')

print('[INFO] DataFrame Info:')
import io
buf = io.StringIO()
df.info(buf=buf)
info = buf.getvalue()
print(info)

print('[HEAD] First 10 Rows:')
print(df.head(10).to_string())

print('\n[DESCRIBE] Statistical Summary:')
print(df.describe(include='all').to_string())

print('\n[NULL COUNT] Missing values per column:')
print(df.isnull().sum().to_string())

# Explicit preprocessing
print("\n--- [Phase 2 Explicit] Preprocessing ---")
print("Converting 'created_at' to datetime (UTC) and setting as index...")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
print(f"created_at NaT count: {df['created_at'].isna().sum()}")

# Set index and sort
df = df.set_index('created_at').sort_index()
print(f"Index set. Range: {df.index.min()} to {df.index.max()}")

# Rename sensor columns
rename_map = {'field1':'pm2_5','field2':'pm10','field3':'temperature','field4':'humidity'}
existing_renames = {k:v for k,v in rename_map.items() if k in df.columns}
df = df.rename(columns=existing_renames)
print('Columns after rename:', df.columns.tolist())

# Report missing values
print('\nMissing values before cleaning:')
print(df.isnull().sum())

# Drop all-null columns
cols_all_null = [c for c in df.columns if df[c].isnull().all()]
if cols_all_null:
    print('Dropping all-null columns:', cols_all_null)
    df = df.drop(columns=cols_all_null)

# Drop entry_id
if 'entry_id' in df.columns:
    df = df.drop(columns=['entry_id'])
    print("Dropped column: entry_id")

# Ensure sensor columns are numeric
sensor_cols = ['pm2_5', 'pm10', 'temperature', 'humidity']
for c in sensor_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Handle outliers
for col in sensor_cols:
    if col in df.columns:
        if 'pm' in col:
            sentinel_threshold = 500
        elif col == 'humidity':
            sentinel_threshold = 100
        elif col == 'temperature':
            sentinel_threshold = 50
        else:
            sentinel_threshold = np.inf
        bad = df[col] > sentinel_threshold
        if bad.any():
            print(f"Setting {bad.sum()} sentinel values in {col} to NaN")
            df.loc[bad, col] = np.nan

print('\nMissing values after basic cleaning:')
print(df.isnull().sum())

# Aggressive filling with rolling window
print("\nStarting aggressive missing value correction using rolling window averages...")
window_size = 10
for col in sensor_cols:
    if col in df.columns:
        rolling_mean = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
        df[col] = df[col].fillna(rolling_mean)
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

print(f"Missing values after rolling window filling: {df[sensor_cols].isnull().sum().sum()} total")
print("\nFinal missing values per sensor column:")
print(df[sensor_cols].isnull().sum())

# Resample to 15-minute frequency
print("\nResampling to 15-minute frequency using mean aggregation...")
df_15T = df[sensor_cols].resample('15T').mean()
print(f"Resampled shape: {df_15T.shape}")
print(f"15T data range: {df_15T.index.min()} to {df_15T.index.max()}")

coverage = df_15T.notna().any(axis=1).sum() / len(df_15T) * 100
print(f"15T coverage (15-min intervals with any data): {coverage:.1f}%")

# Save
output_file = 'feeds_cleaned_15T.csv'
df_15T.to_csv(output_file)
print(f"\nSaved cleaned 15T dataset to '{output_file}'")