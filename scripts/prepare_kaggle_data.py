import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json

DATA_DIR = Path("data/india_aq_1y")
OUT_DIR = Path("data/kaggle_dataset")
ROOT_DIR = Path(__file__).resolve().parents[1]

TARGET_CITIES = [
    "delhi_aq_1y.csv",
    "hyderabad_aq_1y.csv",
    "bengaluru_aq_1y.csv"
]

FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "ozone",
]
TARGET_COLUMN = "us_aqi"
DATA_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]

def prepare_and_impute_city(city_file):
    print(f"Processing {city_file}...")
    in_path = DATA_DIR / city_file
    
    if not in_path.exists():
        print(f"File not found: {in_path}")
        return None
        
    df = pd.read_csv(in_path)
    missing_columns = [column for column in ["timestamp", *DATA_COLUMNS] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{in_path} is missing columns: {missing_columns}")
    
    # Ensure proper datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    source_rows = len(df)
    source_duplicates = int(df.index.duplicated().sum())
    
    # Drop duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample to strict hourly frequency to expose hidden gaps
    # This ensures the time series is continuous
    first_timestamp = df.index.min()
    last_timestamp = df.index.max()
    expected_hours = int(((last_timestamp - first_timestamp) / pd.Timedelta(hours=1)) + 1)
    df = df.resample('1h').asfreq()
    missing_timestamps = int(expected_hours - (source_rows - source_duplicates))
    
    print(f"  Total hours: {len(df)}")
    print(f"  Missing values before imputation:")
    missing_before = df[DATA_COLUMNS].isna().sum()
    print(missing_before)
    
    # Imputation Strategy:
    # 1. Linear interpolation for small gaps (up to 6 hours)
    for col in DATA_COLUMNS:
        df[col] = df[col].interpolate(method='linear', limit=6)
        
    # 2. Forward fill for larger gaps up to 24 hours (carrying over previous day's data)
    for col in DATA_COLUMNS:
        df[col] = df[col].ffill(limit=24)
        
    # 3. Backward fill for anything at the very beginning of the dataset
    for col in DATA_COLUMNS:
        df[col] = df[col].bfill()
        
    # 4. Fill remaining extreme gaps with median of the month
    df['month'] = df.index.month
    for col in DATA_COLUMNS:
        df[col] = df[col].fillna(df.groupby('month')[col].transform('median'))
        # If any absolute gaps remain, fill with global median
        df[col] = df[col].fillna(df[col].median())
        
    df = df.drop(columns=['month'])
    
    print(f"  Missing values after imputation:")
    missing_after = df[DATA_COLUMNS].isna().sum()
    print(missing_after)
    if missing_after.any():
        raise ValueError(f"{city_file} still has missing values after imputation")
    
    # Add city name back in case it's needed
    city_name = city_file.replace('_aq_1y.csv', '')
    df['city'] = city_name
    
    # Save clean dataset
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"clean_{city_file}"
    df.reset_index().to_csv(out_path, index=False)
    print(f"Saved imputed dataset to {out_path}\n")
    return {
        "city": city_name,
        "source_file": str(in_path),
        "output_file": str(out_path),
        "source_rows": source_rows,
        "output_rows": len(df),
        "first_timestamp": first_timestamp.isoformat(),
        "last_timestamp": last_timestamp.isoformat(),
        "expected_hours": expected_hours,
        "missing_timestamps_after_resample": missing_timestamps,
        "duplicate_timestamps": source_duplicates,
        "missing_before": missing_before.astype(int).to_dict(),
        "missing_after": missing_after.astype(int).to_dict(),
    }

if __name__ == "__main__":
    audits = []
    for city_file in TARGET_CITIES:
        audit = prepare_and_impute_city(city_file)
        if audit is not None:
            audits.append(audit)
    shutil.copy2(ROOT_DIR / "scripts" / "kaggle_benchmarking_suite.py", OUT_DIR / "kaggle_benchmarking_suite.py")
    manifest = {
        "target_cities": ["delhi", "hyderabad", "bengaluru"],
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "cadence": "hourly",
        "source": "Open-Meteo Air Quality API",
        "feature_policy": "Five pollutant inputs only; historical us_aqi is the prediction target, not a supervised input.",
        "imputation": [
            "linear interpolation up to 6 hours",
            "forward/backward fill up to 24 hours",
            "monthly median fallback",
            "global median fallback",
        ],
        "city_audits": audits,
        "benchmark_script": "kaggle_benchmarking_suite.py",
    }
    (OUT_DIR / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Data preparation complete.")
