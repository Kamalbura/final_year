"""
run_preprocessing.py — Execute the preprocessing pipeline on collected data.

Loads the combined station data, runs the full preprocessing pipeline
(impute, temporal features, scaling, splits), and saves the results.
"""
import sys
import os
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.aq_forecast.config import DataConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.aq_forecast.data.download import load_all_station_data
from src.aq_forecast.data.preprocessing import run_preprocessing_pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("  AQI Preprocessing Pipeline — Hyderabad")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/2] Loading raw station data...")
    raw_df = load_all_station_data(RAW_DATA_DIR)
    print(f"  Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")
    print(f"  Stations: {raw_df['station'].nunique()}")

    # For preprocessing, aggregate across stations (mean per hour)
    # or use a single station. Using all-station mean for a city-level model.
    print("\n  Aggregating to city-level hourly averages...")
    numeric_cols = raw_df.select_dtypes(include=["number"]).columns
    city_df = raw_df[numeric_cols].resample("1h").mean()
    print(f"  City-level data: {len(city_df)} rows")

    # Step 2: Run preprocessing
    print("\n[2/2] Running preprocessing pipeline...")
    config = DataConfig()
    pipeline_output = run_preprocessing_pipeline(city_df, config)

    # Print results
    print("\n" + "=" * 60)
    print("  PREPROCESSING RESULTS")
    print("=" * 60)
    print(f"  Features: {len(pipeline_output['feature_names'])}")
    print(f"  Feature list: {pipeline_output['feature_names']}")
    print(f"  Train: {len(pipeline_output['train'])} rows")
    print(f"  Val:   {len(pipeline_output['val'])} rows")
    print(f"  Test:  {len(pipeline_output['test'])} rows")

    # Save pipeline output
    output_path = os.path.join(PROCESSED_DATA_DIR, "pipeline_output.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_output, f)
    print(f"\n  Saved pipeline output to {output_path}")

    # Quick validation
    print("\n  Train data sample (first 3 rows):")
    print(pipeline_output["train"].head(3).to_string())

    print("\n  Target (AQI) stats on raw train split:")
    raw_train = pipeline_output["train_raw"]
    print(f"    Mean: {raw_train['AQI'].mean():.1f}")
    print(f"    Std:  {raw_train['AQI'].std():.1f}")
    print(f"    Min:  {raw_train['AQI'].min():.1f}")
    print(f"    Max:  {raw_train['AQI'].max():.1f}")
