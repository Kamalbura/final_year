"""Final validation of pipeline output."""
import pickle

with open("data/processed/pipeline_output.pkl", "rb") as f:
    po = pickle.load(f)

print("Pipeline output keys:", list(po.keys()))
print("Train shape:", po["train"].shape)
print("Val shape:", po["val"].shape)
print("Test shape:", po["test"].shape)
print("Features:", len(po["feature_names"]))
print("Scalers:", list(po["scalers"].keys()))

# NaN check
for split in ["train", "val", "test"]:
    nulls = po[split].isnull().sum().sum()
    print(f"  {split} NaN count: {nulls}")

# AQI distribution
for split in ["train_raw", "val_raw", "test_raw"]:
    df = po[split]
    print(f"  {split} AQI — mean: {df['AQI'].mean():.1f}, "
          f"std: {df['AQI'].std():.1f}, "
          f"min: {df['AQI'].min():.1f}, max: {df['AQI'].max():.1f}")
    print(f"    date range: {df.index[0]} to {df.index[-1]}")

# Check raw data dir
import os
raw_files = os.listdir("data/raw")
print(f"\nRaw files: {raw_files}")
print(f"Processed files: {os.listdir('data/processed')}")
