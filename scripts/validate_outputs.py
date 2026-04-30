"""Quick validation of Kaggle kernel outputs."""
import csv
import json
from pathlib import Path

rows = list(csv.DictReader(open("outputs/kaggle_benchmarks/forecast_rows.csv")))
print("=== Forecast Validation ===")
print(f"Total rows: {len(rows)}")
for city in ["delhi", "hyderabad", "bengaluru"]:
    cr = [r for r in rows if r["city_slug"] == city]
    aqis = [float(r["predicted_us_aqi"]) for r in cr]
    model = cr[0]["model_type"]
    conf = cr[0]["confidence"]
    print(f"  {city}: {len(cr)} rows | model={model} | AQI [{round(min(aqis),1)}, {round(max(aqis),1)}] | confidence={conf}")

print()
results = json.load(open("outputs/kaggle_benchmarks/benchmarks_results.json"))
print("=== Best Models ===")
for b in results["best_models"]:
    print(f"  {b['city']}: {b['model']} | RMSE={round(b['rmse'],2)} | MAE={round(b['mae'],2)} | R2={round(b['r2'],3)}")

ok_rows = [r for r in results["rows"] if r["status"] == "ok"]
skip_rows = [r for r in results["rows"] if r["status"] == "skipped"]
print(f"\nModels: {len(ok_rows)} succeeded, {len(skip_rows)} skipped (CUDA mismatch - fixed in v4)")
print("\nConfig used:")
cfg = results["config"]
print(f"  features={cfg['features']}")
print(f"  target={cfg['target']}, lookback={cfg['lookback']}h, horizon={cfg['horizon']}h, device={cfg['device']}")
