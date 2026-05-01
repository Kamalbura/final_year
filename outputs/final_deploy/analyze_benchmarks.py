import pandas as pd
import numpy as np

# Read the benchmark data
df = pd.read_csv('outputs/smoke_bench/benchmark_summary.csv')

# Filter to keep only the best models
keep_models = [
    'ARIMA', 'VAR', 'XGBoost', 'LightGBM', 'CatBoost',
    'LSTM', 'GRU', 'Bi-LSTM', 'CNN-LSTM', 'Transformer', 'Autoformer'
]

df = df[df['model'].isin(keep_models)]

# Best per city
print("=== BEST MODELS BY CITY (Lowest RMSE) ===")
for city in df['city'].unique():
    city_df = df[df['city'] == city]
    best = city_df.loc[city_df['rmse'].idxmin()]
    print(f"{city}: {best['model']} (RMSE={best['rmse']:.2f}, R2={best['r2']:.2f})")

print("\n=== AVERAGE RMSE BY MODEL ===")
model_avg = df.groupby('model')['rmse'].mean().sort_values()
for m, r in model_avg.items():
    print(f"{m}: {r:.2f}")

# Save final benchmark summary
df.to_csv('outputs/final_deploy/benchmark_summary.csv', index=False)
print("\nSaved to outputs/final_deploy/benchmark_summary.csv")