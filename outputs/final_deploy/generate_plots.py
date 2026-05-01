import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs/final_deploy/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv("outputs/smoke_bench/benchmark_summary.csv")

keep_models = ['ARIMA', 'VAR', 'XGBoost', 'LightGBM', 'CatBoost', 'LSTM', 'GRU', 'Bi-LSTM', 'CNN-LSTM', 'Transformer', 'Autoformer']
df = df[df['model'].isin(keep_models)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_avg = df.groupby('model')['rmse'].mean().sort_values()
colors = ['green' if m in ['XGBoost', 'CatBoost', 'VAR', 'ARIMA'] else 'steelblue' for m in model_avg.index]
axes[0].barh(model_avg.index, model_avg.values, color=colors)
axes[0].set_xlabel('Average RMSE')
axes[0].set_title('Model Comparison: Average RMSE (Lower is Better)')
axes[0].axvline(x=model_avg.min(), color='red', linestyle='--', label=f'Best: {model_avg.min():.1f}')

for i, city in enumerate(df['city'].unique()):
    city_df = df[df['city'] == city]
    city_avg = city_df.groupby('model')['rmse'].mean()
    axes[1].plot(city_avg.index, city_avg.values, marker='o', label=city)

axes[1].set_xlabel('Model')
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE by Model and City')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'model_comparison.png'}")

fig2, ax = plt.subplots(figsize=(10, 6))
cities = df['city'].unique()
x = np.arange(len(keep_models))
width = 0.25

for i, city in enumerate(cities):
    city_df = df[df['city'] == city]
    rmse_vals = [city_df[city_df['model'] == m]['rmse'].values[0] if len(city_df[city_df['model'] == m]) > 0 else 0 for m in keep_models]
    ax.bar(x + i*width, rmse_vals, width, label=city)

ax.set_ylabel('RMSE')
ax.set_title('RMSE by Model and City')
ax.set_xticks(x + width)
ax.set_xticklabels(keep_models, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rmse_by_city.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'rmse_by_city.png'}")

print("\nPlot generation complete!")