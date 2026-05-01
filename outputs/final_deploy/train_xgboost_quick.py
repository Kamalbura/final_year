import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

DATA_DIR = Path("data/kaggle_dataset")
OUTPUT_DIR = Path("outputs/final_deploy")
OUTPUT_DIR.mkdir(exist_ok=True)

def train_xgboost(city: str):
    """Quick XGBoost training with plots"""
    
    output_dir = OUTPUT_DIR / "XGBoost" / city
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*40}")
    print(f"XGBoost - {city}")
    print(f"{'='*40}")
    
    df = pd.read_csv(DATA_DIR / f"clean_{city}_aq_1y.csv")
    
    FEATURES = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
    TARGET = "us_aqi"
    
    df = df.dropna(subset=FEATURES + [TARGET])
    
    lookback = 24
    for f in FEATURES:
        for j in range(1, lookback + 1):
            df[f"{f}_lag{j}"] = df[f].shift(j)
    
    lag_cols = [c for c in df.columns if "_lag" in c]
    X = df[lag_cols].dropna()
    y = df.loc[X.index, TARGET]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    mae = np.mean(np.abs(preds - y_test))
    r2 = model.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(y_test, preds, alpha=0.4, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{city} - Parity Plot\nRMSE={rmse:.2f}, R2={r2:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    errors = preds - y_test
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{city} - Error Histogram')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parity_plot.png', dpi=150)
    print(f"Saved: {output_dir / 'parity_plot.png'}")
    
    joblib.dump(model, output_dir / 'model.joblib')
    print(f"Model saved: {output_dir / 'model.joblib'}")
    
    return {"city": city, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

cities = ["delhi", "hyderabad", "bengaluru"]
results = []

for city in cities:
    r = train_xgboost(city)
    results.append(r)

print(f"\n{'='*40}")
print("SUMMARY - XGBoost")
print(f"{'='*40}")
for r in results:
    print(f"{r['city']:12} RMSE: {r['rmse']:.2f}  R2: {r['r2']:.3f}")

pd.DataFrame(results).to_csv(OUTPUT_DIR / "XGBoost_summary.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'XGBoost_summary.csv'}")