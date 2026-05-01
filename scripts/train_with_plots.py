import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.individual_trainers.ensemble_trainers import EnsembleTrainer

def train_with_plots(city: str, model_type: str = "XGBoost", epochs: int = 100):
    """Train XGBoost model and generate plots"""
    
    output_dir = Path(f"outputs/final_deploy/{model_type}/{city}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Training {model_type} on {city}")
    print(f"{'='*50}")
    
    trainer = EnsembleTrainer(city, model_type, seed=42)
    
    config = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42
    }
    
    df = trainer.load_data()
    data = trainer.prepare_data(df, lookback=168, horizon=24)
    
    X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
    
    print(f"Data: {X_train_flat.shape[0]} train, {X_test_flat.shape[0]} test")
    print(f"Features: {X_train_flat.shape[1]}")
    
    model = trainer.get_model(config=config)
    
    from sklearn.model_selection import learning_curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_flat, data["y_train"],
        train_sizes=train_sizes, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_rmse, 'b-o', label='Train RMSE')
    ax.plot(train_sizes, val_rmse, 'r-o', label='Val RMSE')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('RMSE')
    ax.set_title(f'{model_type} Learning Curve - {city}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curve.png', dpi=150)
    print(f"Saved: {output_dir / 'learning_curve.png'}")
    
    model.fit(X_train_flat, data["y_train"])
    
    preds = model.predict(X_test_flat)
    y_scaler = data["y_scaler"]
    preds_inv = y_scaler.inverse_transform(preds)
    actuals_inv = y_scaler.inverse_transform(data["y_test"])
    
    rmse = np.sqrt(np.mean((preds_inv - actuals_inv)**2))
    mae = np.mean(np.abs(preds_inv - actuals_inv))
    r2 = 1 - np.sum((actuals_inv - preds_inv)**2) / np.sum((actuals_inv - actuals_inv.mean())**2)
    
    print(f"\nMetrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    joblib.dump(model, output_dir / 'best_model.joblib')
    joblib.dump(data["y_scaler"], output_dir / 'scaler.joblib')
    print(f"Model saved: {output_dir / 'best_model.joblib'}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(actuals_inv, preds_inv, alpha=0.5)
    axes[0].plot([actuals_inv.min(), actuals_inv.max()], [actuals_inv.min(), actuals_inv.max()], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Parity Plot - {city}\nRMSE={rmse:.2f}, R²={r2:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    errors = preds_inv.flatten() - actuals_inv.flatten()
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution - {city}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parity_plot.png', dpi=150)
    print(f"Saved: {output_dir / 'parity_plot.png'}")
    
    results = {
        "city": city,
        "model": model_type,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_samples": X_train_flat.shape[0],
        "test_samples": X_test_flat.shape[0]
    }
    
    pd.DataFrame([results]).to_csv(output_dir / 'metrics.csv', index=False)
    print(f"Metrics saved: {output_dir / 'metrics.csv'}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="delhi")
    parser.add_argument("--model", default="XGBoost")
    args = parser.parse_args()
    
    train_with_plots(args.city, args.model)