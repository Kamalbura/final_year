from __future__ import annotations
import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS, TARGET_COLUMN

class RFTrainer(AQBaseTrainer):
    def __init__(self, city, **kwargs):
        super().__init__(city, "RandomForest", **kwargs)

    def objective(self, trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        # Flatten for RF
        X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
        X_val_flat = data["X_val"].reshape(data["X_val"].shape[0], -1)
        
        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=self.seed
        ))
        
        model.fit(X_train_flat, data["y_train"])
        preds = model.predict(X_val_flat)
        val_loss = np.mean((preds - data["y_val"])**2)
        
        return float(val_loss)

    def train(self, config: dict):
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
        X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
        
        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            n_jobs=4,
            random_state=self.seed
        ))
        
        model.fit(X_train_flat, data["y_train"])
        
        joblib.dump(model, self.output_dir / "best_model.joblib")
        
        preds = model.predict(X_test_flat)
        y_scaler = data["y_scaler"]
        preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        actuals_inv = y_scaler.inverse_transform(data["y_test"].reshape(-1, 1)).reshape(data["y_test"].shape)
        
        metrics = {
            "rmse": float(np.sqrt(np.mean((preds_inv - actuals_inv)**2))),
            "mae": float(np.mean(np.abs(preds_inv - actuals_inv)))
        }
        
        self.save_results(metrics, config)
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    trainer = RFTrainer(args.city)
    best_params = trainer.optimize(n_trials=args.trials)
    trainer.train(best_params)
