from __future__ import annotations
import joblib
import numpy as np
import optuna
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS, TARGET_COLUMN

class EnsembleTrainer(AQBaseTrainer):
    def __init__(self, city, model_type, **kwargs):
        super().__init__(city, model_type, **kwargs)
        self.model_type = model_type

    def get_model(self, trial: Optional[optuna.Trial] = None, config: Optional[dict] = None):
        if self.model_type == "XGBoost":
            params = config or {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_jobs": -1,
                "random_state": self.seed
            }
            return MultiOutputRegressor(XGBRegressor(**params))
        
        elif self.model_type == "LightGBM":
            params = config or {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", -1, 15),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "n_jobs": -1,
                "random_state": self.seed,
                "verbosity": -1
            }
            return MultiOutputRegressor(LGBMRegressor(**params))
        
        elif self.model_type == "CatBoost":
            params = config or {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
                "random_seed": self.seed,
                "verbose": False,
                "allow_writing_files": False
            }
            return MultiOutputRegressor(CatBoostRegressor(**params))
        
        elif self.model_type == "SVR":
            params = config or {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            }
            return MultiOutputRegressor(SVR(**params))
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def objective(self, trial: optuna.Trial) -> float:
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
        X_val_flat = data["X_val"].reshape(data["X_val"].shape[0], -1)
        
        model = self.get_model(trial=trial)
        model.fit(X_train_flat, data["y_train"])
        
        preds = model.predict(X_val_flat)
        val_loss = np.mean((preds - data["y_val"])**2)
        return float(val_loss)

    def train(self, config: dict):
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
        X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
        
        model = self.get_model(config=config)
        model.fit(X_train_flat, data["y_train"])
        
        joblib.dump(model, self.output_dir / "best_model.joblib")
        
        preds = model.predict(X_test_flat)
        y_scaler = data["y_scaler"]
        
        # Correct inverse transform for multi-output
        preds_inv = y_scaler.inverse_transform(preds)
        actuals_inv = y_scaler.inverse_transform(data["y_test"])
        
        metrics = {
            "rmse": float(np.sqrt(np.mean((preds_inv - actuals_inv)**2))),
            "mae": float(np.mean(np.abs(preds_inv - actuals_inv))),
            "r2": float(r2_score(actuals_inv, preds_inv))
        }
        
        self.save_results(metrics, config)
        return metrics

if __name__ == "__main__":
    import argparse
    from sklearn.metrics import r2_score
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", choices=["XGBoost", "LightGBM", "CatBoost", "SVR"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    
    trainer = EnsembleTrainer(args.city, args.model)
    best_params = trainer.optimize(n_trials=args.trials)
    trainer.train(best_params)
