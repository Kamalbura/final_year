from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, r2_score
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS, TARGET_COLUMN

class StatisticalTrainer(AQBaseTrainer):
    def __init__(self, city, model_type, **kwargs):
        super().__init__(city, model_type, **kwargs)
        self.model_type = model_type

    def objective(self, trial: optuna.Trial) -> float:
        df = self.load_data()
        # Statistical models often use the raw series
        n = len(df)
        train_series = df[TARGET_COLUMN].iloc[:int(n*0.7)].values
        val_series = df[TARGET_COLUMN].iloc[int(n*0.7):int(n*0.85)].values

        if self.model_type == "ARIMA":
            p = trial.suggest_int("p", 0, 5)
            d = trial.suggest_int("d", 0, 2)
            q = trial.suggest_int("q", 0, 5)
            try:
                model = ARIMA(train_series, order=(p, d, q))
                model_fit = model.fit()
                # Forecast 24 steps (assuming hourly or similar)
                forecast = model_fit.forecast(steps=len(val_series))
                loss = np.mean((forecast - val_series)**2)
                return float(loss)
            except:
                return 1e9

        elif self.model_type == "SARIMA":
            p = trial.suggest_int("p", 0, 3)
            d = trial.suggest_int("d", 0, 1)
            q = trial.suggest_int("q", 0, 3)
            P = trial.suggest_int("P", 0, 2)
            D = trial.suggest_int("D", 0, 1)
            Q = trial.suggest_int("Q", 0, 2)
            s = 24 # Daily seasonality
            try:
                model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=len(val_series))
                loss = np.mean((forecast - val_series)**2)
                return float(loss)
            except:
                return 1e9
        
        elif self.model_type == "VAR":
            # VAR uses all features
            train_multi = df[FEATURE_COLUMNS + [TARGET_COLUMN]].iloc[:int(n*0.7)].values
            val_multi = df[FEATURE_COLUMNS + [TARGET_COLUMN]].iloc[int(n*0.7):int(n*0.85)].values
            maxlags = trial.suggest_int("maxlags", 1, 48)
            try:
                model = VAR(train_multi)
                model_fit = model.fit(maxlags=maxlags)
                forecast = model_fit.forecast(train_multi[-model_fit.k_ar:], steps=len(val_multi))
                # Predict target only for loss
                loss = np.mean((forecast[:, -1] - val_multi[:, -1])**2)
                return float(loss)
            except:
                return 1e9
        
        return 1e9

    def train(self, config: dict):
        df = self.load_data()
        n = len(df)
        train_series = df[TARGET_COLUMN].iloc[:int(n*0.7)].values
        test_series = df[TARGET_COLUMN].iloc[int(n*0.85):].values

        if self.model_type == "ARIMA":
            model = ARIMA(train_series, order=(config["p"], config["d"], config["q"]))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test_series))
        
        elif self.model_type == "SARIMA":
            model = SARIMAX(train_series, order=(config["p"], config["d"], config["q"]),
                            seasonal_order=(config["P"], config["D"], config["Q"], 24))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(test_series))
            
        elif self.model_type == "VAR":
            train_multi = df[FEATURE_COLUMNS + [TARGET_COLUMN]].iloc[:int(n*0.7)].values
            test_multi = df[FEATURE_COLUMNS + [TARGET_COLUMN]].iloc[int(n*0.85):].values
            model = VAR(train_multi)
            model_fit = model.fit(maxlags=config["maxlags"])
            forecast_all = model_fit.forecast(train_multi[-model_fit.k_ar:], steps=len(test_multi))
            forecast = forecast_all[:, -1]
            test_series = test_multi[:, -1]

        joblib.dump(model_fit, self.output_dir / "best_model.joblib")
        
        metrics = {
            "rmse": float(np.sqrt(np.mean((forecast - test_series)**2))),
            "mae": float(np.mean(np.abs(forecast - test_series))),
            "r2": float(r2_score(test_series, forecast))
        }
        
        self.save_results(metrics, config)
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", choices=["ARIMA", "SARIMA", "VAR"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    trainer = StatisticalTrainer(args.city, args.model)
    best_params = trainer.optimize(n_trials=args.trials)
    trainer.train(best_params)
