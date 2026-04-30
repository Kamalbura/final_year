from __future__ import annotations
import os
import sys
import json
import time
import logging
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Best Practice: Structured Logging
LOG_DIR = Path("logs/training")
LOG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
TARGET_COLUMN = "us_aqi"

class AQBaseTrainer:
    def __init__(
        self,
        city: str,
        model_name: str,
        data_dir: str = "data/kaggle_dataset",
        output_dir: str = "outputs/individual_trainers",
        seed: int = 42
    ):
        self.city = city
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / model_name / city
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self._set_seed()
        self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized {model_name} trainer for {city} on {self.device}")

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _setup_logging(self):
        log_file = LOG_DIR / f"{self.model_name}_{self.city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(f"{self.model_name}_{self.city}")

    def load_data(self) -> pd.DataFrame:
        path = self.data_dir / f"clean_{self.city}_aq_1y.csv"
        if not path.exists():
            path = self.data_dir / f"{self.city}_aq_1y.csv"
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
            
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def prepare_data(self, df: pd.DataFrame, lookback: int, horizon: int) -> Dict[str, Any]:
        n = len(df)
        train_df = df.iloc[:int(n*0.7)].copy()
        val_df = df.iloc[int(n*0.7):int(n*0.85)].copy()
        test_df = df.iloc[int(n*0.85):].copy()

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x_train_s = x_scaler.fit_transform(train_df[FEATURE_COLUMNS])
        y_train_s = y_scaler.fit_transform(train_df[[TARGET_COLUMN]]).flatten()

        x_val_s = x_scaler.transform(val_df[FEATURE_COLUMNS])
        y_val_s = y_scaler.transform(val_df[[TARGET_COLUMN]]).flatten()

        x_test_s = x_scaler.transform(test_df[FEATURE_COLUMNS])
        y_test_s = y_scaler.transform(test_df[[TARGET_COLUMN]]).flatten()

        def window_xy(x_arr, y_arr, lb, hz):
            x_l, y_l = [], []
            for i in range(len(x_arr) - lb - hz + 1):
                x_l.append(x_arr[i : i + lb])
                y_l.append(y_arr[i + lb : i + lb + hz])
            return np.array(x_l, dtype=np.float32), np.array(y_l, dtype=np.float32)

        X_train, y_train = window_xy(x_train_s, y_train_s, lookback, horizon)
        X_val, y_val = window_xy(x_val_s, y_val_s, lookback, horizon)
        X_test, y_test = window_xy(x_test_s, y_test_s, lookback, horizon)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "x_scaler": x_scaler, "y_scaler": y_scaler
        }

    def train(self, config: Dict[str, Any]):
        raise NotImplementedError("Subclasses must implement train()")

    def optimize(self, n_trials: int = 20):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)
        self.logger.info(f"Best trial: {study.best_trial.value}")
        self.logger.info(f"Best params: {study.best_params}")
        return study.best_params

    def objective(self, trial: optuna.Trial) -> float:
        raise NotImplementedError("Subclasses must implement objective()")

    def save_results(self, metrics: Dict[str, float], config: Dict[str, Any]):
        res_path = self.output_dir / "results.json"
        payload = {
            "city": self.city,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "config": config
        }
        with open(res_path, 'w') as f:
            json.dump(payload, f, indent=4)
        self.logger.info(f"Results saved to {res_path}")
