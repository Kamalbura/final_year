"""
xgboost_model.py — XGBoost wrapper for AQI prediction.

Provides a sklearn-compatible interface with built-in support for
Optuna hyperparameter tuning, early stopping, and feature importance.
"""

import numpy as np
import torch
import xgboost as xgb
from typing import Dict, Optional, Tuple

from src.aq_forecast.config import ModelConfig


class XGBoostModel:
    """
    XGBoost gradient-boosted tree model wrapper.

    Uses tabular features (including engineered lags and rolling stats)
    instead of raw sliding windows. Supports GPU acceleration if available.
    """

    def __init__(self, config: ModelConfig, random_state: int = 42):
        self.config = config
        self.model = None
        self.random_state = random_state

        self.params = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate_gb,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda,
            "min_child_weight": 1,
            "gamma": 0.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            early_stopping_rounds: int = 50) -> "XGBoostModel":
        """
        Train the XGBoost model with optional early stopping.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training targets (n_samples,).
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
            early_stopping_rounds: Patience for early stopping.

        Returns:
            self
        """
        self.model = xgb.XGBRegressor(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """Return feature importance scores."""
        importance = self.model.feature_importances_
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        return dict(enumerate(importance))

    def update_params(self, new_params: Dict) -> None:
        """Update model parameters (used by Optuna)."""
        self.params.update(new_params)

    @staticmethod
    def get_optuna_search_space() -> Dict:
        """Define Optuna hyperparameter search space."""
        return {
            "n_estimators": ("int", 100, 2000),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("float_log", 0.005, 0.3),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.3, 1.0),
            "reg_alpha": ("float_log", 1e-3, 10.0),
            "reg_lambda": ("float_log", 1e-3, 10.0),
            "min_child_weight": ("int", 1, 20),
            "gamma": ("float", 0.0, 5.0),
        }
