"""
lightgbm_model.py — LightGBM wrapper for AQI prediction.

LightGBM uses histogram-based gradient boosting with leaf-wise tree
growth, offering faster training and often better performance than
XGBoost on large datasets. Especially effective with high-cardinality
categorical features.
"""

import numpy as np
import lightgbm as lgb
import torch
from typing import Dict, List, Optional

from src.aq_forecast.config import ModelConfig


class LightGBMModel:
    """
    LightGBM gradient-boosted tree model wrapper.

    Uses tabular features (including engineered lags and rolling stats).
    Leaf-wise growth provides deeper, more accurate trees.
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
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            early_stopping_rounds: int = 50) -> "LightGBMModel":
        """
        Train the LightGBM model with optional early stopping.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
            early_stopping_rounds: Patience.

        Returns:
            self
        """
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                                   verbose=False)
            )
        callbacks.append(lgb.log_evaluation(period=0))

        self.model = lgb.LGBMRegressor(**self.params)

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if eval_set else None,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
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
        """Define Optuna hyperparameter search space.

        Note:
            Constraint: num_leaves must be < 2^max_depth to prevent
            overfitting. The Optuna objective should enforce this via
            ``num_leaves = min(num_leaves, 2**max_depth - 1)``.
        """
        return {
            "n_estimators": ("int", 100, 2000),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("float_log", 0.005, 0.3),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.3, 1.0),
            "reg_alpha": ("float_log", 1e-3, 10.0),
            "reg_lambda": ("float_log", 1e-3, 10.0),
            "num_leaves": ("int", 15, 127),
            "min_child_samples": ("int", 5, 100),
            "min_child_weight": ("float_log", 1e-4, 10.0),
            "path_smooth": ("float", 0.0, 10.0),
            "boosting_type": ("categorical", ["gbdt", "dart"]),
        }
