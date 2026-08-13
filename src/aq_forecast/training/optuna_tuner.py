"""
optuna_tuner.py — Optuna hyperparameter optimization for all models.

Provides a unified interface to tune both deep learning (PyTorch)
and gradient-boosting (XGBoost/LightGBM) models using Optuna's
TPE sampler with median pruning.
"""

import os
import logging
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

from src.aq_forecast.config import (
    ModelConfig, TrainConfig, OptunaConfig, DataConfig,
    DEVICE, CHECKPOINT_DIR, DL_MODELS, GB_MODELS
)
from src.aq_forecast.models.model_factory import build_model
from src.aq_forecast.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _suggest_param(trial: optuna.Trial, name: str,
                   spec: tuple) -> Any:
    """
    Suggest a hyperparameter value from an Optuna trial.

    Args:
        trial: Optuna trial object.
        name: Parameter name.
        spec: Tuple of (type, *args) defining the search space.
    """
    param_type = spec[0]
    if param_type == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif param_type == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif param_type == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Unknown param type: {param_type}")


class OptunaTuner:
    """
    Unified Optuna hyperparameter tuner for DL and GB models.

    Handles:
      - DL models: tunes hidden_dim, num_layers, dropout, lr, batch_size
      - GB models: tunes n_estimators, max_depth, lr, regularization, etc.
      - Pruning: MedianPruner or HyperbandPruner for DL models
      - Best checkpoint saving and parameter logging
    """

    def __init__(self, optuna_config: OptunaConfig = None):
        self.optuna_config = optuna_config or OptunaConfig()
        self.best_params = {}  # model_name → best params dict

    def _create_study(self, model_name: str) -> optuna.Study:
        """Create an Optuna study with configured sampler and pruner."""
        cfg = self.optuna_config

        # Sampler
        if cfg.sampler == "tpe":
            sampler = TPESampler(seed=42)
        else:
            sampler = CmaEsSampler(seed=42)

        # Pruner
        if cfg.pruner == "median":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        else:
            pruner = HyperbandPruner(min_resource=5, max_resource=100)

        study = optuna.create_study(
            study_name=f"{cfg.study_name}_{model_name}",
            direction=cfg.direction,
            sampler=sampler,
            pruner=pruner,
        )
        return study

    def tune_dl_model(self, model_name: str,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      data_config: DataConfig,
                      input_dim: int) -> Dict:
        """
        Tune a deep learning model's hyperparameters.

        Search space:
            - hidden_dim: [32, 64, 128, 256]
            - num_layers: [1, 2, 3, 4]
            - dropout: [0.1, 0.5]
            - learning_rate: [1e-5, 1e-2] (log scale)
            - batch_size: [32, 64, 128]
            - scheduler: ["cosine", "plateau"]

        Args:
            model_name: Name of the DL model to tune.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            data_config: DataConfig with lookback/horizon.
            input_dim: Number of input features.

        Returns:
            Dict of best hyperparameters.
        """
        def objective(trial: optuna.Trial) -> float:
            # ── Suggest hyperparameters ──
            hidden_dim = trial.suggest_categorical(
                "hidden_dim", [32, 64, 128, 256]
            )
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            )
            scheduler = trial.suggest_categorical(
                "scheduler", ["cosine", "plateau"]
            )

            # TCN-specific
            if model_name == "TCN":
                n_channels = trial.suggest_categorical(
                    "tcn_channels", [32, 64, 128]
                )
                n_blocks = trial.suggest_int("tcn_blocks", 3, 6)
                kernel_size = trial.suggest_categorical(
                    "tcn_kernel_size", [3, 5, 7]
                )
                tcn_num_channels = [n_channels] * n_blocks
            else:
                tcn_num_channels = [64, 64, 64, 64]
                kernel_size = 3

            # Transformer-specific
            if model_name == "Transformer":
                d_model = trial.suggest_categorical(
                    "d_model", [64, 128, 256]
                )
                nhead = trial.suggest_categorical("nhead", [4, 8])
                n_enc_layers = trial.suggest_int("n_enc_layers", 2, 6)
                dim_ff = trial.suggest_categorical(
                    "dim_feedforward", [128, 256, 512]
                )
            else:
                d_model, nhead, n_enc_layers, dim_ff = 128, 8, 3, 256

            # ── Build model config ──
            model_config = ModelConfig(
                model_name=model_name,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                tcn_num_channels=tcn_num_channels,
                tcn_kernel_size=kernel_size,
                transformer_d_model=d_model,
                transformer_nhead=nhead,
                transformer_num_encoder_layers=n_enc_layers,
                transformer_dim_feedforward=dim_ff,
            )

            train_config = TrainConfig(
                epochs=50,  # Reduced epochs for HPO
                learning_rate=lr,
                weight_decay=weight_decay,
                patience=10,
                scheduler=scheduler,
            )

            # ── Train model ──
            model = build_model(
                model_name, model_config,
                horizon=data_config.horizon
            )
            trainer = Trainer(model, train_config, model_name=model_name)

            # Abbreviated training with pruning
            for epoch in range(1, train_config.epochs + 1):
                train_loss = trainer._train_epoch(train_loader)
                val_loss = trainer._validate(val_loader)

                # Report to Optuna for pruning
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping
                if trainer.early_stopping.step(val_loss):
                    break

            return trainer.early_stopping.best_loss

        # ── Run study ──
        study = self._create_study(model_name)
        logger.info(f"Starting Optuna HPO for {model_name} "
                    f"({self.optuna_config.n_trials} trials)")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout,
        )

        best = study.best_params
        self.best_params[model_name] = best
        logger.info(f"Best params for {model_name}: {best}")
        logger.info(f"Best val loss: {study.best_value:.6f}")

        return best

    def tune_gb_model(self, model_name: str,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Tune an XGBoost or LightGBM model's hyperparameters.

        Args:
            model_name: "XGBoost" or "LightGBM".
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Returns:
            Dict of best hyperparameters.
        """
        from src.aq_forecast.models.xgboost_model import XGBoostModel
        from src.aq_forecast.models.lightgbm_model import LightGBMModel

        ModelClass = XGBoostModel if model_name == "XGBoost" else LightGBMModel
        search_space = ModelClass.get_optuna_search_space()

        def objective(trial: optuna.Trial) -> float:
            # Suggest all params from the model's search space
            params = {}
            for param_name, spec in search_space.items():
                params[param_name] = _suggest_param(trial, param_name, spec)

            # LightGBM constraint: num_leaves must be < 2^max_depth
            if "num_leaves" in params and "max_depth" in params:
                max_leaves = 2 ** params["max_depth"] - 1
                params["num_leaves"] = min(params["num_leaves"], max_leaves)

            config = ModelConfig()
            model = ModelClass(config=config)
            model.update_params(params)
            model.fit(X_train, y_train, X_val, y_val,
                      early_stopping_rounds=50)

            predictions = model.predict(X_val)
            rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
            return rmse

        # ── Run study ──
        study = self._create_study(model_name)
        logger.info(f"Starting Optuna HPO for {model_name} "
                    f"({self.optuna_config.n_trials} trials)")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout,
        )

        best = study.best_params
        self.best_params[model_name] = best
        logger.info(f"Best params for {model_name}: {best}")
        logger.info(f"Best val RMSE: {study.best_value:.6f}")

        return best

    def get_best_model_config(self, model_name: str,
                              input_dim: int) -> ModelConfig:
        """
        Build a ModelConfig from the best Optuna parameters.

        Args:
            model_name: Model name.
            input_dim: Number of input features.

        Returns:
            ModelConfig with optimized hyperparameters.
        """
        params = self.best_params.get(model_name, {})

        config = ModelConfig(
            model_name=model_name,
            input_dim=input_dim,
            hidden_dim=params.get("hidden_dim", 128),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.2),
        )

        # TCN specifics
        if model_name == "TCN":
            n_ch = params.get("tcn_channels", 64)
            n_blocks = params.get("tcn_blocks", 4)
            config.tcn_num_channels = [n_ch] * n_blocks
            config.tcn_kernel_size = params.get("tcn_kernel_size", 3)

        # Transformer specifics
        if model_name == "Transformer":
            config.transformer_d_model = params.get("d_model", 128)
            config.transformer_nhead = params.get("nhead", 8)
            config.transformer_num_encoder_layers = params.get("n_enc_layers", 3)
            config.transformer_dim_feedforward = params.get("dim_feedforward", 256)

        return config

    def get_best_train_config(self, model_name: str) -> TrainConfig:
        """Build a TrainConfig from the best Optuna parameters."""
        params = self.best_params.get(model_name, {})
        return TrainConfig(
            learning_rate=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 1e-5),
            scheduler=params.get("scheduler", "cosine"),
        )
