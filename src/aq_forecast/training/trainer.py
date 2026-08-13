"""
trainer.py — Training loop for deep learning models.

Handles epoch-level training, validation, early stopping,
checkpointing, learning rate scheduling, and gradient clipping.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from src.aq_forecast.config import (
    TrainConfig, DEVICE, CHECKPOINT_DIR
)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    PyTorch training loop for DL time-series models.

    Features:
        - Mixed-precision training (AMP) when GPU is available
        - Gradient clipping for stable training
        - Learning rate scheduling (cosine, step, plateau)
        - Early stopping with patience
        - Model checkpointing (best + periodic)
        - Training history logging
    """

    def __init__(self, model: nn.Module, config: TrainConfig,
                 model_name: str = "model"):
        self.model = model.to(DEVICE)
        self.config = config
        self.model_name = model_name

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function — Huber loss is robust to outliers in AQI data
        self.criterion = nn.HuberLoss(delta=1.0)

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        # AMP scaler for mixed-precision training
        self.scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def _build_scheduler(self):
        """Create learning rate scheduler based on config."""
        cfg = self.config
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=1e-7
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.scheduler_step_size,
                gamma=cfg.scheduler_gamma
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=cfg.scheduler_gamma,
                patience=cfg.patience // 2, min_lr=1e-7
            )
        return None

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            self.optimizer.zero_grad()

            # Mixed-precision forward pass
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Training history dict.
        """
        logger.info(f"Training {self.model_name} on {DEVICE} "
                    f"for up to {self.config.epochs} epochs")

        best_val_loss = float("inf")
        best_model_path = os.path.join(
            CHECKPOINT_DIR, f"{self.model_name}_best.pt"
        )

        for epoch in range(1, self.config.epochs + 1):
            t_start = time.time()

            # Train & validate
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            elapsed = time.time() - t_start

            # Logging
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                }, best_model_path)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopping.step(val_loss):
                logger.info(f"  Early stopping at epoch {epoch} "
                            f"(best val loss: {best_val_loss:.6f})")
                break

        # Load best weights
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=DEVICE,
                                    weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"  Loaded best model from epoch {checkpoint['epoch']}")

        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on a DataLoader.

        Returns:
            Tuple of (predictions, ground_truth) numpy arrays.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            preds = self.model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        return predictions, targets
