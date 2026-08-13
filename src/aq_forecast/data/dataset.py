"""
dataset.py — PyTorch Dataset and DataLoader factories for time-series data.

Creates sliding-window samples from preprocessed DataFrames.
Supports both sequence-to-one and sequence-to-sequence forecasting.
Also provides flat feature matrices for gradient-boosting models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List

from src.aq_forecast.config import TARGET, DataConfig


class AQITimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for deep learning models.

    Each sample consists of:
        X: (lookback, num_features) — past observations
        y: (horizon,)               — future AQI values to predict

    If horizon == 1, y is a scalar (single-step forecast).
    """

    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 lookback: int = 72, horizon: int = 24):
        """
        Args:
            df: Scaled DataFrame with features and target.
            feature_cols: List of feature column names.
            lookback: Number of past timesteps for input.
            horizon: Number of future timesteps to predict.
        """
        self.lookback = lookback
        self.horizon = horizon

        # Extract numpy arrays for fast indexing
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[TARGET].values.astype(np.float32)

        # Valid indices: we need lookback past + horizon future
        self.n_samples = len(df) - lookback - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data for lookback={lookback}, horizon={horizon}. "
                f"Got {len(df)} rows."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input window: [idx, idx + lookback)
        x_start = idx
        x_end = idx + self.lookback

        # Target window: [idx + lookback, idx + lookback + horizon)
        y_start = x_end
        y_end = y_start + self.horizon

        X = torch.tensor(self.features[x_start:x_end], dtype=torch.float32)
        y = torch.tensor(self.targets[y_start:y_end], dtype=torch.float32)

        # For single-step prediction, squeeze to scalar
        if self.horizon == 1:
            y = y.squeeze()

        return X, y


def create_dataloaders(pipeline_output: Dict,
                       config: DataConfig = None,
                       batch_size: int = 64,
                       num_workers: int = 0
                       ) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from the preprocessing pipeline output.

    Args:
        pipeline_output: Dict from run_preprocessing_pipeline().
        config: DataConfig with lookback/horizon.
        batch_size: Batch size for training.
        num_workers: DataLoader workers (0 for Windows compatibility).

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    if config is None:
        config = pipeline_output.get("config", DataConfig())

    feature_names = pipeline_output["feature_names"]

    loaders = {}
    for split in ["train", "val", "test"]:
        df = pipeline_output[split]
        dataset = AQITimeSeriesDataset(
            df=df,
            feature_cols=feature_names,
            lookback=config.lookback,
            horizon=config.horizon,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle time-series data
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders


def prepare_gb_data(pipeline_output: Dict,
                    config: DataConfig = None
                    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare flat feature matrices for gradient-boosting models.

    XGBoost and LightGBM don't use sliding windows — they take
    tabular input with lag features already engineered.

    Args:
        pipeline_output: Dict from run_preprocessing_pipeline().
        config: DataConfig instance.

    Returns:
        Dict with 'train', 'val', 'test' tuples of (X, y) numpy arrays.
    """
    feature_names = pipeline_output["feature_names"]

    result = {}
    for split in ["train", "val", "test"]:
        df = pipeline_output[split]
        X = df[feature_names].values.astype(np.float32)
        y = df[TARGET].values.astype(np.float32)
        result[split] = (X, y)

    return result
