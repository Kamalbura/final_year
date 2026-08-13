"""
preprocessing.py — Data cleaning, feature engineering, and scaling.

Handles missing values, temporal encoding, AQI computation,
and train/val/test splits for the air quality pipeline.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Optional

from src.aq_forecast.config import (
    POLLUTANT_FEATURES, METEO_FEATURES, TEMPORAL_FEATURES,
    TARGET, DataConfig
)
from src.aq_forecast.data.download import compute_aqi, get_aqi_category

logger = logging.getLogger(__name__)


def impute_missing_values(df: pd.DataFrame,
                          method: str = "linear") -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategies:
        - "linear": Linear interpolation + forward/backward fill for edges.
        - "forward_fill": Forward fill, then backward fill.
        - "knn": Simple column-mean fill (lightweight KNN proxy).

    Args:
        df: Input DataFrame with potential NaN values.
        method: Imputation strategy.

    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    missing_before = df[numeric_cols].isnull().sum().sum()
    logger.info(f"Missing values before imputation: {missing_before}")

    if method == "linear":
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )
    elif method == "forward_fill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif method == "knn":
        # Lightweight: fill with column means
        for col in numeric_cols:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    # Guard: drop columns that are still entirely NaN
    still_all_nan = [c for c in numeric_cols if df[c].isnull().all()]
    if still_all_nan:
        logger.warning("Dropping %d all-NaN columns: %s", len(still_all_nan), still_all_nan)
        df = df.drop(columns=still_all_nan)

    # Clip pollutant columns to non-negative (interpolation can produce negatives)
    from src.aq_forecast.config import POLLUTANT_FEATURES
    clip_cols = [c for c in POLLUTANT_FEATURES if c in df.columns]
    if clip_cols:
        df[clip_cols] = df[clip_cols].clip(lower=0)

    remaining_numeric = df.select_dtypes(include=[np.number]).columns
    missing_after = df[remaining_numeric].isnull().sum().sum()
    logger.info(f"Missing values after imputation: {missing_after}")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer cyclical and categorical temporal features from the datetime index.

    Creates:
        - hour, day_of_week, day_of_month, month (raw)
        - Sine/Cosine encodings for hour, day_of_week, month (cyclical)
        - is_weekend binary flag

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        DataFrame with temporal features added.
    """
    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"] = idx.month

    # Cyclical encoding — prevents discontinuities (e.g. hour 23 → 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    logger.info("Added temporal features (cyclical + categorical)")
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "AQI",
                     lags: List[int] = None) -> pd.DataFrame:
    """
    Add lag features for the target variable.

    Useful for gradient-boosting models (XGBoost, LightGBM) that don't
    inherently capture sequential dependencies.

    Args:
        df: Input DataFrame.
        target_col: Column to create lags for.
        lags: List of lag periods (e.g., [1, 3, 6, 12, 24]).

    Returns:
        DataFrame with lag columns added.
    """
    df = df.copy()
    if lags is None:
        lags = [1, 3, 6, 12, 24, 48]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24]:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )

    logger.info(f"Added lag features (lags={lags}) and rolling stats")
    return df


def ensure_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataset has an AQI column. Compute it if missing.

    Args:
        df: DataFrame with pollutant columns.

    Returns:
        DataFrame with AQI and AQI_Category columns.
    """
    df = df.copy()

    if "AQI" not in df.columns:
        logger.info("AQI column not found — computing from pollutant sub-indices")
        df["AQI"] = df.apply(compute_aqi, axis=1)
    else:
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    # Add categorical label
    df["AQI_Category"] = df["AQI"].apply(get_aqi_category)

    # Drop rows where AQI could not be computed
    valid_mask = df["AQI"].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} rows with invalid AQI")
        df = df[valid_mask]

    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the data to a consistent hourly frequency.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        Hourly-resampled DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_hourly = df[numeric_cols].resample("1h").mean()
    logger.info(f"Resampled to hourly: {len(df_hourly)} rows")
    return df_hourly


def build_feature_matrix(df: pd.DataFrame,
                         config: DataConfig) -> Tuple[List[str], pd.DataFrame]:
    """
    Assemble the final feature matrix based on available columns.

    Args:
        df: Preprocessed DataFrame.
        config: DataConfig with feature flags.

    Returns:
        Tuple of (feature_names, feature DataFrame).
    """
    features = []

    # Pollutant features (exclude AQI itself)
    for col in POLLUTANT_FEATURES:
        if col in df.columns:
            features.append(col)

    # Meteorological features
    if config.use_meteo:
        for col in METEO_FEATURES:
            if col in df.columns:
                features.append(col)

    # Temporal features
    if config.use_temporal:
        for col in TEMPORAL_FEATURES:
            if col in df.columns:
                features.append(col)

    # Lag and rolling features
    lag_cols = [c for c in df.columns
                if c.startswith("AQI_lag_") or c.startswith("AQI_rolling_")]
    features.extend(lag_cols)

    logger.info(f"Feature matrix: {len(features)} features selected")
    return features, df[features + [TARGET]].copy()


def scale_data(df: pd.DataFrame, feature_cols: List[str],
               scaler_type: str = "standard",
               existing_scaler: Optional[Dict] = None
               ) -> Tuple[pd.DataFrame, Dict]:
    """
    Scale features and target using StandardScaler or MinMaxScaler.

    Args:
        df: Input DataFrame.
        feature_cols: Columns to scale as features.
        scaler_type: "standard" or "minmax".
        existing_scaler: Pre-fitted scalers dict for transform-only mode.

    Returns:
        Tuple of (scaled DataFrame, scalers dict).
    """
    df = df.copy()
    ScalerClass = StandardScaler if scaler_type == "standard" else MinMaxScaler

    if existing_scaler is None:
        # Fit new scalers
        feature_scaler = ScalerClass()
        target_scaler = ScalerClass()

        df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
        df[[TARGET]] = target_scaler.fit_transform(df[[TARGET]])

        scalers = {"feature": feature_scaler, "target": target_scaler}
    else:
        # Transform using existing scalers
        df[feature_cols] = existing_scaler["feature"].transform(df[feature_cols])
        df[[TARGET]] = existing_scaler["target"].transform(df[[TARGET]])
        scalers = existing_scaler

    return df, scalers


def chronological_split(df: pd.DataFrame,
                        config: DataConfig
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets.

    No shuffling — preserves temporal order to prevent data leakage.

    Args:
        df: Full preprocessed DataFrame.
        config: DataConfig with split ratios.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Chronological split — Train: {len(train_df)}, "
                f"Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def run_preprocessing_pipeline(df: pd.DataFrame,
                               config: DataConfig = None
                               ) -> Dict:
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps:
        1. Ensure AQI column exists
        2. Resample to hourly frequency
        3. Impute missing values
        4. Add temporal features
        5. Add lag features (for GB models)
        6. Build feature matrix
        7. Chronological train/val/test split
        8. Scale data

    Args:
        df: Raw DataFrame loaded from CSVs.
        config: DataConfig instance.

    Returns:
        Dictionary containing all pipeline outputs:
            - train/val/test DataFrames (scaled)
            - feature_names
            - scalers
            - raw splits (unscaled)
    """
    if config is None:
        config = DataConfig()

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info("=" * 60)

    # Step 1: Ensure AQI column
    df = ensure_aqi_column(df)

    # Step 2: Resample to hourly
    df = resample_to_hourly(df)

    # Step 3: Impute missing values
    df = impute_missing_values(df, method=config.imputation_method)

    # Step 4: Temporal features
    df = add_temporal_features(df)

    # Step 5: Lag features
    df = add_lag_features(df)

    # Drop rows with NaN from lag creation
    df = df.dropna()

    # Step 6: Build feature matrix
    feature_names, df_features = build_feature_matrix(df, config)

    # Step 7: Chronological split
    train_df, val_df, test_df = chronological_split(df_features, config)

    # Step 8: Scale data — fit on train, transform val and test
    train_scaled, scalers = scale_data(
        train_df, feature_names, config.scaler_type
    )
    val_scaled, _ = scale_data(
        val_df, feature_names, config.scaler_type, existing_scaler=scalers
    )
    test_scaled, _ = scale_data(
        test_df, feature_names, config.scaler_type, existing_scaler=scalers
    )

    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Train: {len(train_scaled)}, Val: {len(val_scaled)}, "
                f"Test: {len(test_scaled)}")

    return {
        "train": train_scaled,
        "val": val_scaled,
        "test": test_scaled,
        "train_raw": train_df,
        "val_raw": val_df,
        "test_raw": test_df,
        "feature_names": feature_names,
        "scalers": scalers,
        "config": config,
    }
