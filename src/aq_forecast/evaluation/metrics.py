"""
metrics.py — Evaluation metrics for AQI prediction.

Computes regression metrics (RMSE, MAE, R²) and classification
metrics (F1-score for AQI category prediction).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, classification_report, confusion_matrix
)
from typing import Dict, Optional

from src.aq_forecast.config import AQI_CATEGORIES
from src.aq_forecast.aqi import get_aqi_category


def compute_regression_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute core regression metrics for AQI prediction.

    Args:
        y_true: Ground truth AQI values.
        y_pred: Predicted AQI values.

    Returns:
        Dict with RMSE, MAE, R², MAPE.
    """
    # Flatten for multi-horizon predictions
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute F1-score by mapping continuous AQI to categories.

    Maps predicted and true AQI values to categorical labels
    (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)
    and computes weighted and macro F1-scores.

    Args:
        y_true: Ground truth AQI values (continuous).
        y_pred: Predicted AQI values (continuous).

    Returns:
        Dict with F1 scores (weighted, macro, per-class).
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Map to categories
    true_cats = [get_aqi_category(v) for v in y_true]
    pred_cats = [get_aqi_category(v) for v in y_pred]

    # Get all unique labels
    all_labels = sorted(set(true_cats + pred_cats))
    # Remove "Unknown" if present
    all_labels = [l for l in all_labels if l != "Unknown"]

    if not all_labels:
        return {"F1_weighted": 0.0, "F1_macro": 0.0}

    f1_w = f1_score(true_cats, pred_cats, labels=all_labels,
                    average="weighted", zero_division=0)
    f1_m = f1_score(true_cats, pred_cats, labels=all_labels,
                    average="macro", zero_division=0)

    # Per-class F1
    per_class = {}
    f1_per = f1_score(true_cats, pred_cats, labels=all_labels,
                      average=None, zero_division=0)
    for label, score in zip(all_labels, f1_per):
        per_class[f"F1_{label}"] = round(score, 4)

    result = {
        "F1_weighted": round(f1_w, 4),
        "F1_macro": round(f1_m, 4),
    }
    result.update(per_class)
    return result


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all regression and classification metrics.

    Args:
        y_true: Ground truth AQI values.
        y_pred: Predicted AQI values.

    Returns:
        Combined dict of all metrics.
    """
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    cls_metrics = compute_classification_metrics(y_true, y_pred)

    metrics = {}
    metrics.update(reg_metrics)
    metrics.update(cls_metrics)
    return metrics


def get_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray) -> pd.DataFrame:
    """
    Generate a confusion matrix for AQI category predictions.

    Returns:
        DataFrame with confusion matrix (rows = true, cols = predicted).
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_cats = [get_aqi_category(v) for v in y_true]
    pred_cats = [get_aqi_category(v) for v in y_pred]

    labels = list(AQI_CATEGORIES.keys())
    cm = confusion_matrix(true_cats, pred_cats, labels=labels)

    return pd.DataFrame(cm, index=labels, columns=labels)
