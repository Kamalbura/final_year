from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _to_numpy(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(_to_numpy(y_true), _to_numpy(y_pred))))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(_to_numpy(y_true), _to_numpy(y_pred)))


def r2(y_true, y_pred) -> float:
    return float(r2_score(_to_numpy(y_true), _to_numpy(y_pred)))


def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    if seasonality < 1:
        raise ValueError("seasonality must be at least 1.")
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = _to_numpy(y_pred).reshape(-1)
    y_train = _to_numpy(y_train).reshape(-1)
    if len(y_train) <= seasonality:
        raise ValueError("y_train must be longer than the seasonality for MASE.")
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def prediction_interval_coverage(y_true, lower, upper) -> float:
    y_true = _to_numpy(y_true)
    lower = _to_numpy(lower)
    upper = _to_numpy(upper)
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("y_true, lower, and upper must have the same shape.")
    coverage = np.logical_and(y_true >= lower, y_true <= upper)
    return float(np.mean(coverage))
