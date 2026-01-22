""" Metrics for Task 2: Time Series Forecasting. """
# src/task2_metrics.py
from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    """ Mean Absolute Error. """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """ Root Mean Squared Error. """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    MAPE in percent. Adds eps to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def all_metrics(y_true, y_pred) -> dict:
    """ Computes all metrics and returns a dictionary. """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
    }
