""" Metrics for Task 2: Time Series Forecasting. """
# src/task2_metrics.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _to_float_array(x) -> np.ndarray:
    """Converts input to a numpy float array."""
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    MAPE in percent. Adds eps to avoid division by zero.
    """
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def all_metrics(y_true, y_pred) -> Dict[str, float]:
    """Computes core metrics and returns a dictionary."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
    }


def error_stats(y_true, y_pred) -> Dict[str, float]:
    """
    Richer error diagnostics used in your business writeup:
    bias, dispersion, and quantiles of error.
    Error defined as: (y_pred - y_true)
    """
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)

    err = (y_pred - y_true).astype(float)
    abs_err = np.abs(err)

    def q(p: float) -> float:
        return float(np.quantile(err, p))

    return {
        "n": int(err.shape[0]),
        "mean_error_bias": float(err.mean()),
        "median_error": float(np.median(err)),
        "std_error": float(err.std(ddof=1)) if err.shape[0] > 1 else 0.0,
        "median_abs_error": float(np.median(abs_err)),
        "q05_error": q(0.05),
        "q25_error": q(0.25),
        "q50_error": q(0.50),
        "q75_error": q(0.75),
        "q95_error": q(0.95),
    }


def fit_bias_offset(y_true, y_pred) -> float:
    """
    Fit a simple bias offset on validation:
      offset = mean(y_true - y_pred)
    Then calibrated_pred = y_pred + offset
    """
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)
    return float(np.mean(y_true - y_pred))


def apply_bias_offset(y_pred, offset: float):
    """Apply bias offset calibration."""
    y_pred = _to_float_array(y_pred)
    return y_pred + float(offset)


def win_rate(y_true, pred_a, pred_b) -> Tuple[float, float, float]:
    """
    Win-rate by absolute error.
    Returns: (wins_a_pct, wins_b_pct, ties_pct)
    """
    y_true = _to_float_array(y_true)
    pred_a = _to_float_array(pred_a)
    pred_b = _to_float_array(pred_b)

    abs_a = np.abs(pred_a - y_true)
    abs_b = np.abs(pred_b - y_true)

    wins_a = float(np.mean(abs_a < abs_b) * 100.0)
    wins_b = float(np.mean(abs_b < abs_a) * 100.0)
    ties = float(np.mean(abs_a == abs_b) * 100.0)
    return wins_a, wins_b, ties
