""" ARIMA modeling for Task 2: Time Series Forecasting. """
# src/task2_arima.py
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src import config
from src.task2_metrics import all_metrics


def _ensure_dir(path: str) -> None:
    """ Creates directory if it does not exist. """
    os.makedirs(path, exist_ok=True)


def fit_auto_arima(y_train: pd.Series):
    """
    Returns fitted pmdarima model.
    """
    try:
        from pmdarima import auto_arima
    except ImportError as e:
        raise ImportError(
            "pmdarima is required for auto_arima. Install: pip install pmdarima") from e

    model = auto_arima(
        y_train,
        seasonal=config.ARIMA_SEASONAL,
        m=config.ARIMA_M,
        start_p=config.ARIMA_START_P,
        start_q=config.ARIMA_START_Q,
        max_p=config.ARIMA_MAX_P,
        max_q=config.ARIMA_MAX_Q,
        max_d=config.ARIMA_MAX_D,
        start_P=config.ARIMA_START_P_SEASONAL,
        start_Q=config.ARIMA_START_Q_SEASONAL,
        max_P=config.ARIMA_MAX_P_SEASONAL,
        max_Q=config.ARIMA_MAX_Q_SEASONAL,
        max_D=config.ARIMA_MAX_D_SEASONAL,
        stepwise=config.ARIMA_STEPWISE,
        trace=config.ARIMA_TRACE,
        suppress_warnings=config.ARIMA_SUPPRESS_WARNINGS,
        error_action=config.ARIMA_ERROR_ACTION,
        information_criterion="aic",
        n_fits=50,
    )
    return model


def arima_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = config.TASK2_TARGET_COL,
    date_col: str = config.TASK2_DATE_COL,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fits ARIMA on train target, forecasts for len(test).
    Returns:
      forecast_df: columns [date, y_true, y_pred]
      params: dict of chosen orders + model info
    """
    y_train = train_df[target_col].astype(float)
    y_test = test_df[target_col].astype(float)

    model = fit_auto_arima(y_train)
    n_periods = len(test_df)

    preds = model.predict(n_periods=n_periods)
    preds = np.asarray(preds, dtype=float)

    forecast_df = pd.DataFrame({
        date_col: pd.to_datetime(test_df[date_col]).values,
        "y_true": y_test.values,
        "y_pred": preds,
    })

    order = getattr(model, "order", None)
    seasonal_order = getattr(model, "seasonal_order", None)

    params = {
        "asset": config.TASK2_ASSET,
        "target_col": target_col,
        "seasonal": bool(config.ARIMA_SEASONAL),
        "m": int(config.ARIMA_M),
        "order": list(order) if order is not None else None,
        "seasonal_order": list(seasonal_order) if seasonal_order is not None else None,
        "aic": float(getattr(model, "aic", lambda: np.nan)()),
        "bic": float(getattr(model, "bic", lambda: np.nan)()),
    }

    params["metrics"] = all_metrics(
        forecast_df["y_true"], forecast_df["y_pred"])
    return forecast_df, params


def save_arima_outputs(forecast_df: pd.DataFrame, params: Dict, save_model=None) -> None:
    """ Saves forecast CSV, params JSON, and optionally the pmdarima model. """
    _ensure_dir(config.TASK2_FORECASTS_DIR)
    _ensure_dir(config.TASK2_METRICS_DIR)
    _ensure_dir(config.TASK2_MODELS_DIR)

    forecast_df.to_csv(config.TASK2_ARIMA_FORECAST_PATH, index=False)

    with open(config.TASK2_ARIMA_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Optional: save pmdarima model
    if save_model is not None:
        try:
            import joblib
            joblib.dump(save_model, config.TASK2_ARIMA_MODEL_PATH)
        except Exception:
            # keep non-fatal
            pass
