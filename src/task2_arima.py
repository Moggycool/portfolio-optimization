""" ARIMA modeling for Task 2: Time Series Forecasting. """
# src/task2_arima.py
from __future__ import annotations

import json
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from src import config
from src.task2_metrics import all_metrics


def _ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def fit_auto_arima(y_train: pd.Series):
    """
    Returns fitted pmdarima model.
    """
    try:
        from pmdarima import auto_arima
    except ImportError as e:
        raise ImportError(
            "pmdarima is required for auto_arima. Install: pip install pmdarima"
        ) from e

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


def _safe_model_info(model) -> Dict:
    """Extract minimal model info without crashing."""
    order = getattr(model, "order", None)
    seasonal_order = getattr(model, "seasonal_order", None)

    def _call_or_nan(attr_name: str) -> float:
        attr = getattr(model, attr_name, None)
        try:
            return float(attr()) if callable(attr) else float(attr)
        except Exception:
            return float("nan")

    return {
        "seasonal": bool(config.ARIMA_SEASONAL),
        "m": int(config.ARIMA_M),
        "order": list(order) if order is not None else None,
        "seasonal_order": list(seasonal_order) if seasonal_order is not None else None,
        "aic": _call_or_nan("aic"),
        "bic": _call_or_nan("bic"),
    }


def walk_forward_arima_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    refit_each_step: bool = True,   # kept for API compatibility; we’ll interpret below
    refit_every: int = 1,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward 1-step ahead ARIMA forecasting using pmdarima's update().

    If refit_each_step=True: refit every step (slow, but possible).
    If refit_each_step=False: fit once, then update each step; optionally refit every N.
    """
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(
            f"target_col '{target_col}' must exist in both train_df and test_df")
    if date_col not in test_df.columns:
        raise KeyError(f"date_col '{date_col}' must exist in test_df")

    y_train = train_df[target_col].astype(float).values
    y_test = test_df[target_col].astype(float).values
    dates = pd.to_datetime(test_df[date_col]).values

    preds: List[float] = []

    # initial fit
    model = fit_auto_arima(pd.Series(y_train))

    for i in range(len(y_test)):
        # optional refit policy
        if refit_each_step:
            # refit on all data seen so far (train + test[:i])
            y_hist = np.concatenate([y_train, y_test[:i]]).astype(float)
            model = fit_auto_arima(pd.Series(y_hist))
        else:
            # periodic refit if requested (still using update in between)
            if i > 0 and (i % max(int(refit_every), 1) == 0):
                y_hist = np.concatenate([y_train, y_test[:i]]).astype(float)
                model = fit_auto_arima(pd.Series(y_hist))

        # 1-step forecast
        try:
            yhat = float(model.predict(n_periods=1)[0])
        except Exception as e:
            if verbose:
                print(
                    f"[walk_forward_arima_forecast] step {i} predict failed: {e}")
            yhat = float(y_test[i - 1]) if i > 0 else float(y_train[-1])

        preds.append(yhat)

        # IMPORTANT: update with the actual observation
        try:
            model.update(y_test[i])
        except Exception as e:
            # if update fails, we can ignore and rely on periodic refits
            if verbose:
                print(
                    f"[walk_forward_arima_forecast] step {i} update failed: {e}")

        if verbose and (i % 25 == 0):
            print(
                f"[walk_forward] {i}/{len(y_test)} yhat={yhat:.4f} y={y_test[i]:.4f}")

    forecast_df = pd.DataFrame(
        {date_col: dates, "y_true": y_test,
            "y_pred": np.asarray(preds, dtype=float)}
    )

    params = {
        "asset": config.TASK2_ASSET,
        "target_col": target_col,
        "strategy": "walk_forward",
        "refit_each_step": bool(refit_each_step),
        "refit_every": int(refit_every),
    }
    params.update(_safe_model_info(model))
    params["metrics"] = all_metrics(
        forecast_df["y_true"], forecast_df["y_pred"])
    return forecast_df, params


def direct_arima_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fits ARIMA on train target, forecasts for len(test) in one shot.
    (Kept for comparison; less finance-correct for backtesting than walk-forward.)
    """
    y_train = train_df[target_col].astype(float)
    y_test = test_df[target_col].astype(float)

    model = fit_auto_arima(y_train)
    n_periods = len(test_df)

    preds = model.predict(n_periods=n_periods)
    preds = np.asarray(preds, dtype=float)

    forecast_df = pd.DataFrame(
        {
            date_col: pd.to_datetime(test_df[date_col]).values,
            "y_true": y_test.values,
            "y_pred": preds,
        }
    )

    params = {
        "asset": config.TASK2_ASSET,
        "target_col": target_col,
        "strategy": "direct",
    }
    params.update(_safe_model_info(model))
    params["metrics"] = all_metrics(
        forecast_df["y_true"], forecast_df["y_pred"])
    return forecast_df, params


def arima_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = config.TASK2_TARGET_COL,
    date_col: str = config.TASK2_DATE_COL,
    strategy: str = "walk_forward",
    refit_each_step: bool = True,
    refit_every: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main entrypoint.

    strategy:
      - "walk_forward" (recommended): rolling-origin 1-step ahead
      - "direct": fit once, forecast len(test) horizon
    """
    strategy = (strategy or "").lower().strip()
    if strategy not in {"walk_forward", "direct"}:
        raise ValueError("strategy must be one of: {'walk_forward', 'direct'}")

    if strategy == "walk_forward":
        return walk_forward_arima_forecast(
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            date_col=date_col,
            refit_each_step=refit_each_step,
            refit_every=refit_every,
        )

    return direct_arima_forecast(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        date_col=date_col,
    )


def _standardize_arima_forecast_df(
    forecast_df: pd.DataFrame,
    date_col: str = config.TASK2_DATE_COL,
) -> pd.DataFrame:
    """
    Ensure a stable schema for downstream:
      [date_col, y_true, arima_pred]

    Tolerates existing schemas:
      - prediction column: y_pred / pred / prediction / arima_pred
      - date column: config.TASK2_DATE_COL or 'date'
    """
    df = forecast_df.copy()

    # Date column normalization
    if date_col not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": date_col})
    if date_col not in df.columns:
        raise ValueError(
            f"ARIMA forecast_df missing date column '{date_col}'. Columns={list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])

    # y_true must exist (your code already guarantees it)
    if "y_true" not in df.columns:
        raise ValueError(
            f"ARIMA forecast_df missing 'y_true'. Columns={list(df.columns)}")

    # Prediction column normalization
    if "arima_pred" in df.columns:
        pass
    elif "y_pred" in df.columns:
        df = df.rename(columns={"y_pred": "arima_pred"})
    elif "pred" in df.columns:
        df = df.rename(columns={"pred": "arima_pred"})
    elif "prediction" in df.columns:
        df = df.rename(columns={"prediction": "arima_pred"})
    else:
        raise ValueError(
            "ARIMA forecast_df missing prediction column. Expected one of "
            "['arima_pred','y_pred','pred','prediction'] "
            f"but got columns={list(df.columns)}"
        )

    # Stable ordering
    df = df.sort_values(date_col).reset_index(drop=True)

    # Keep stable cols first; allow extra cols to pass through
    stable = [date_col, "y_true", "arima_pred"]
    extras = [c for c in df.columns if c not in stable]
    return df[stable + extras]


def save_arima_outputs(forecast_df: pd.DataFrame, params: Dict, save_model=None) -> None:
    """Saves forecast CSV, params JSON, and optionally the pmdarima model."""
    _ensure_dir(config.TASK2_FORECASTS_DIR)
    _ensure_dir(config.TASK2_METRICS_DIR)
    _ensure_dir(config.TASK2_MODELS_DIR)

    forecast_df = _standardize_arima_forecast_df(
        forecast_df, date_col=config.TASK2_DATE_COL
    )
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
