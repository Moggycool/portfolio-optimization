""" A script to train ARIMA model for Task 2: Time Series Forecasting. """
# scripts/04_task2_train_arima.py
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import pandas as pd

from src import config
from src.task2_arima import arima_forecast, save_arima_outputs


def _ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing required columns {missing}. Columns={list(df.columns)}")


def _standardize_forecast_columns(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce a stable schema for downstream merge/compare:
      date, y_true, arima_pred
    Tolerates common alternatives like y_pred/prediction/adj_close.
    """
    df = forecast_df.copy()

    # date
    if config.TASK2_DATE_COL not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": config.TASK2_DATE_COL})
    if config.TASK2_DATE_COL not in df.columns:
        raise ValueError(
            f"Forecast DF missing date column '{config.TASK2_DATE_COL}'. Columns={list(df.columns)}")
    df[config.TASK2_DATE_COL] = pd.to_datetime(df[config.TASK2_DATE_COL])

    # y_true
    if "y_true" not in df.columns:
        if "actual" in df.columns:
            df = df.rename(columns={"actual": "y_true"})
        elif config.TASK2_TARGET_COL in df.columns:
            # sometimes forecast df carries adj_close as the true
            df = df.rename(columns={config.TASK2_TARGET_COL: "y_true"})
        else:
            raise ValueError(
                f"Forecast DF missing y_true. Columns={list(df.columns)}")

    # prediction -> arima_pred
    if "arima_pred" not in df.columns:
        if "y_pred" in df.columns:
            df = df.rename(columns={"y_pred": "arima_pred"})
        elif "pred" in df.columns:
            df = df.rename(columns={"pred": "arima_pred"})
        elif "prediction" in df.columns:
            df = df.rename(columns={"prediction": "arima_pred"})
        else:
            # last resort: detect a single numeric column aside from y_true
            excluded = {config.TASK2_DATE_COL, "y_true", "asset"}
            numeric_cols = [
                c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) == 1:
                df = df.rename(columns={numeric_cols[0]: "arima_pred"})
            else:
                raise ValueError(
                    f"Forecast DF missing prediction column. Columns={list(df.columns)}")

    # keep only the stable columns + any extras (optional), but stable first
    stable = [config.TASK2_DATE_COL, "y_true", "arima_pred"]
    keep = stable + [c for c in df.columns if c not in stable]
    df = df[keep].sort_values(config.TASK2_DATE_COL).reset_index(drop=True)
    return df


def _save_csv_json(forecast_df: pd.DataFrame, params: Dict, forecast_path: str, params_path: str) -> None:
    """Save forecast DataFrame to CSV and params dict to JSON."""
    _ensure_dir(os.path.dirname(forecast_path))
    _ensure_dir(os.path.dirname(params_path))
    forecast_df.to_csv(forecast_path, index=False)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def _fit_and_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_name: str,
) -> Tuple[pd.DataFrame, Dict]:
    forecast_df, params = arima_forecast(
        train_df,
        test_df,
        strategy="walk_forward",
        refit_each_step=True,
    )
    forecast_df = _standardize_forecast_columns(forecast_df)
    params = dict(params)
    params["split"] = split_name
    return forecast_df, params


def main() -> None:
    """Main training script for ARIMA model on Task 2 data."""
    # --- Load splits ---
    train_df = pd.read_parquet(config.TASK2_TRAIN_SPLIT_PATH)
    val_df = pd.read_parquet(config.TASK2_VAL_SPLIT_PATH)
    test_df = pd.read_parquet(config.TASK2_TEST_SPLIT_PATH)

    # --- Basic validation ---
    _require_cols(train_df, [config.TASK2_DATE_COL,
                  config.TASK2_TARGET_COL], "TRAIN")
    _require_cols(val_df, [config.TASK2_DATE_COL,
                  config.TASK2_TARGET_COL], "VAL")
    _require_cols(test_df, [config.TASK2_DATE_COL,
                  config.TASK2_TARGET_COL], "TEST")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Ensure chronological order
    train_df = train_df.sort_values(
        config.TASK2_DATE_COL).reset_index(drop=True)
    val_df = val_df.sort_values(config.TASK2_DATE_COL).reset_index(drop=True)
    test_df = test_df.sort_values(config.TASK2_DATE_COL).reset_index(drop=True)

    # -------------------------
    # Validation: TRAIN -> VAL
    # -------------------------
    val_forecast_df, val_params = _fit_and_forecast(
        train_df, val_df, split_name="val")

    # Save validation outputs (new files, driven by config)
    val_forecast_path = getattr(
        config,
        "TASK2_ARIMA_VAL_FORECAST_PATH",
        os.path.join(config.TASK2_FORECASTS_DIR,
                     "tsla_arima_forecast_val.csv"),
    )
    val_params_path = os.path.join(
        config.TASK2_METRICS_DIR, "arima_params_val.json")
    _save_csv_json(val_forecast_df, val_params,
                   val_forecast_path, val_params_path)

    # -------------------------
    # Test: (TRAIN+VAL) -> TEST
    # -------------------------
    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    trainval_df = trainval_df.sort_values(
        config.TASK2_DATE_COL).reset_index(drop=True)

    test_forecast_df, test_params = _fit_and_forecast(
        trainval_df, test_df, split_name="test")

    # Save test outputs to your existing canonical paths (keeps downstream compatibility)
    # NOTE: save_arima_outputs() will save to config.TASK2_ARIMA_FORECAST_PATH etc.
    save_arima_outputs(test_forecast_df, test_params, save_model=None)

    print("Saved ARIMA validation + test forecasts + params.")
    if "metrics" in val_params:
        print("VAL metrics:", val_params["metrics"])
    if "metrics" in test_params:
        print("TEST metrics:", test_params["metrics"])


if __name__ == "__main__":
    main()
