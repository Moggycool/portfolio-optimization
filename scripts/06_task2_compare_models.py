""" Compare ARIMA and LSTM model forecasts for Task 2.

Produces:
- merged forecasts CSV (test set): includes raw + calibrated predictions
- model comparison CSV: MAE/RMSE/MAPE (raw + calibrated)
- error diagnostics CSV: bias, quantiles, win-rates (raw + calibrated)
"""
# scripts/06_task2_compare_models.py
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import pandas as pd

from src import config
from src.task2_metrics import (
    all_metrics,
    error_stats,
    fit_bias_offset,
    apply_bias_offset,
    win_rate,
)


def _load_forecast_csv(path: str, kind: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} forecast file not found: {path}")
    df = pd.read_csv(path)

    date_col = getattr(config, "TASK2_DATE_COL", "date")
    if date_col not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": date_col})

    if date_col not in df.columns:
        raise ValueError(
            f"{kind} forecast CSV missing date column '{date_col}': {path}. Columns={list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    if date_col != "date":
        # normalize to 'date' so the rest of your code stays unchanged
        df = df.rename(columns={date_col: "date"})
    return df


def _infer_pred_col(df: pd.DataFrame, kind: str) -> str:
    """Infers prediction column name for given kind of model."""
    # common in our pipeline
    if f"{kind.lower()}_pred" in df.columns:
        return f"{kind.lower()}_pred"
    for c in ["y_pred", "pred", "prediction"]:
        if c in df.columns:
            return c
    excluded = {"date", "asset", "y_true", "actual", "adj_close"}
    numeric_cols = [
        c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(
        f"Could not infer prediction column for {kind}. Columns={list(df.columns)}")


def _infer_true_col(df: pd.DataFrame) -> str:
    """Infers true value column name."""
    for c in ["y_true", "actual", "adj_close"]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer y_true column. Columns={list(df.columns)}")


def _prep_one(df: pd.DataFrame, kind: str, pred_name: str) -> pd.DataFrame:
    true_col = _infer_true_col(df)
    pred_col = _infer_pred_col(df, kind)
    out = df[["date", true_col, pred_col]].rename(
        columns={true_col: "y_true", pred_col: pred_name})
    return out


def _merge_on_date(arima_df: pd.DataFrame, lstm_df: pd.DataFrame) -> pd.DataFrame:
    """Merges two forecast dataframes on 'date' column."""
    merged = pd.merge(
        arima_df, lstm_df[["date", "lstm_pred"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.shape[0] == 0:
        raise ValueError(
            "Merged forecasts are empty after inner join on date. Check date alignment.")
    return merged


def _default_val_paths() -> Tuple[str, str]:
    """Returns default validation forecast file paths for ARIMA and LSTM models."""
    # if you later add config.TASK2_ARIMA_VAL_FORECAST_PATH, config.TASK2_LSTM_VAL_FORECAST_PATH
    # you can switch to those cleanly
    arima_val = getattr(config, "TASK2_ARIMA_VAL_FORECAST_PATH",
                        os.path.join(config.TASK2_FORECASTS_DIR, "tsla_arima_forecast_val.csv"))
    lstm_val = getattr(config, "TASK2_LSTM_VAL_FORECAST_PATH",
                       os.path.join(config.TASK2_FORECASTS_DIR, "tsla_lstm_forecast_val.csv"))
    return arima_val, lstm_val


def _maybe_fit_calibration_offsets() -> Optional[dict]:
    """
    If val forecasts exist for both models, fit bias offsets on validation.
    Returns dict with offsets or None if missing.
    """
    arima_val_path, lstm_val_path = _default_val_paths()
    if not (os.path.exists(arima_val_path) and os.path.exists(lstm_val_path)):
        print("VAL forecasts not found for both models; skipping calibration.")
        print("Expected (default):")
        print(" -", arima_val_path)
        print(" -", lstm_val_path)
        return None

    arima_val = _prep_one(_load_forecast_csv(
        arima_val_path, "ARIMA_VAL"), "ARIMA", "arima_pred")
    lstm_val = _prep_one(_load_forecast_csv(
        lstm_val_path, "LSTM_VAL"), "LSTM", "lstm_pred")
    val = _merge_on_date(arima_val, lstm_val)

    offsets = {
        "arima_offset": fit_bias_offset(val["y_true"], val["arima_pred"]),
        "lstm_offset": fit_bias_offset(val["y_true"], val["lstm_pred"]),
        "val_n": int(val.shape[0]),
    }
    print("Fitted calibration offsets on VAL:", offsets)
    return offsets


def _comparison_rows(df: pd.DataFrame, suffix: str = "") -> list[dict]:
    """
    Build comparison rows for a dataframe with y_true and arima_pred/lstm_pred.
    suffix used to label calibrated results.
    """
    rows = []
    for model_name, pred_col in [("ARIMA", "arima_pred"), ("LSTM_multivariate", "lstm_pred")]:
        m = all_metrics(df["y_true"], df[pred_col])
        rows.append({"model": model_name + suffix, **m})
    return rows


def _diagnostics_rows(df: pd.DataFrame, suffix: str = "") -> list[dict]:
    """
    Build diagnostics rows for a dataframe with y_true and arima_pred/lstm_pred."""
    rows = []
    for model_name, pred_col in [("ARIMA", "arima_pred"), ("LSTM_multivariate", "lstm_pred")]:
        m = all_metrics(df["y_true"], df[pred_col])
        s = error_stats(df["y_true"], df[pred_col])
        rows.append({"model": model_name + suffix, **s, **m})

    wins_arima, wins_lstm, ties = win_rate(
        df["y_true"], df["arima_pred"], df["lstm_pred"])
    rows.append({"model": "ARIMA_win_rate" + suffix,
                "win_rate_vs_other_pct": wins_arima, "n": int(df.shape[0])})
    rows.append({"model": "LSTM_win_rate" + suffix,
                "win_rate_vs_other_pct": wins_lstm, "n": int(df.shape[0])})
    rows.append({"model": "TIES" + suffix,
                "win_rate_vs_other_pct": ties, "n": int(df.shape[0])})
    return rows


def main() -> None:
    """Main script to compare ARIMA and LSTM model forecasts."""
    os.makedirs(config.TASK2_METRICS_DIR, exist_ok=True)
    os.makedirs(config.TASK2_FORECASTS_DIR, exist_ok=True)

    # --- Load TEST forecasts ---
    arima_test_raw = _prep_one(_load_forecast_csv(
        config.TASK2_ARIMA_FORECAST_PATH, "ARIMA_TEST"), "ARIMA", "arima_pred")
    lstm_test_raw = _prep_one(_load_forecast_csv(
        config.TASK2_LSTM_FORECAST_PATH, "LSTM_TEST"), "LSTM", "lstm_pred")
    test = _merge_on_date(arima_test_raw, lstm_test_raw)

    # --- Fit calibration on VAL (if available), apply to TEST ---
    offsets = _maybe_fit_calibration_offsets()

    test_out = test.copy()
    if offsets is not None:
        test_out["arima_pred_cal"] = apply_bias_offset(
            test_out["arima_pred"], offsets["arima_offset"])
        test_out["lstm_pred_cal"] = apply_bias_offset(
            test_out["lstm_pred"], offsets["lstm_offset"])
    else:
        test_out["arima_pred_cal"] = pd.NA
        test_out["lstm_pred_cal"] = pd.NA

    # Save merged forecasts (TEST)
    merged_out_path = config.TASK2_FORECASTS_MERGED_PATH
    test_out.to_csv(merged_out_path, index=False)
    print("Saved merged TEST forecasts:", merged_out_path)

    # --- Build comparison table (raw + calibrated if present) ---
    comparison_rows = []
    comparison_rows += _comparison_rows(test, suffix="")

    if offsets is not None:
        test_cal = test.copy()
        test_cal["arima_pred"] = apply_bias_offset(
            test_cal["arima_pred"], offsets["arima_offset"])
        test_cal["lstm_pred"] = apply_bias_offset(
            test_cal["lstm_pred"], offsets["lstm_offset"])
        comparison_rows += _comparison_rows(test_cal, suffix="_calibrated")

    comparison = pd.DataFrame(comparison_rows).sort_values("RMSE")
    out_path = config.TASK2_MODEL_COMPARISON_PATH
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        comparison.to_csv(out_path, index=False)
        print("Saved model comparison:", out_path)
    except PermissionError:
        # Common on Windows if the CSV is open in Excel.
        ts = time.strftime("%Y%m%d_%H%M%S")
        fallback = os.path.join(
            os.path.dirname(out_path) or ".",
            f"model_comparison_{ts}.csv",
        )
        comparison.to_csv(fallback, index=False)
        print("Permission denied writing:", out_path)
        print("Wrote model comparison to:", fallback)
    print(comparison)

    # --- Diagnostics table (raw + calibrated if present) ---
    diag_rows = []
    diag_rows += _diagnostics_rows(test, suffix="")

    if offsets is not None:
        test_cal = test.copy()
        test_cal["arima_pred"] = apply_bias_offset(
            test_cal["arima_pred"], offsets["arima_offset"])
        test_cal["lstm_pred"] = apply_bias_offset(
            test_cal["lstm_pred"], offsets["lstm_offset"])
        diag_rows += _diagnostics_rows(test_cal, suffix="_calibrated")

    diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(config.TASK2_METRICS_DIR, "error_diagnostics.csv")
    diag.to_csv(diag_path, index=False)
    print("Saved error diagnostics:", diag_path)


if __name__ == "__main__":
    main()
