""" A script to compare ARIMA and LSTM model forecasts for Task 2.
Saves a comparison CSV with MAE, RMSE, MAPE for both models."""
# scripts/06_task2_compare_models.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src import config


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _load_forecast_csv(path: str, kind: str) -> pd.DataFrame:
    """Loads a forecast CSV and ensures 'date' column is present."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} forecast file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(
            f"{kind} forecast CSV missing 'date' column: {path}. Columns={list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_merged_forecasts(
    arima_path: str,
    lstm_path: str,
    merged_out_path: str,
) -> pd.DataFrame:
    """
    Creates a merged dataframe with columns:
      date, y_true, arima_pred, lstm_pred

    It is tolerant to common column naming differences by detecting the prediction column.
    """
    arima = _load_forecast_csv(arima_path, "ARIMA")
    lstm = _load_forecast_csv(lstm_path, "LSTM")

    # --- infer prediction column names safely ---
    def infer_pred_col(df: pd.DataFrame, kind: str) -> str:
        candidates = [c for c in df.columns if c.lower(
        ) in {"y_pred", "pred", "prediction", f"{kind.lower()}_pred"}]
        # common in our pipeline: arima_pred / lstm_pred
        if f"{kind.lower()}_pred" in df.columns:
            return f"{kind.lower()}_pred"
        if "y_pred" in df.columns:
            return "y_pred"
        if "pred" in df.columns:
            return "pred"
        if "prediction" in df.columns:
            return "prediction"
        # fallback: choose a numeric column not date/asset/y_true
        excluded = {"date", "asset", "y_true", "actual", "adj_close"}
        numeric_cols = [
            c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        raise ValueError(
            f"Could not infer prediction column for {kind}. Columns={list(df.columns)}")

    arima_pred_col = infer_pred_col(arima, "ARIMA")
    lstm_pred_col = infer_pred_col(lstm, "LSTM")

    # --- infer y_true column (actuals) ---
    def infer_true_col(df: pd.DataFrame) -> str:
        """Infers the column name for true values."""
        if "y_true" in df.columns:
            return "y_true"
        if "actual" in df.columns:
            return "actual"
        if "adj_close" in df.columns:
            return "adj_close"
        raise ValueError(
            f"Could not infer y_true column. Columns={list(df.columns)}")

    arima_true_col = infer_true_col(arima)
    lstm_true_col = infer_true_col(lstm)

    arima_use = arima[["date", arima_true_col, arima_pred_col]].rename(
        columns={arima_true_col: "y_true", arima_pred_col: "arima_pred"}
    )
    lstm_use = lstm[["date", lstm_true_col, lstm_pred_col]].rename(
        columns={lstm_true_col: "y_true_lstm", lstm_pred_col: "lstm_pred"}
    )

    merged = pd.merge(
        arima_use, lstm_use[["date", "lstm_pred"]], on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)

    # basic sanity
    if merged.shape[0] == 0:
        raise ValueError(
            "Merged forecasts are empty after inner join on date. Check date alignment.")

    os.makedirs(os.path.dirname(merged_out_path), exist_ok=True)
    merged.to_csv(merged_out_path, index=False)
    print("Saved merged forecasts:", merged_out_path)
    return merged


def error_diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a diagnostics table with error statistics for each model."""
    out_rows = []
    for model_name, pred_col in [("ARIMA", "arima_pred"), ("LSTM_multivariate", "lstm_pred")]:
        err = (df[pred_col] - df["y_true"]).astype(float)

        out_rows.append({
            "model": model_name,
            "n": int(err.shape[0]),
            "MAE": mae(df["y_true"], df[pred_col]),
            "RMSE": rmse(df["y_true"], df[pred_col]),
            "MAPE_pct": mape(df["y_true"], df[pred_col]),
            "mean_error_bias": float(err.mean()),
            "median_error": float(err.median()),
            "std_error": float(err.std(ddof=1)),
            "median_abs_error": float(err.abs().median()),
            "q05_error": float(err.quantile(0.05)),
            "q25_error": float(err.quantile(0.25)),
            "q50_error": float(err.quantile(0.50)),
            "q75_error": float(err.quantile(0.75)),
            "q95_error": float(err.quantile(0.95)),
        })

    diag = pd.DataFrame(out_rows)

    abs_err_arima = (df["arima_pred"] - df["y_true"]).abs()
    abs_err_lstm = (df["lstm_pred"] - df["y_true"]).abs()

    wins_arima = float((abs_err_arima < abs_err_lstm).mean() * 100.0)
    wins_lstm = float((abs_err_lstm < abs_err_arima).mean() * 100.0)
    ties = float((abs_err_arima == abs_err_lstm).mean() * 100.0)

    diag["win_rate_vs_other_pct"] = np.nan
    diag.loc[diag["model"] == "ARIMA", "win_rate_vs_other_pct"] = wins_arima
    diag.loc[diag["model"] == "LSTM_multivariate",
             "win_rate_vs_other_pct"] = wins_lstm

    diag = pd.concat(
        [diag, pd.DataFrame(
            [{"model": "TIES", "n": int(df.shape[0]), "win_rate_vs_other_pct": ties}])],
        ignore_index=True,
    )

    return diag


def main() -> None:
    """Main function to compare ARIMA and LSTM models."""
    merged_path = config.TASK2_FORECASTS_MERGED_PATH
    arima_path = config.TASK2_ARIMA_FORECAST_PATH
    lstm_path = config.TASK2_LSTM_FORECAST_PATH

    metrics_dir = config.TASK2_METRICS_DIR
    os.makedirs(metrics_dir, exist_ok=True)

    # Build merged forecasts if missing
    if not os.path.exists(merged_path):
        print("Merged forecasts not found. Building from ARIMA + LSTM forecasts...")
        df = build_merged_forecasts(arima_path, lstm_path, merged_path)
    else:
        df = pd.read_csv(merged_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # required columns
    for c in ["y_true", "arima_pred", "lstm_pred"]:
        if c not in df.columns:
            raise ValueError(
                f"Merged forecasts missing column '{c}'. Columns={list(df.columns)}")

    # Required comparison table
    comparison = pd.DataFrame([
        {"model": "ARIMA", "MAE": mae(df["y_true"], df["arima_pred"]), "RMSE": rmse(
            df["y_true"], df["arima_pred"]), "MAPE_pct": mape(df["y_true"], df["arima_pred"])},
        {"model": "LSTM_multivariate", "MAE": mae(df["y_true"], df["lstm_pred"]), "RMSE": rmse(
            df["y_true"], df["lstm_pred"]), "MAPE_pct": mape(df["y_true"], df["lstm_pred"])},
    ]).sort_values("RMSE")

    comparison.to_csv(config.TASK2_MODEL_COMPARISON_PATH, index=False)
    print("Saved model comparison:", config.TASK2_MODEL_COMPARISON_PATH)
    print(comparison)

    # NEW: extra diagnostics CSV
    diag = error_diagnostics_table(df)
    diag_path = os.path.join(metrics_dir, "error_diagnostics.csv")
    diag.to_csv(diag_path, index=False)
    print("Saved error diagnostics:", diag_path)


if __name__ == "__main__":
    main()
