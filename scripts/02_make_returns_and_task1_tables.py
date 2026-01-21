""" Generate returns dataset and Task 1 analysis tables. """
# scripts/02_make_returns_and_task1_tables.py
from __future__ import annotations

import pandas as pd

from src.config import (
    PRICES_PATH, RETURNS_PATH,
    TASK1_ADF_PATH, TASK1_RISK_PATH, TASK1_OUTLIERS_PATH,
    PRICE_COL, ANNUALIZATION_FACTOR, RISK_FREE_RATE_ANNUAL, DEFAULT_VAR_LEVEL,
)
from src.io import load_parquet, save_parquet, save_csv
from src.data_prep import add_daily_returns
from src.stationarity import adf_test
from src.risk_metrics import historical_var, sharpe_ratio, annualized_vol
from src.eda import detect_outliers_zscore


def main() -> None:
    """Generate returns dataset and Task 1 analysis tables."""
    prices = load_parquet(PRICES_PATH)
    prices_ret = add_daily_returns(prices, price_col=PRICE_COL)
    save_parquet(prices_ret, RETURNS_PATH)
    print(f"Saved returns dataset to: {RETURNS_PATH}")

    # --- ADF results (price and returns) ---
    adf_rows = []
    for asset, g in prices_ret.groupby("asset"):
        g = g.sort_values("date")
        adf_price = adf_test(g[PRICE_COL])
        adf_ret = adf_test(g["return"].dropna())

        adf_rows.append({
            "asset": asset,
            "series": PRICE_COL,
            **{k: v for k, v in adf_price.items() if k != "critical_values"},
            "crit_1pct": adf_price["critical_values"]["1%"],
            "crit_5pct": adf_price["critical_values"]["5%"],
            "crit_10pct": adf_price["critical_values"]["10%"],
        })
        adf_rows.append({
            "asset": asset,
            "series": "return",
            **{k: v for k, v in adf_ret.items() if k != "critical_values"},
            "crit_1pct": adf_ret["critical_values"]["1%"],
            "crit_5pct": adf_ret["critical_values"]["5%"],
            "crit_10pct": adf_ret["critical_values"]["10%"],
        })

    adf_df = pd.DataFrame(adf_rows).sort_values(["asset", "series"])
    save_csv(adf_df, TASK1_ADF_PATH)
    print(f"Saved ADF results to: {TASK1_ADF_PATH}")

    # --- Risk metrics ---
    risk_rows = []
    for asset, g in prices_ret.groupby("asset"):
        r = g["return"]
        risk_rows.append({
            "asset": asset,
            "var_level": DEFAULT_VAR_LEVEL,
            "VaR": historical_var(r, level=DEFAULT_VAR_LEVEL),
            "ann_vol": annualized_vol(r, annualization_factor=ANNUALIZATION_FACTOR),
            "sharpe_rf_annual": RISK_FREE_RATE_ANNUAL,
            "sharpe": sharpe_ratio(r, rf_annual=RISK_FREE_RATE_ANNUAL, annualization_factor=ANNUALIZATION_FACTOR),
        })

    risk_df = pd.DataFrame(risk_rows).sort_values("asset")
    save_csv(risk_df, TASK1_RISK_PATH)
    print(f"Saved risk metrics to: {TASK1_RISK_PATH}")

    # --- Outliers (z-score) ---
    outliers = detect_outliers_zscore(prices_ret, return_col="return", z=3.0)
    save_csv(outliers, TASK1_OUTLIERS_PATH)
    print(f"Saved outliers to: {TASK1_OUTLIERS_PATH}")


if __name__ == "__main__":
    main()
