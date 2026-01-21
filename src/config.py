""" Configuration constants for the financial analysis project."""
# src/config.py
from __future__ import annotations

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-01-15"

PRICE_COL = "adj_close"          # Use adjusted close for returns/risk metrics
ANNUALIZATION_FACTOR = 252       # Trading days per year

RISK_FREE_RATE_ANNUAL = 0.02     # 2% annual (user choice)
DEFAULT_VAR_LEVEL = 0.95

# Project paths (relative to repo root)
PROCESSED_DIR = "data/processed"
PRICES_PATH = f"{PROCESSED_DIR}/prices.parquet"
RETURNS_PATH = f"{PROCESSED_DIR}/returns.parquet"

TASK1_ADF_PATH = f"{PROCESSED_DIR}/task1_adf_results.csv"
TASK1_RISK_PATH = f"{PROCESSED_DIR}/task1_risk_metrics.csv"
TASK1_OUTLIERS_PATH = f"{PROCESSED_DIR}/task1_outliers.csv"
