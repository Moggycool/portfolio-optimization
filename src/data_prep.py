""" Data preparation utility functions for the financial analysis project. """
# src/data_prep.py
from __future__ import annotations

import pandas as pd


REQUIRED_COLS = ["date", "asset", "open", "high",
                 "low", "close", "adj_close", "volume"]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has the required schema."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("Column 'date' must be datetime64 dtype.")


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning policy:
    - sort by asset/date
    - drop duplicates (asset, date) keeping last
    - forward-fill within each asset for price columns (open/high/low/close/adj_close)
    - volume: fill missing with 0 (rare); keep numeric
    - drop remaining missing rows (if any)
    """
    validate_schema(df)

    out = df.copy()
    out = out.sort_values(["asset", "date"]).drop_duplicates(
        ["asset", "date"], keep="last")

    price_cols = ["open", "high", "low", "close", "adj_close"]
    for c in price_cols + ["volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Forward-fill prices per asset (common for small gaps); then drop any remaining NA
    out[price_cols] = out.groupby("asset", group_keys=False)[
        price_cols].ffill()

    # Volume gaps -> 0 (explicit choice)
    out["volume"] = out["volume"].fillna(0)

    out = out.dropna(subset=price_cols)  # must have prices

    return out


def add_daily_returns(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """
    Adds 'return' = pct_change(price_col) within each asset.
    """
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not found in DataFrame.")

    out = df.copy()
    out = out.sort_values(["asset", "date"])
    out["return"] = out.groupby("asset", group_keys=False)[
        price_col].pct_change()
    return out
