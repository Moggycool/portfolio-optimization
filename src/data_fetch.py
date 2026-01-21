""" Data fetching utility functions for the financial analysis project. """
# src/data_fetch.py
from __future__ import annotations

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "yfinance is required. Add it to requirements.txt and install it.") from e


def fetch_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for multiple tickers from yfinance and return a tidy (long-form) DataFrame.

    Output columns:
      date, asset, open, high, low, close, adj_close, volume
    """
    # group_by="column" gives multiindex columns: (field, ticker)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="column",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise ValueError(
            "No data returned from yfinance. Check tickers/date range/network.")

    # If single ticker, yfinance returns single-level columns; normalize to multiindex
    if not isinstance(raw.columns, pd.MultiIndex):
        # columns like: Open, High, Low, Close, Adj Close, Volume
        raw.columns = pd.MultiIndex.from_product([raw.columns, [tickers[0]]])

    # Stack tickers into rows
    # raw has index Date; columns (Field, Ticker)

    tidy = raw.stack(level=1, future_stack=True).reset_index()

    # Robust: regardless of whether columns are named Date/Ticker or level_0/level_1
    tidy = tidy.rename(
        columns={tidy.columns[0]: "date", tidy.columns[1]: "asset"})

    # Standardize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    tidy = tidy.rename(columns=rename_map)

    # Keep only expected columns
    keep = ["date", "asset", "open", "high",
            "low", "close", "adj_close", "volume"]
    tidy = tidy[keep]

    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None)
    tidy["asset"] = tidy["asset"].astype(str)

    return tidy
