""" Fetch and save financial price data using yfinance. """
# scripts/01_fetch_data.py
from __future__ import annotations

from src.config import TICKERS, START_DATE, END_DATE, PRICES_PATH
from src.data_fetch import fetch_yfinance_prices
from src.data_prep import clean_prices
from src.io import save_parquet


def main() -> None:
    """Fetch, clean, and save price data."""
    df = fetch_yfinance_prices(TICKERS, START_DATE, END_DATE)
    df = clean_prices(df)
    save_parquet(df, PRICES_PATH)
    print(f"Saved cleaned prices to: {PRICES_PATH}")
    print(df.groupby("asset")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
