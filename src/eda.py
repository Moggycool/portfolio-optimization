""" Exploratory data analysis utility functions for the financial analysis project. """
# src/eda.py
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling volatility (std dev) of returns over specified window."""
    return returns.rolling(window=window).std()


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling mean of a series over specified window."""
    return series.rolling(window=window).mean()


def detect_outliers_zscore(
    returns_df: pd.DataFrame,
    return_col: str = "return",
    z: float = 3.0,
) -> pd.DataFrame:
    """
    returns_df must include: date, asset, return_col
    Outputs rows where |zscore| >= z, computed per asset.
    """
    df = returns_df[["date", "asset", return_col]].dropna().copy()

    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sigma

    df["zscore"] = df.groupby("asset", group_keys=False)[
        return_col].apply(_zscore)
    outliers = df[df["zscore"].abs() >= z].sort_values(["asset", "date"])
    return outliers


def summary_stats(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic descriptive stats for key numeric columns per asset.
    """
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    return (
        prices_df.groupby("asset")[numeric_cols]
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"level_0": "field", "level_1": "stat"})
    )


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a daily returns series.
    """
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())
