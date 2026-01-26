""" Expected returns calculation utilities for Task 4. """
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def annualize_mean_simple_return(daily_simple_mean: float, trading_days: int = 252) -> float:
    """ Approximate annualized mean simple return from daily mean simple return. """
    return daily_simple_mean * trading_days


def annualize_mean_log_return(daily_log_mean: float, trading_days: int = 252) -> float:
    """ Exact annualized mean simple return from daily mean log return. """
    return float(np.exp(trading_days * daily_log_mean) - 1.0)


def load_tsla_forecast_logrets(path: Path) -> pd.Series:
    """ Load TSLA forecasted daily log returns from CSV file. """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise KeyError(
            f"Expected 'date' column in {path}. Found columns={list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Task3 logret CSV has ret_mean/ret_lower/ret_upper
    if "ret_mean" not in df.columns:
        raise KeyError(
            f"Expected 'ret_mean' in {path}. Found columns={list(df.columns)}")

    return df["ret_mean"].astype(float)


def build_expected_returns_vector(
    returns_wide: pd.DataFrame,
    tsla_forecast_logrets: pd.Series,
    trading_days: int = 252,
) -> pd.Series:
    """ Build expected returns vector for all assets. 
    Expected returns:
      - TSLA: from forecast mean daily log return -> annual simple return
      - SPY/BND: from historical mean daily simple return -> annual simple return (approx)
    """
    mu = {}
    mu["TSLA"] = annualize_mean_log_return(
        tsla_forecast_logrets.mean(), trading_days=trading_days)
    for asset in ["SPY", "BND"]:
        mu[asset] = annualize_mean_simple_return(
            float(returns_wide[asset].mean()), trading_days=trading_days)

    # preserve column order of returns_wide
    return pd.Series(mu).reindex(returns_wide.columns)
