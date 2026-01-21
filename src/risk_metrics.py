""" Risk metrics utility functions for the financial analysis project. """
# src/risk_metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(daily_returns: pd.Series, level: float = 0.95) -> float:
    """
    Historical Value at Risk (VaR) at confidence level.
    Convention: VaR is reported as a positive number representing loss threshold.
    Example: level=0.95 -> 5% left-tail quantile of returns (loss).
    """
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return float("nan")
    q = np.quantile(r, 1 - level)
    return float(-q)


def annualized_return(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    Annualized average return from daily returns."""
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return float("nan")
    return float(r.mean() * annualization_factor)


def annualized_vol(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    Annualized volatility (std dev) from daily returns."""
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(annualization_factor))


def sharpe_ratio(
    daily_returns: pd.Series,
    rf_annual: float = 0.02,
    annualization_factor: int = 252,
) -> float:
    """
    Sharpe = (E[R] - Rf) / Vol, all annualized.
    rf_annual is a constant annual risk-free rate assumption.
    """
    r_ann = annualized_return(daily_returns, annualization_factor)
    vol_ann = annualized_vol(daily_returns, annualization_factor)
    if vol_ann == 0 or np.isnan(vol_ann):
        return float("nan")
    return float((r_ann - rf_annual) / vol_ann)
