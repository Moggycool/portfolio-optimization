""" Risk calculation utilities for Task 4: Black-Litterman Model Implementation. """
from __future__ import annotations
import numpy as np
import pandas as pd


def cov_annualized(returns_wide: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """Compute annualized covariance matrix from daily returns."""
    return returns_wide.cov() * trading_days


def portfolio_performance(weights: np.ndarray, mu_annual: np.ndarray, cov_annual: np.ndarray, rf: float):
    """Calculate portfolio performance: return, volatility, Sharpe ratio."""
    w = weights.reshape(-1, 1)
    ret = float((mu_annual.reshape(1, -1) @ w)[0, 0])
    vol = float(np.sqrt((w.T @ cov_annual @ w)[0, 0]))
    sharpe = (ret - rf) / vol if vol > 0 else -np.inf
    return ret, vol, sharpe
