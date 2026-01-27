from __future__ import annotations
import numpy as np
import pandas as pd


def cumulative_from_returns(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns).cumprod()


def total_return(cum: pd.Series) -> float:
    return float(cum.iloc[-1] - 1.0)


def annualized_return(cum: pd.Series, trading_days: int = 252) -> float:
    n = len(cum)
    if n <= 1:
        return 0.0
    return float(cum.iloc[-1] ** (trading_days / n) - 1.0)


def annualized_volatility(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float, trading_days: int = 252) -> float:
    rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=1) * np.sqrt(trading_days)
    if vol == 0:
        return float("-inf")
    ann_excess = excess.mean() * trading_days
    return float(ann_excess / vol)


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())
