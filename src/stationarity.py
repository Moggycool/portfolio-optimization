""" Stationarity test utility functions for the financial analysis project. """
# src/stationarity.py
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, autolag: str = "AIC") -> dict:
    """
    Augmented Dickey-Fuller test.
    Returns a dict with test statistic, p-value, critical values, and a boolean stationarity flag (p<0.05).
    """
    s = pd.Series(series).dropna()
    if s.empty:
        raise ValueError(
            "ADF test received an empty series after dropping NaNs.")

    result = adfuller(s, autolag=autolag)
    test_stat, p_value, used_lag, n_obs, critical_values, _icbest = result

    return {
        "test_stat": float(test_stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "is_stationary_5pct": bool(p_value < 0.05),
    }
