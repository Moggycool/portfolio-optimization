"""Module implementing portfolio simulation strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- helpers (needed by the strategy functions) ---


def _to_weight_vector(assets: list[str], w_dict: dict) -> np.ndarray:
    """Convert a weight dict to an ndarray aligned with `assets`."""
    return np.array([float(w_dict.get(a, 0.0)) for a in assets], dtype=float)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize weight vector to sum to 1 (no-op if sum is zero)."""
    s = float(np.sum(w))
    return (w / s) if s != 0.0 else w


def _ensure_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Coerce an Index to a DatetimeIndex (raises if conversion fails)."""
    if isinstance(index, pd.DatetimeIndex):
        return index
    return pd.DatetimeIndex(pd.to_datetime(index))


def get_monthly_rebalance_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Rebalance on the first available trading day of each month in `index`."""
    idx = _ensure_datetime_index(index)
    month = idx.to_period("M")
    is_first = month != month.shift(1)
    return idx[is_first]


def simulate_portfolio_hold(returns: pd.DataFrame, weights: dict, transaction_cost_bps: float = 0.0) -> pd.Series:
    """Basic strategy: hold fixed weights throughout the period."""
    assets = list(returns.columns)
    w = normalize_weights(_to_weight_vector(assets, weights))
    port_ret = returns.values @ w

    # apply one-time transaction cost at start if enabled
    if transaction_cost_bps and transaction_cost_bps > 0:
        tc = (transaction_cost_bps / 10000.0) * float(np.sum(np.abs(w)))
        port_ret[0] -= tc

    return pd.Series(port_ret, index=returns.index, name="strategy_return")


def simulate_portfolio_monthly_rebalance(
    returns: pd.DataFrame,
    target_weights: dict,
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """Advanced strategy: rebalance to target weights monthly."""
    idx = _ensure_datetime_index(returns.index)

    assets = list(returns.columns)
    target = normalize_weights(_to_weight_vector(assets, target_weights))

    reb_dates = get_monthly_rebalance_dates(idx)
    reb_set = set(reb_dates)  # faster `in` checks

    w = target.copy()
    port_rets: list[float] = []
    turnover = pd.Series(0.0, index=idx, name="turnover")

    for dt, r in zip(idx, returns.to_numpy()):
        dt = pd.Timestamp(dt)  # ensure scalar Timestamp

        if dt in reb_set:
            t = float(np.sum(np.abs(target - w)))
            turnover.at[dt] = t  # scalar label assignment
            w = target.copy()
            tc = (
                (transaction_cost_bps / 10000.0) * t
                if transaction_cost_bps and transaction_cost_bps > 0
                else 0.0
            )
        else:
            tc = 0.0

        pr = float(r @ w) - tc
        port_rets.append(pr)

        # weights drift after returns
        w = w * (1.0 + r)
        w = normalize_weights(w)

    return pd.Series(port_rets, index=idx, name="strategy_return"), turnover, reb_dates
