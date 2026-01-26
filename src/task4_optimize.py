"""" Portfolio optimization utilities for Task 4. """
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from src.task4_risk import portfolio_performance


def _constraint_sum_to_one():
    """ Constraint: sum of weights equals 1. """
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _bounds_long_only(n):
    return tuple((0.0, 1.0) for _ in range(n))


def min_vol_portfolio(mu_annual, cov_annual, rf=0.0):
    """ Find minimum volatility portfolio weights. """
    mu_annual = np.asarray(mu_annual)
    cov_annual = np.asarray(cov_annual)
    n = len(mu_annual)
    x0 = np.repeat(1/n, n)

    def obj(w):
        _, vol, _ = portfolio_performance(w, mu_annual, cov_annual, rf)
        return vol

    res = minimize(obj, x0, bounds=_bounds_long_only(
        n), constraints=[_constraint_sum_to_one()])
    if not res.success:
        raise RuntimeError("Min-vol optimization failed: " + res.message)
    return res.x


def max_sharpe_portfolio(mu_annual, cov_annual, rf=0.0):
    """ Find maximum Sharpe ratio portfolio weights. """
    mu_annual = np.asarray(mu_annual)
    cov_annual = np.asarray(cov_annual)
    n = len(mu_annual)
    x0 = np.repeat(1/n, n)

    def obj(w):
        _, _, s = portfolio_performance(w, mu_annual, cov_annual, rf)
        return -s

    res = minimize(obj, x0, bounds=_bounds_long_only(
        n), constraints=[_constraint_sum_to_one()])
    if not res.success:
        raise RuntimeError("Max-Sharpe optimization failed: " + res.message)
    return res.x


def efficient_frontier(mu_annual, cov_annual, rf=0.0, n_points=60):
    """ Compute efficient frontier points. """
    mu_annual = np.asarray(mu_annual)
    cov_annual = np.asarray(cov_annual)
    n = len(mu_annual)

    targets = np.linspace(float(mu_annual.min()),
                          float(mu_annual.max()), n_points)

    vols, rets, weights = [], [], []
    for tgt in targets:
        x0 = np.repeat(1/n, n)
        cons = [
            _constraint_sum_to_one(),
            {"type": "eq", "fun": lambda w, t=tgt: (mu_annual @ w) - t},
        ]

        def obj(w):
            _, vol, _ = portfolio_performance(w, mu_annual, cov_annual, rf)
            return vol

        res = minimize(obj, x0, bounds=_bounds_long_only(n), constraints=cons)
        if res.success:
            w = res.x
            r, v, _ = portfolio_performance(w, mu_annual, cov_annual, rf)
            rets.append(r)
            vols.append(v)
            weights.append(w)

    return np.array(rets), np.array(vols), np.array(weights)
