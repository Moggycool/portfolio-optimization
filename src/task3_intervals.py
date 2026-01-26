"""Helper functions for Task 3: Intervals and price path simulations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statistics import NormalDist


def sigma_from_pi(ret_lower: pd.Series, ret_upper: pd.Series, alpha: float) -> pd.Series:
    """Compute standard deviation from prediction interval bounds."""
    z = NormalDist().inv_cdf(1 - alpha / 2)
    sigma = (ret_upper - ret_lower) / (2 * z)
    return sigma.clip(lower=1e-12)


def simulate_price_paths_from_logrets(
    last_price: float,
    logret_mean: pd.Series,
    logret_sigma: pd.Series,
    n_sims: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate future price paths from *log-return* forecasts:
      log(P_t) = log(P0) + cumsum(logret_t)
      logret_t ~ Normal(mean_t, sigma_t)
    Returns DataFrame shape: (steps, n_sims) indexed by forecast dates.
    """
    rng = np.random.default_rng(seed)
    steps = len(logret_mean)

    mu = logret_mean.to_numpy().reshape(steps, 1)
    sig = logret_sigma.to_numpy().reshape(steps, 1)

    eps = rng.normal(loc=0.0, scale=1.0, size=(steps, n_sims))
    r = mu + sig * eps

    log_price = np.log(last_price) + np.cumsum(r, axis=0)
    prices = np.exp(log_price)

    return pd.DataFrame(prices, index=logret_mean.index)


def price_quantiles(paths: pd.DataFrame, qs=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Compute price quantiles from simulated price paths."""
    q = paths.quantile(q=list(qs), axis=1).T
    q.columns = [f"price_p{int(x*100):02d}" for x in qs]
    return q
