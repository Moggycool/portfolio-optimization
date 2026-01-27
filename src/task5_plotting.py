"""Module for plotting cumulative returns of strategy and benchmark.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_returns(
    strategy_cum: pd.Series,
    bench_cum: pd.Series,
    out_path: str,
    title: str,
):
    """Plot cumulative returns of strategy and benchmark."""
    # Convert to types matplotlib/pylance are happy with
    x_s = pd.to_datetime(strategy_cum.index).to_pydatetime()
    y_s = strategy_cum.to_numpy(dtype=float)

    x_b = pd.to_datetime(bench_cum.index).to_pydatetime()
    y_b = bench_cum.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_s, y_s, label="Strategy", linewidth=2)
    ax.plot(x_b, y_b, label="Benchmark (60% SPY / 40% BND)", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
