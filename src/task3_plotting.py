"""Module for plotting price forecasts and confidence intervals."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_forecast(
    history_price: pd.Series,
    future_q: pd.DataFrame,
    out_path: str,
    title: str,
    split_info: dict | None = None,
    test_overlay: pd.DataFrame | None = None,
):
    """
    history_price: pd.Series of historical adj_close (index=date)
    test_overlay: DataFrame indexed by date with optional columns:
        - actual_price
        - arima_price
        - lstm_price
    future_q: DataFrame indexed by future date with:
        - price_p05, price_p50, price_p95
    """
    plt.figure(figsize=(12, 6))

    # Historical
    plt.plot(
        history_price.index,
        history_price.to_numpy(dtype=float),
        label="Historical actual (Adj Close)",
        color="black",
        linewidth=1.5,
        alpha=0.85
    )

    # Test overlay (Task 2)
    if test_overlay is not None and len(test_overlay) > 0:
        if "actual_price" in test_overlay.columns:
            plt.plot(
                test_overlay.index,
                test_overlay["actual_price"].to_numpy(dtype=float),
                label="Test actual (Task 2 window)",
                color="dimgray",
                linewidth=1.5,
                alpha=0.9
            )
        if "arima_price" in test_overlay.columns:
            plt.plot(
                test_overlay.index,
                test_overlay["arima_price"].to_numpy(dtype=float),
                label="Test prediction (ARIMA)",
                color="tab:orange",
                linestyle="--",
                linewidth=2,
                alpha=0.95
            )
        if "lstm_price" in test_overlay.columns:
            plt.plot(
                test_overlay.index,
                test_overlay["lstm_price"].to_numpy(dtype=float),
                label="Test prediction (LSTM)",
                color="tab:green",
                linestyle="--",
                linewidth=2,
                alpha=0.85
            )

    # Future band
    plt.fill_between(
        future_q.index,
        future_q["price_p05"].to_numpy(dtype=float),
        future_q["price_p95"].to_numpy(dtype=float),
        color="tab:blue",
        alpha=0.20,
        label="95% prediction band (fan chart)",
    )
    plt.plot(
        future_q.index,
        future_q["price_p50"].to_numpy(dtype=float),
        label="Future forecast median (p50)",
        color="tab:blue",
        linewidth=2.2,
    )

    # Forecast start marker
    plt.axvline(future_q.index.min(), color="gray", linestyle="--",
                linewidth=1, label="Forecast start")

    # Optional split markers
    if split_info:
        if "cutoff_date_val" in split_info:
            plt.axvline(pd.to_datetime(
                split_info["cutoff_date_val"]), color="orange", linestyle=":", linewidth=1, label="Val cutoff")
        if "cutoff_date_test" in split_info:
            plt.axvline(pd.to_datetime(
                split_info["cutoff_date_test"]), color="red", linestyle=":", linewidth=1, label="Test cutoff")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ci_width(width: pd.Series, out_path: str, title: str):
    """
    Plot the width of the confidence interval over time.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(width.index, width.to_numpy(dtype=float),
             color="tab:purple", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Band width (p95 - p05)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
