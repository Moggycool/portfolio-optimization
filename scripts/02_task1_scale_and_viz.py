"""Task 1: create scaling evidence + required visualizations.

Produces:
- data/task1/processed/scaled_task1_prices.parquet
- outputs/task1/viz/task1_prices_timeseries.png
- outputs/task1/viz/task1_daily_pct_change.png
- outputs/task1/viz/task1_rolling_mean_std.png

Run:
  python -m scripts.02_task1_scale_and_viz
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    denom = s.max() - s.min()
    if float(denom) == 0.0:
        return s * 0.0
    return (s - s.min()) / denom


def main() -> None:
    prices = pd.read_parquet(config.PRICES_PATH)
    returns = pd.read_parquet(config.RETURNS_PATH)

    # Normalize date column
    prices["date"] = pd.to_datetime(prices["date"])
    returns["date"] = pd.to_datetime(returns["date"])

    price_col = getattr(config, "PRICE_COL", "adj_close")

    # --- Scaling evidence parquet ---
    scaled = prices.copy()
    scaled[f"{price_col}_scaled"] = scaled.groupby(
        "asset")[price_col].transform(_minmax)

    scaled_path = getattr(config, "TASK1_SCALED_PRICES_PATH",
                          "data/task1/processed/scaled_task1_prices.parquet")
    _ensure_dir(os.path.dirname(scaled_path) or ".")
    scaled.to_parquet(scaled_path, index=False)
    print("Saved scaled dataset:", scaled_path)

    # --- Plots directory ---
    viz_dir = getattr(config, "TASK1_VIZ_DIR", "outputs/task1/viz")
    _ensure_dir(viz_dir)

    # 1) Prices time series
    plt.figure(figsize=(12, 5))
    for asset, g in prices.sort_values("date").groupby("asset"):
        plt.plot(g["date"], g[price_col], label=asset, linewidth=1.5)
    plt.title(f"Prices over time ({price_col})")
    plt.xlabel("Date")
    plt.ylabel(price_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(viz_dir, "task1_prices_timeseries.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Saved:", out1)

    # 2) Daily percent change (from returns if present; else from prices)
    plt.figure(figsize=(12, 5))
    if "return" in returns.columns:
        ret_col = "return"
        tmp = returns.copy()
        tmp["pct_change"] = tmp[ret_col].astype(float) * 100.0
    else:
        tmp = prices.sort_values("date").copy()
        tmp["pct_change"] = tmp.groupby(
            "asset")[price_col].pct_change() * 100.0

    for asset, g in tmp.groupby("asset"):
        plt.plot(g["date"], g["pct_change"], label=asset, linewidth=1.0)
    plt.title("Daily % change")
    plt.xlabel("Date")
    plt.ylabel("%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(viz_dir, "task1_daily_pct_change.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Saved:", out2)

    # 3) Rolling mean/std of daily % change (20-day window)
    window = 20
    tmp2 = tmp.sort_values("date").copy()
    tmp2["roll_mean"] = tmp2.groupby("asset")["pct_change"].transform(
        lambda s: s.rolling(window).mean())
    tmp2["roll_std"] = tmp2.groupby("asset")["pct_change"].transform(
        lambda s: s.rolling(window).std())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for asset, g in tmp2.groupby("asset"):
        axes[0].plot(g["date"], g["roll_mean"], label=asset, linewidth=1.2)
        axes[1].plot(g["date"], g["roll_std"], label=asset, linewidth=1.2)

    axes[0].set_title(f"Rolling mean of daily % change ({window}D)")
    axes[1].set_title(f"Rolling std of daily % change ({window}D)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[1].set_xlabel("Date")
    fig.tight_layout()
    out3 = os.path.join(viz_dir, "task1_rolling_mean_std.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print("Saved:", out3)


if __name__ == "__main__":
    main()
