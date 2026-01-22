""" A module for plotting results for Task 2: Time Series Forecasting. """
# src/task2_plots.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config


def _ensure_dir(path: str) -> None:
    """ Creates directory if it does not exist. """
    os.makedirs(path, exist_ok=True)


def plot_forecasts(
    merged_forecasts: pd.DataFrame,
    date_col: str = config.TASK2_DATE_COL,
    actual_col: str = "y_true",
    save_path: str = config.TASK2_FORECAST_PLOT_PATH,
) -> None:
    """
    merged_forecasts expects columns:
      date, y_true, arima_pred, lstm_pred (names produced in compare script below)
    """
    _ensure_dir(config.TASK2_FIGURES_DIR)

    df = merged_forecasts.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    plt.figure(figsize=(12, 5))
    plt.plot(df[date_col], df[actual_col],
             label="Actual (TSLA adj_close)", linewidth=2)
    if "arima_pred" in df.columns:
        plt.plot(df[date_col], df["arima_pred"],
                 label="ARIMA forecast", linewidth=1.5)
    if "lstm_pred" in df.columns:
        plt.plot(df[date_col], df["lstm_pred"],
                 label="LSTM forecast", linewidth=1.5)

    plt.title("TSLA Forecasts on Test Period")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_acf_pacf(series: pd.Series, save_path: str = config.TASK2_ACF_PACF_PLOT_PATH, lags: int = 60) -> None:
    """ Plots and saves ACF and PACF plots for the given series. """
    _ensure_dir(config.TASK2_FIGURES_DIR)

    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for ACF/PACF plots. Install: pip install statsmodels") from e

    s = pd.Series(series).dropna().astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(s, lags=lags, ax=axes[0])
    plot_pacf(s, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    fig.suptitle("ACF/PACF of TSLA adj_close", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
