""" Plotting utilities for Task 4. """
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_cov_heatmap(cov: pd.DataFrame, out_path: str):
    """ Plot heatmap of covariance matrix. """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cov, annot=True, fmt=".4f", cmap="coolwarm", square=True)
    plt.title("Annualized Covariance Matrix (TSLA, SPY, BND)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_efficient_frontier(rets, vols, minvol_point, maxsharpe_point, out_path: str):
    """ Plot efficient frontier with min-vol and max-sharpe points. """
    plt.figure(figsize=(8, 6))
    plt.plot(vols, rets, label="Efficient Frontier", color="tab:blue")
    plt.scatter([minvol_point["vol"]], [minvol_point["ret"]],
                color="tab:green", s=90, label="Min Volatility")
    plt.scatter([maxsharpe_point["vol"]], [maxsharpe_point["ret"]],
                color="tab:red", s=90, label="Max Sharpe")
    plt.xlabel("Volatility (annualized)")
    plt.ylabel("Expected Return (annualized)")
    plt.title("Efficient Frontier (TSLA view from forecast; SPY/BND from history)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
