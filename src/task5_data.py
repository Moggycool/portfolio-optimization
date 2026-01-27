"""Module for loading and processing financial data for backtesting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def pivot_returns_long_to_wide(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    ret_col: str,
    assets: list[str],
) -> pd.DataFrame:
    """Pivot long-format returns DataFrame to wide-format for specified assets."""
    out = df[df[asset_col].isin(assets)].copy()
    out[date_col] = pd.to_datetime(out[date_col])

    # Ensure deterministic "last" when duplicates exist
    out = out.sort_values([date_col, asset_col])

    def _last(x: pd.Series) -> Any:
        # pandas stubs don't accept aggfunc="last"; use a callable instead
        return x.iloc[-1]

    wide = (
        out.pivot_table(
            index=date_col,
            columns=asset_col,
            values=ret_col,
            aggfunc=_last,
        )
        .sort_index()
    )
    wide = wide.dropna(how="any")
    return wide


def slice_backtest_window(returns_wide: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice the returns DataFrame to the specified backtest window."""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return returns_wide.loc[(returns_wide.index >= s) & (returns_wide.index <= e)].copy()


def load_json(path: Path) -> dict:
    """Load a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))
