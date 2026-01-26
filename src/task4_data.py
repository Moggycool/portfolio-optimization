""" Data loading and transformation utilities for Task 4. """
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def pivot_returns_long_to_wide(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    ret_col: str,
    assets: list[str],
) -> pd.DataFrame:
    """Pivot long-format returns DataFrame to wide-format."""
    out = df[df[asset_col].isin(assets)].copy()
    out[date_col] = pd.to_datetime(out[date_col])

    # Ensure a deterministic "last" when duplicates exist
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

    # drop dates with missing any asset to keep covariance consistent
    wide = wide.dropna(how="any")
    return wide
