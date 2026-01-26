"""Helper functions for Task 3: Data loading and processing."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def get_asset_series(
    df: pd.DataFrame,
    asset: str,
    value_col: str,
    date_col: str = "date",
    asset_col: str = "asset",
) -> pd.Series:
    """Extract a time series for a specific asset from a DataFrame."""
    out = df.loc[df[asset_col] == asset, [date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out.sort_values(date_col, inplace=True)
    s = out.set_index(date_col)[value_col]
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    return s


def align_on_intersection(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two series on their common index intersection."""
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def make_future_bday_index(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    """Create a future business-day DatetimeIndex starting after last_date."""
    start = last_date + pd.tseries.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=steps)


def load_split_info(path: Path) -> dict | None:
    """Load JSON split info from file, or return None if file does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_bday_frequency(ser: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Ensure a business-day DatetimeIndex with freq for statsmodels stability.
    Missing business days are filled (returns -> 0 by default).
    """
    ser = ser.copy()
    ser.index = pd.to_datetime(ser.index)
    # create full business-day index between min and max observed dates
    full_idx = pd.bdate_range(ser.index.min(), ser.index.max(), freq="B")
    ser = ser.reindex(full_idx)
    ser = ser.fillna(fill_value)
    ser.index.name = "date"
    return ser


def to_log_return(simple_ret: pd.Series) -> pd.Series:
    """
    Convert simple return r_t to log return log(1+r_t).
    Guards against r <= -1.
    """
    r = simple_ret.copy()
    r = r.clip(lower=-0.999999)
    return pd.Series(np.log1p(r.to_numpy()), index=r.index, name=r.name)
