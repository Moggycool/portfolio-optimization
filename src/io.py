""" Input/output utility functions for the financial analysis project. """
# src/io.py
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def ensure_dir(path: str | Path) -> None:
    """Ensure that a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to a Parquet file."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file."""
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to a CSV file."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: str | Path) -> None:
    """Save a dictionary to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
