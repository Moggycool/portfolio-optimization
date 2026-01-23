""" Data loading and processing for Task 2: Time Series Forecasting. """
# src/task2_data.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class SplitInfo:
    asset: str
    split_year: int
    val_year: int
    cutoff_date_val: str
    cutoff_date_test: str
    n_total: int
    n_train: int
    n_val: int
    n_test: int


def _ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_prices(path: str = config.PRICES_PATH) -> pd.DataFrame:
    """Loads the raw prices parquet file."""
    df = pd.read_parquet(path)
    if config.TASK2_DATE_COL in df.columns:
        df[config.TASK2_DATE_COL] = pd.to_datetime(df[config.TASK2_DATE_COL])
    return df


def filter_asset(df: pd.DataFrame, asset: str = config.TASK2_ASSET) -> pd.DataFrame:
    """Filters the dataframe for a single asset and sorts by date."""
    out = df.loc[df["asset"] == asset].copy()
    out = out.sort_values(config.TASK2_DATE_COL).reset_index(drop=True)
    return out


def find_last_trading_day_in_year(df_asset: pd.DataFrame, year: int) -> pd.Timestamp:
    """Finds the last trading day in the given year for the asset dataframe."""
    dates = pd.to_datetime(df_asset[config.TASK2_DATE_COL])
    in_year = df_asset.loc[dates.dt.year == year, config.TASK2_DATE_COL]
    if in_year.empty:
        raise ValueError(f"No rows found for year={year} for asset.")
    return pd.to_datetime(in_year.max())


def make_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered features named exactly as config.TASK2_ENGINEERED_FEATURES.
    Operates on an asset-filtered dataframe sorted by date.
    """
    out = df.copy()

    # 1d simple return on adj_close
    out["ret_1d"] = out[config.TASK2_TARGET_COL].pct_change()

    # 1d log return
    out["logret_1d"] = np.log(out[config.TASK2_TARGET_COL]).diff()

    # 20d rolling volatility of log returns
    out["vol_20d"] = out["logret_1d"].rolling(20).std()

    # SMAs on adj_close
    out["sma_20"] = out[config.TASK2_TARGET_COL].rolling(20).mean()
    out["sma_60"] = out[config.TASK2_TARGET_COL].rolling(60).mean()

    # volume change
    out["vol_chg_1d"] = out["volume"].pct_change()

    return out


def build_feature_frame(df_asset: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with date + features + target (adj_close is also a feature).
    """
    df_asset = df_asset.sort_values(
        config.TASK2_DATE_COL).reset_index(drop=True)
    feat = df_asset.copy()

    if config.TASK2_USE_ENGINEERED_FEATURES:
        feat = make_engineered_features(feat)

    keep_cols = [config.TASK2_DATE_COL, config.TASK2_TARGET_COL] + \
        config.task2_feature_cols()
    keep_cols = list(dict.fromkeys(keep_cols))  # de-dup preserving order
    feat = feat[keep_cols].copy()

    # Drop rows with NaNs (rolling features)
    feat = feat.dropna().reset_index(drop=True)
    return feat


def chronological_split_2way(
    df_feat: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: date <= cutoff_date
    Test : date > cutoff_date
    """
    d = pd.to_datetime(df_feat[config.TASK2_DATE_COL])
    train = df_feat.loc[d <= cutoff_date].copy().reset_index(drop=True)
    test = df_feat.loc[d > cutoff_date].copy().reset_index(drop=True)

    if len(test) == 0:
        raise ValueError("Test split is empty. Check cutoff date logic.")
    return train, test


def chronological_split_3way(
    df_feat: pd.DataFrame,
    cutoff_date_val: pd.Timestamp,
    cutoff_date_test: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train: date <= cutoff_date_val
    Val  : cutoff_date_val < date <= cutoff_date_test
    Test : date > cutoff_date_test
    """
    if cutoff_date_val >= cutoff_date_test:
        raise ValueError(
            f"Invalid cutoffs: cutoff_date_val ({cutoff_date_val}) must be < cutoff_date_test ({cutoff_date_test})."
        )

    d = pd.to_datetime(df_feat[config.TASK2_DATE_COL])
    train = df_feat.loc[d <= cutoff_date_val].copy().reset_index(drop=True)
    val = df_feat.loc[(d > cutoff_date_val) & (
        d <= cutoff_date_test)].copy().reset_index(drop=True)
    test = df_feat.loc[d > cutoff_date_test].copy().reset_index(drop=True)

    if len(val) == 0:
        raise ValueError(
            "Validation split is empty. Check cutoff_date_val / feature dropna effects.")
    if len(test) == 0:
        raise ValueError("Test split is empty. Check cutoff_date_test.")
    return train, val, test


def save_splits_and_features(
    df_asset_raw: pd.DataFrame,
    split_year: int = config.TASK2_SPLIT_YEAR,
    val_year: Optional[int] = None,
) -> SplitInfo:
    """
    Produces raw and feature splits for Task 2.

    REQUIRED (recommended) artifacts if VAL is enabled:
      - data/task2/splits/tsla_train.parquet (raw)
      - data/task2/splits/tsla_val.parquet   (raw)
      - data/task2/splits/tsla_test.parquet  (raw)
      - data/task2/features/tsla_features_train.parquet
      - data/task2/features/tsla_features_val.parquet
      - data/task2/features/tsla_features_test.parquet
      - outputs/task2/metrics/split_info.json

    If config.TASK2_VAL_SPLIT_PATH does not exist in config, falls back to 2-way train/test.
    """
    _ensure_dir(config.TASK2_SPLITS_DIR)
    _ensure_dir(config.TASK2_FEATURES_DIR)
    _ensure_dir(config.TASK2_METRICS_DIR)

    df_asset_raw = df_asset_raw.sort_values(
        config.TASK2_DATE_COL).reset_index(drop=True)

    # Determine if we should produce a val split
    has_val_paths = hasattr(config, "TASK2_VAL_SPLIT_PATH")
    if val_year is None:
        # allow optional config override
        val_year = getattr(config, "TASK2_VAL_YEAR", split_year - 1)

    cutoff_test = find_last_trading_day_in_year(df_asset_raw, split_year)

    if has_val_paths:
        cutoff_val = find_last_trading_day_in_year(df_asset_raw, val_year)

        # --- raw 3-way split ---
        d_raw = pd.to_datetime(df_asset_raw[config.TASK2_DATE_COL])
        raw_train = df_asset_raw.loc[d_raw <=
                                     cutoff_val].copy().reset_index(drop=True)
        raw_val = df_asset_raw.loc[(d_raw > cutoff_val) & (
            d_raw <= cutoff_test)].copy().reset_index(drop=True)
        raw_test = df_asset_raw.loc[d_raw >
                                    cutoff_test].copy().reset_index(drop=True)

        if len(raw_val) == 0:
            raise ValueError("Raw VAL split is empty. Check val_year/cutoff.")
        if len(raw_test) == 0:
            raise ValueError(
                "Raw TEST split is empty. Check split_year/cutoff.")

        raw_train.to_parquet(config.TASK2_TRAIN_SPLIT_PATH, index=False)
        raw_val.to_parquet(config.TASK2_VAL_SPLIT_PATH, index=False)
        raw_test.to_parquet(config.TASK2_TEST_SPLIT_PATH, index=False)

        # --- feature 3-way split ---
        feat = build_feature_frame(df_asset_raw)
        feat_train, feat_val, feat_test = chronological_split_3way(
            feat, cutoff_val, cutoff_test)

        feat_train.to_parquet(
            f"{config.TASK2_FEATURES_DIR}/tsla_features_train.parquet", index=False)
        feat_val.to_parquet(
            f"{config.TASK2_FEATURES_DIR}/tsla_features_val.parquet", index=False)
        feat_test.to_parquet(
            f"{config.TASK2_FEATURES_DIR}/tsla_features_test.parquet", index=False)

        info = SplitInfo(
            asset=config.TASK2_ASSET,
            split_year=int(split_year),
            val_year=int(val_year),
            cutoff_date_val=str(pd.to_datetime(cutoff_val).date()),
            cutoff_date_test=str(pd.to_datetime(cutoff_test).date()),
            n_total=int(len(df_asset_raw)),
            n_train=int(len(raw_train)),
            n_val=int(len(raw_val)),
            n_test=int(len(raw_test)),
        )

    else:
        # --- raw 2-way split ---
        d_raw = pd.to_datetime(df_asset_raw[config.TASK2_DATE_COL])
        raw_train = df_asset_raw.loc[d_raw <=
                                     cutoff_test].copy().reset_index(drop=True)
        raw_test = df_asset_raw.loc[d_raw >
                                    cutoff_test].copy().reset_index(drop=True)

        raw_train.to_parquet(config.TASK2_TRAIN_SPLIT_PATH, index=False)
        raw_test.to_parquet(config.TASK2_TEST_SPLIT_PATH, index=False)

        # --- feature 2-way split ---
        feat = build_feature_frame(df_asset_raw)
        feat_train, feat_test = chronological_split_2way(feat, cutoff_test)

        feat_train.to_parquet(
            f"{config.TASK2_FEATURES_DIR}/tsla_features_train.parquet", index=False)
        feat_test.to_parquet(
            f"{config.TASK2_FEATURES_DIR}/tsla_features_test.parquet", index=False)

        info = SplitInfo(
            asset=config.TASK2_ASSET,
            split_year=int(split_year),
            val_year=int(val_year),
            cutoff_date_val=str(pd.to_datetime(cutoff_test).date()),
            cutoff_date_test=str(pd.to_datetime(cutoff_test).date()),
            n_total=int(len(df_asset_raw)),
            n_train=int(len(raw_train)),
            n_val=0,
            n_test=int(len(raw_test)),
        )

    with open(config.TASK2_SPLIT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info.__dict__, f, indent=2)

    return info
