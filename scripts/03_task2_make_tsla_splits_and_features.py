""" A script to create train/val/test splits and features for Task 2: Time Series Forecasting. """
# scripts/03_task2_make_splits_and_features.py
from __future__ import annotations

import os

from src.task2_data import load_prices, filter_asset, save_splits_and_features
from src import config


def _assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected artifact not found: {path}")


def main():
    """Main function to create and save splits and features for Task 2."""
    df = load_prices(config.PRICES_PATH)
    tsla = filter_asset(df, config.TASK2_ASSET)

    # IMPORTANT:
    # save_splits_and_features must be updated in src/task2_data.py to create
    # train/val/test and write TASK2_TRAIN_SPLIT_PATH, TASK2_VAL_SPLIT_PATH, TASK2_TEST_SPLIT_PATH
    info = save_splits_and_features(tsla, split_year=config.TASK2_SPLIT_YEAR)

    # Verify artifacts exist (fail fast if split builder wasn't updated)
    _assert_exists(config.TASK2_TRAIN_SPLIT_PATH)
    _assert_exists(config.TASK2_TEST_SPLIT_PATH)

    # val is now REQUIRED for proper calibration / model selection
    if not hasattr(config, "TASK2_VAL_SPLIT_PATH"):
        raise AttributeError(
            "Missing config.TASK2_VAL_SPLIT_PATH. Add it in src/config.py (recommended)."
        )
    _assert_exists(config.TASK2_VAL_SPLIT_PATH)

    print("Saved Task 2 train/val/test splits & features.")
    print(info)


if __name__ == "__main__":
    main()
