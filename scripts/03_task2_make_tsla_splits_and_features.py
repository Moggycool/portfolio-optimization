""" A script to create train/test splits and features for Task 2: Time Series Forecasting. """
# scripts/03_task2_make_splits_and_features.py
from __future__ import annotations

from src.task2_data import load_prices, filter_asset, save_splits_and_features
from src import config


def main():
    """ Main function to create and save splits and features for Task 2. """
    df = load_prices(config.PRICES_PATH)
    tsla = filter_asset(df, config.TASK2_ASSET)

    info = save_splits_and_features(tsla, split_year=config.TASK2_SPLIT_YEAR)
    print("Saved Task 2 splits & features.")
    print(info)


if __name__ == "__main__":
    main()
