""" A script to train ARIMA model for Task 2: Time Series Forecasting. """
# scripts/04_task2_train_arima.py
from __future__ import annotations

import pandas as pd

from src import config
from src.task2_arima import arima_forecast, save_arima_outputs


def main():
    """ Main function to train ARIMA and save forecasts and parameters. """
    train_df = pd.read_parquet(config.TASK2_TRAIN_SPLIT_PATH)
    test_df = pd.read_parquet(config.TASK2_TEST_SPLIT_PATH)

    forecast_df, params = arima_forecast(train_df, test_df)
    # set to model if you want pickle
    save_arima_outputs(forecast_df, params, save_model=None)

    print("Saved ARIMA forecast + params.")
    print(params["metrics"])


if __name__ == "__main__":
    main()
