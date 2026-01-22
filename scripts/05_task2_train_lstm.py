""" A script to train LSTM model for Task 2: Time Series Forecasting. """
# scripts/05_task2_train_lstm.py
from __future__ import annotations

import pandas as pd

from src import config
from src.task2_lstm import train_lstm_and_forecast, save_lstm_outputs


def main():
    """ Main function to train LSTM and save forecasts and model. """
    train_feat = pd.read_parquet(
        f"{config.TASK2_FEATURES_DIR}/tsla_features_train.parquet")
    test_feat = pd.read_parquet(
        f"{config.TASK2_FEATURES_DIR}/tsla_features_test.parquet")

    feature_cols = config.task2_feature_cols()

    forecast_df, info, model = train_lstm_and_forecast(
        train_feat=train_feat,
        test_feat=test_feat,
        feature_cols=feature_cols,
        target_col=config.TASK2_TARGET_COL,
        date_col=config.TASK2_DATE_COL,
    )

    save_lstm_outputs(forecast_df, info, model)

    print("Saved LSTM forecast + model + architecture.")
    print(info["metrics"])


if __name__ == "__main__":
    main()
