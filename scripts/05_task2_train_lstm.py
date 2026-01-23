""" A script to train LSTM model for Task 2: Time Series Forecasting. """
# scripts/05_task2_train_lstm.py
from __future__ import annotations

import os
import pandas as pd

from src import config
from src.task2_lstm import train_lstm_and_forecast, save_lstm_outputs


def main():
    """Main function to train LSTM and save forecasts and model."""
    # --- Load feature splits ---
    train_feat = pd.read_parquet(
        f"{config.TASK2_FEATURES_DIR}/tsla_features_train.parquet")

    # VAL is required for calibration; fail loudly if missing
    val_path = f"{config.TASK2_FEATURES_DIR}/tsla_features_val.parquet"
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Missing validation features: {val_path}. "
            "Update src/task2_data.py to generate train/val/test features and rerun 03_task2_make_splits_and_features.py."
        )
    val_feat = pd.read_parquet(val_path)

    test_feat = pd.read_parquet(
        f"{config.TASK2_FEATURES_DIR}/tsla_features_test.parquet")

    feature_cols = config.task2_feature_cols()

    # --- Train on TRAIN, forecast on VAL (for calibration fitting) ---
    val_forecast_df, val_info, model = train_lstm_and_forecast(
        train_feat=train_feat,
        test_feat=val_feat,  # reuse function: forecast on val
        feature_cols=feature_cols,
        target_col=config.TASK2_TARGET_COL,
        date_col=config.TASK2_DATE_COL,
    )

    # Save VAL forecast (separately, do not overwrite standard TEST outputs)
    os.makedirs(config.TASK2_FORECASTS_DIR, exist_ok=True)
    val_out_path = getattr(
        config,
        "TASK2_LSTM_VAL_FORECAST_PATH",
        os.path.join(config.TASK2_FORECASTS_DIR, "tsla_lstm_forecast_val.csv"),
    )
    val_forecast_df.to_csv(val_out_path, index=False)
    print("Saved LSTM VAL forecast:", val_out_path)
    if isinstance(val_info, dict) and "metrics" in val_info:
        print("VAL metrics:", val_info["metrics"])

    # --- Forecast on TEST (final evaluation) ---
    test_forecast_df, test_info, model = train_lstm_and_forecast(
        train_feat=train_feat,
        test_feat=test_feat,
        feature_cols=feature_cols,
        target_col=config.TASK2_TARGET_COL,
        date_col=config.TASK2_DATE_COL,
    )

    # Standard save for TEST (keeps your pipeline contract)
    save_lstm_outputs(test_forecast_df, test_info, model)
    print("Saved LSTM TEST forecast + model + architecture.")
    if isinstance(test_info, dict) and "metrics" in test_info:
        print("TEST metrics:", test_info["metrics"])


if __name__ == "__main__":
    main()
