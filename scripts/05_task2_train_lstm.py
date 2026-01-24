""" A script to train LSTM model for Task 2: Time Series Forecasting (returns-primary). """
from __future__ import annotations

import os
import pandas as pd

from src import config
from src.task2_lstm import train_lstm_and_forecast, save_lstm_outputs


def main():
    """Main function to train LSTM and save forecasts and model."""
    # --- Load feature splits ---
    train_feat = pd.read_parquet(config.TASK2_FEATURES_TRAIN_PATH)

    # VAL is required for calibration; fail loudly if missing
    if not os.path.exists(config.TASK2_FEATURES_VAL_PATH):
        raise FileNotFoundError(
            f"Missing validation features: {config.TASK2_FEATURES_VAL_PATH}. "
            "Rerun the split/features generation script."
        )
    val_feat = pd.read_parquet(config.TASK2_FEATURES_VAL_PATH)

    test_feat = pd.read_parquet(config.TASK2_FEATURES_TEST_PATH)

    feature_cols = config.task2_feature_cols()

    # --- Train on TRAIN, forecast on VAL (for calibration fitting) ---
    val_ret_df, val_price_df, val_info, model = train_lstm_and_forecast(
        train_feat=train_feat,
        test_feat=val_feat,
        feature_cols=feature_cols,
        target_col=config.TASK2_TARGET_COL,
        price_col=config.TASK2_PRICE_COL,
        date_col=config.TASK2_DATE_COL,
    )

    # Save VAL outputs (do not overwrite standard TEST outputs)
    os.makedirs(config.TASK2_FORECASTS_DIR, exist_ok=True)

    save_lstm_outputs(
        forecast_ret_df=val_ret_df,
        forecast_price_df=val_price_df,
        info=val_info,
        model=model,
        forecast_path=getattr(config, "TASK2_LSTM_VAL_FORECAST_PATH", os.path.join(
            config.TASK2_FORECASTS_DIR, "tsla_lstm_forecast_val.csv")),
        price_forecast_path=getattr(config, "TASK2_LSTM_VAL_FORECAST_PRICE_PATH", os.path.join(
            config.TASK2_FORECASTS_DIR, "tsla_lstm_forecast_val_price.csv")),
        # For VAL run, do not overwrite the final model/arch unless you want to.
        # We'll still write them to the default paths so notebook can load architecture.
        arch_path=config.TASK2_LSTM_ARCH_PATH,
        model_path=config.TASK2_LSTM_MODEL_PATH,
    )

    print("Saved LSTM VAL forecasts (returns + price).")
    if isinstance(val_info, dict) and "metrics" in val_info:
        print("VAL metrics (returns):", val_info["metrics"])

    # --- Forecast on TEST (final evaluation) ---
    test_ret_df, test_price_df, test_info, model = train_lstm_and_forecast(
        train_feat=train_feat,
        test_feat=test_feat,
        feature_cols=feature_cols,
        target_col=config.TASK2_TARGET_COL,
        price_col=config.TASK2_PRICE_COL,
        date_col=config.TASK2_DATE_COL,
    )

    # Standard save for TEST (keeps your pipeline contract)
    save_lstm_outputs(
        forecast_ret_df=test_ret_df,
        forecast_price_df=test_price_df,
        info=test_info,
        model=model,
        forecast_path=config.TASK2_LSTM_FORECAST_PATH,
        price_forecast_path=config.TASK2_LSTM_FORECAST_PRICE_PATH,
        arch_path=config.TASK2_LSTM_ARCH_PATH,
        model_path=config.TASK2_LSTM_MODEL_PATH,
    )

    print("Saved LSTM TEST forecast + price reconstruction + model + architecture.")
    if isinstance(test_info, dict) and "metrics" in test_info:
        print("TEST metrics (returns):", test_info["metrics"])


if __name__ == "__main__":
    main()
