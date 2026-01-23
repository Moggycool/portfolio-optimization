""" LSTM modeling for Task 2: Time Series Forecasting. """
# src/task2_lstm.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import config
from src.task2_metrics import all_metrics


def _ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _set_seeds(seed: int) -> None:
    """Best-effort reproducibility across numpy / tensorflow."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


@dataclass
class LSTMRunInfo:
    lookback: int
    horizon: int
    feature_cols: List[str]
    target_col: str
    scaler_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    units_1: int
    units_2: int
    dropout: float
    rec_dropout: float


def make_sequences(
    X: np.ndarray, y: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (n, n_features)
    y: (n,)
    Returns:
      X_seq: (n_samples, lookback, n_features)
      y_seq: (n_samples,)
    Predicts y at t + horizon - 1 using previous lookback steps ending at t-1.
    """
    Xs, ys = [], []
    n = len(X)
    for end in range(lookback, n - horizon + 1):
        start = end - lookback
        Xs.append(X[start:end, :])
        ys.append(y[end + horizon - 1])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def get_scalers(scaler_type: str):
    """Returns tuple of (X_scaler, y_scaler) based on scaler_type."""
    st = scaler_type.lower()
    if st == "minmax":
        return MinMaxScaler(), MinMaxScaler()
    if st == "standard":
        return StandardScaler(), StandardScaler()
    raise ValueError(
        f"Unknown scaler_type={scaler_type}. Use 'minmax' or 'standard'.")


def build_lstm_model(input_shape: Tuple[int, int]):
    """input_shape = (lookback, n_features)"""
    try:
        from tensorflow.keras import layers, models, optimizers
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for LSTM. Install: pip install tensorflow") from e

    lookback, n_features = input_shape

    model = models.Sequential()
    model.add(layers.Input(shape=(lookback, n_features)))

    return_sequences = bool(config.LSTM_UNITS_2 and config.LSTM_UNITS_2 > 0)
    model.add(
        layers.LSTM(
            config.LSTM_UNITS_1,
            return_sequences=return_sequences,
            dropout=config.LSTM_DROPOUT,
            recurrent_dropout=config.LSTM_REC_DROPOUT,
        )
    )

    if config.LSTM_UNITS_2 and config.LSTM_UNITS_2 > 0:
        model.add(
            layers.LSTM(
                config.LSTM_UNITS_2,
                return_sequences=False,
                dropout=config.LSTM_DROPOUT,
                recurrent_dropout=config.LSTM_REC_DROPOUT,
            )
        )

    model.add(layers.Dense(1))

    opt = optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse")
    return model


def train_lstm_and_forecast(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = config.TASK2_TARGET_COL,
    date_col: str = config.TASK2_DATE_COL,
) -> Tuple[pd.DataFrame, Dict, object]:
    """
    Train on train_feat, forecast next-day for the provided test_feat period using sliding windows.
    Returns:
      forecast_df: [date, y_true, lstm_pred]
      info: dict (architecture & metrics)
      model: trained keras model
    """
    _set_seeds(config.TASK2_RANDOM_SEED)

    # --- Prepare arrays ---
    X_train_raw = train_feat[feature_cols].astype(float).values
    y_train_raw = train_feat[target_col].astype(float).values.reshape(-1, 1)

    X_test_raw = test_feat[feature_cols].astype(float).values
    y_test_raw = test_feat[target_col].astype(float).values.reshape(-1, 1)

    # --- Fit scalers on TRAIN only ---
    x_scaler, y_scaler = get_scalers(config.LSTM_SCALER_TYPE)
    X_train = x_scaler.fit_transform(X_train_raw)
    y_train = y_scaler.fit_transform(y_train_raw).reshape(-1)

    # --- Transform test features; keep y_true in original scale for metrics ---
    X_test = x_scaler.transform(X_test_raw)
    y_test_true = y_test_raw.reshape(-1)

    # Sequence forecasting uses appended context (train + test) for window availability.
    X_all = np.vstack([X_train, X_test])
    y_all_scaled = np.concatenate(
        [y_train, y_scaler.transform(y_test_raw).reshape(-1)])

    lookback = int(config.LSTM_LOOKBACK)
    horizon = int(config.LSTM_HORIZON)

    X_seq, y_seq = make_sequences(X_all, y_all_scaled, lookback, horizon)

    # Each y_seq element corresponds to an original (train+test) index:
    # target_idx = lookback + horizon - 1 + seq_idx
    first_target_idx = lookback + horizon - 1
    target_indices = np.arange(first_target_idx, first_target_idx + len(y_seq))

    # Targets strictly AFTER the last train index belong to test forecast evaluation
    train_last_idx = len(train_feat) - 1
    test_mask = target_indices > train_last_idx

    X_train_seq = X_seq[~test_mask]
    y_train_seq = y_seq[~test_mask]

    X_test_seq = X_seq[test_mask]

    # Dates for predicted targets
    all_dates = pd.to_datetime(
        pd.concat([train_feat[date_col], test_feat[date_col]],
                  axis=0).reset_index(drop=True)
    )
    test_dates = all_dates.iloc[target_indices[test_mask]].values

    # True y for those dates in ORIGINAL scale
    y_all_true = np.concatenate(
        [train_feat[target_col].values, test_feat[target_col].values])
    y_true = y_all_true[target_indices[test_mask]].astype(float)

    # --- Build + train model ---
    model = build_lstm_model(input_shape=(lookback, len(feature_cols)))

    callbacks = []
    try:
        import tensorflow as tf
    except Exception:
        tf = None

    if tf is not None and config.LSTM_EARLY_STOPPING:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.LSTM_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        )

    if tf is not None and config.LSTM_REDUCE_LR_ON_PLATEAU:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.LSTM_REDUCE_LR_FACTOR,
                patience=config.LSTM_REDUCE_LR_PATIENCE,
                min_lr=config.LSTM_MIN_LR,
            )
        )

    history = model.fit(
        X_train_seq,
        y_train_seq,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        validation_split=config.LSTM_VALIDATION_SPLIT,
        shuffle=False,  # time series
        callbacks=callbacks if callbacks else None,
        verbose=1,
    )

    # --- Predict (scaled -> invert to original price scale) ---
    y_pred_scaled = model.predict(X_test_seq, verbose=0).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(
        y_pred_scaled).reshape(-1).astype(float)

    forecast_df = pd.DataFrame(
        {
            date_col: pd.to_datetime(test_dates),
            "y_true": y_true,
            "lstm_pred": y_pred,
        }
    )

    run_info = LSTMRunInfo(
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler_type=config.LSTM_SCALER_TYPE,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        learning_rate=config.LSTM_LEARNING_RATE,
        units_1=config.LSTM_UNITS_1,
        units_2=int(config.LSTM_UNITS_2),
        dropout=config.LSTM_DROPOUT,
        rec_dropout=config.LSTM_REC_DROPOUT,
    )

    info = {
        "run": asdict(run_info),
        "history_last": {
            "loss": float(history.history["loss"][-1]) if "loss" in history.history else None,
            "val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        },
        "metrics": all_metrics(forecast_df["y_true"], forecast_df["lstm_pred"]),
    }

    return forecast_df, info, model


def save_lstm_outputs(forecast_df: pd.DataFrame, info: Dict, model) -> None:
    """Saves forecast CSV, run info JSON, and the Keras model."""
    _ensure_dir(config.TASK2_FORECASTS_DIR)
    _ensure_dir(config.TASK2_METRICS_DIR)
    _ensure_dir(config.TASK2_MODELS_DIR)

    forecast_df.to_csv(config.TASK2_LSTM_FORECAST_PATH, index=False)

    with open(config.TASK2_LSTM_ARCH_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # Save model (.keras)
    model.save(config.TASK2_LSTM_MODEL_PATH)
