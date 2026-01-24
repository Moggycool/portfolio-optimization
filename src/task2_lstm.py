""" LSTM modeling for Task 2: Time Series Forecasting (returns-primary). """
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple, Dict, List, Optional

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
    price_col: str
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


def get_x_scaler(scaler_type: str):
    """Returns an X scaler based on scaler_type."""
    st = scaler_type.lower()
    if st == "minmax":
        return MinMaxScaler()
    if st == "standard":
        return StandardScaler()
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


def reconstruct_price_from_logrets(p0: float, logrets: np.ndarray) -> np.ndarray:
    """P_t = p0 * exp(cumsum(logrets))"""
    r = np.asarray(logrets, dtype=float).reshape(-1)
    return float(p0) * np.exp(np.cumsum(r))


def _add_next_day_target(df_feat: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Creates supervised next-day target y from target_col.
    If target_col is logret_1d, then:
      y_t = logret_1d_{t+1}
    """
    out = df_feat.sort_values(
        config.TASK2_DATE_COL).reset_index(drop=True).copy()
    out["y"] = out[target_col].shift(-1)
    return out


def train_lstm_and_forecast(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = config.TASK2_TARGET_COL,
    price_col: str = config.TASK2_PRICE_COL,
    date_col: str = config.TASK2_DATE_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, object]:
    """
    Train on train_feat, forecast next-day returns for the provided test_feat period.

    Returns:
      forecast_ret_df: [date, y_true, lstm_pred]   # returns space (official scoring)
      forecast_price_df: [date, p0, y_true_price, y_pred_price]  # reconstructed path
      info: dict (architecture & metrics in returns space)
      model: trained keras model
    """
    _set_seeds(config.TASK2_RANDOM_SEED)

    # --- Build supervised y (next day) ---
    train_sup = _add_next_day_target(
        train_feat, target_col=target_col).dropna().reset_index(drop=True)
    test_sup = _add_next_day_target(
        test_feat, target_col=target_col).dropna().reset_index(drop=True)

    # Ensure price column exists for reconstruction
    if price_col not in train_sup.columns or price_col not in test_sup.columns:
        raise ValueError(
            f"Missing price_col={price_col} in feature frames. Ensure task2_data keeps it.")

    # --- Prepare arrays ---
    X_train_raw = train_sup[feature_cols].astype(float).values
    y_train = train_sup["y"].astype(float).values  # returns (no y-scaler)

    X_test_raw = test_sup[feature_cols].astype(float).values
    y_test_true = test_sup["y"].astype(float).values

    # --- Fit X scaler on TRAIN only ---
    x_scaler = get_x_scaler(config.LSTM_SCALER_TYPE)
    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)

    # Sequence forecasting uses appended context (train + test) for window availability.
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test_true])

    lookback = int(config.LSTM_LOOKBACK)
    horizon = int(config.LSTM_HORIZON)

    X_seq, y_seq = make_sequences(X_all, y_all, lookback, horizon)

    # Mapping from sequence index -> original row index of y (in train_sup+test_sup)
    first_target_idx = lookback + horizon - 1
    target_indices = np.arange(first_target_idx, first_target_idx + len(y_seq))

    train_last_idx = len(train_sup) - 1
    test_mask = target_indices > train_last_idx

    X_train_seq = X_seq[~test_mask]
    y_train_seq = y_seq[~test_mask]
    X_test_seq = X_seq[test_mask]
    y_true = y_seq[test_mask]

    # Dates for predicted targets
    all_dates = pd.to_datetime(
        pd.concat([train_sup[date_col], test_sup[date_col]],
                  axis=0).reset_index(drop=True)
    )
    pred_dates = all_dates.iloc[target_indices[test_mask]].values

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

    # --- Predict returns ---
    y_pred = model.predict(X_test_seq, verbose=0).reshape(-1).astype(float)

    forecast_ret_df = pd.DataFrame(
        {
            date_col: pd.to_datetime(pred_dates),
            "y_true": y_true.astype(float),
            "lstm_pred": y_pred,
        }
    )

    # --- Price reconstruction (secondary reporting) ---
    # We need p0: last observed price BEFORE the first predicted return date.
    # pred_dates correspond to target_indices[test_mask] in concatenated (train_sup+test_sup).
    # Let first_pred_global_idx be that first target index; p0 is price at (first_pred_global_idx - 1)
    # because return at t is log(P_t) - log(P_{t-1}).
    feat_all = pd.concat([train_sup, test_sup], axis=0).reset_index(drop=True)
    first_pred_global_idx = int(target_indices[test_mask][0])
    p0 = float(feat_all[price_col].iloc[first_pred_global_idx - 1])

    y_true_price = reconstruct_price_from_logrets(
        p0=p0, logrets=forecast_ret_df["y_true"].values)
    y_pred_price = reconstruct_price_from_logrets(
        p0=p0, logrets=forecast_ret_df["lstm_pred"].values)

    forecast_price_df = pd.DataFrame(
        {
            date_col: forecast_ret_df[date_col].values,
            "p0": p0,
            "y_true_price": y_true_price,
            "y_pred_price": y_pred_price,
        }
    )

    run_info = LSTMRunInfo(
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        price_col=price_col,
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
        # Metrics are computed in RETURNS space (primary)
        "metrics": all_metrics(forecast_ret_df["y_true"], forecast_ret_df["lstm_pred"]),
    }

    return forecast_ret_df, forecast_price_df, info, model


def save_lstm_outputs(
    forecast_ret_df: pd.DataFrame,
    forecast_price_df: pd.DataFrame,
    info: Dict,
    model,
    forecast_path: Optional[str] = None,
    arch_path: Optional[str] = None,
    model_path: Optional[str] = None,
    price_forecast_path: Optional[str] = None,
) -> None:
    """Saves forecast CSVs (returns + price), run info JSON, and the Keras model."""
    _ensure_dir(config.TASK2_FORECASTS_DIR)
    _ensure_dir(config.TASK2_METRICS_DIR)
    _ensure_dir(config.TASK2_MODELS_DIR)

    forecast_path = forecast_path or config.TASK2_LSTM_FORECAST_PATH
    arch_path = arch_path or config.TASK2_LSTM_ARCH_PATH
    model_path = model_path or config.TASK2_LSTM_MODEL_PATH
    price_forecast_path = price_forecast_path or config.TASK2_LSTM_FORECAST_PRICE_PATH

    forecast_ret_df.to_csv(forecast_path, index=False)
    forecast_price_df.to_csv(price_forecast_path, index=False)

    with open(arch_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    model.save(model_path)
