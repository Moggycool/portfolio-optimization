"""
Configuration constants for the financial analysis project.

Task 1: data extraction, cleaning, returns, stationarity, risk metrics
Task 2: TSLA time series forecasting (ARIMA/SARIMA + multivariate LSTM)
"""
# src/config.py
from __future__ import annotations


# -----------------------------------------------------------------------------
# Global / Task 1 settings (existing)
# -----------------------------------------------------------------------------
TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-01-15"

PRICE_COL = "adj_close"          # Use adjusted close for returns/risk metrics
ANNUALIZATION_FACTOR = 252       # Trading days per year

RISK_FREE_RATE_ANNUAL = 0.02     # 2% annual (user choice)
DEFAULT_VAR_LEVEL = 0.95


# -----------------------------------------------------------------------------
# Paths (existing + Task 2 additions)
# -----------------------------------------------------------------------------
# Task 1 processed paths (existing)
PROCESSED_DIR = "data/processed"
PRICES_PATH = f"{PROCESSED_DIR}/prices.parquet"
RETURNS_PATH = f"{PROCESSED_DIR}/returns.parquet"

TASK1_ADF_PATH = f"{PROCESSED_DIR}/task1_adf_results.csv"
TASK1_RISK_PATH = f"{PROCESSED_DIR}/task1_risk_metrics.csv"
TASK1_OUTLIERS_PATH = f"{PROCESSED_DIR}/task1_outliers.csv"

# Task 2 data folders (you created these)
TASK2_DATA_DIR = "data/task2"
TASK2_SPLITS_DIR = f"{TASK2_DATA_DIR}/splits"
TASK2_FEATURES_DIR = f"{TASK2_DATA_DIR}/features"

# Task 2 outputs (you created these)
TASK2_OUTPUTS_DIR = "outputs/task2"
TASK2_MODELS_DIR = f"{TASK2_OUTPUTS_DIR}/models"
TASK2_FORECASTS_DIR = f"{TASK2_OUTPUTS_DIR}/forecasts"
TASK2_METRICS_DIR = f"{TASK2_OUTPUTS_DIR}/metrics"
TASK2_FIGURES_DIR = f"{TASK2_OUTPUTS_DIR}/figures"


# -----------------------------------------------------------------------------
# Task 2: forecasting scope
# -----------------------------------------------------------------------------
TASK2_ASSET = "TSLA"
TASK2_DATE_COL = "date"
TASK2_TARGET_COL = "adj_close"   # per your decision

# Split rule: train ends on the last *trading day* present in year 2024 for TSLA
TASK2_SPLIT_YEAR = 2024

# Persisted split artifacts (recommended)
TASK2_TRAIN_SPLIT_PATH = f"{TASK2_SPLITS_DIR}/tsla_train.parquet"
TASK2_TEST_SPLIT_PATH = f"{TASK2_SPLITS_DIR}/tsla_test.parquet"
TASK2_SPLIT_INFO_PATH = f"{TASK2_METRICS_DIR}/split_info.json"


# -----------------------------------------------------------------------------
# Task 2: multivariate features for LSTM
# -----------------------------------------------------------------------------
TASK2_BASE_FEATURE_COLS = [
    "open", "high", "low", "close", "adj_close", "volume"
]

# Optional engineered features (your feature builder should create these names)
TASK2_ENGINEERED_FEATURES = [
    "ret_1d",
    "logret_1d",
    "vol_20d",
    "sma_20",
    "sma_60",
    "vol_chg_1d",
]

TASK2_USE_ENGINEERED_FEATURES = True


def task2_feature_cols() -> list[str]:
    """Return the list of feature columns to use for Task 2."""
    cols = list(TASK2_BASE_FEATURE_COLS)
    if TASK2_USE_ENGINEERED_FEATURES:
        cols += list(TASK2_ENGINEERED_FEATURES)
    return cols


# -----------------------------------------------------------------------------
# Task 2: ARIMA/SARIMA (pmdarima.auto_arima settings)
# -----------------------------------------------------------------------------
ARIMA_SEASONAL = False     # keep False unless you explicitly justify seasonality
# used only if ARIMA_SEASONAL=True (weekly trading pattern)
ARIMA_M = 5

ARIMA_START_P = 0
ARIMA_START_Q = 0
ARIMA_MAX_P = 5
ARIMA_MAX_Q = 5
ARIMA_MAX_D = 2

ARIMA_START_P_SEASONAL = 0
ARIMA_START_Q_SEASONAL = 0
ARIMA_MAX_P_SEASONAL = 2
ARIMA_MAX_Q_SEASONAL = 2
ARIMA_MAX_D_SEASONAL = 1

ARIMA_STEPWISE = True
ARIMA_TRACE = True
ARIMA_SUPPRESS_WARNINGS = True
ARIMA_ERROR_ACTION = "ignore"

TASK2_ARIMA_MODEL_PATH = f"{TASK2_MODELS_DIR}/arima_model.pkl"
TASK2_ARIMA_PARAMS_PATH = f"{TASK2_METRICS_DIR}/arima_params.json"
TASK2_ARIMA_FORECAST_PATH = f"{TASK2_FORECASTS_DIR}/tsla_arima_forecast.csv"


# -----------------------------------------------------------------------------
# Task 2: LSTM settings
# -----------------------------------------------------------------------------
TASK2_RANDOM_SEED = 42

LSTM_LOOKBACK = 60         # last 60 trading days -> predict next day
LSTM_HORIZON = 1

LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 1e-3

# Baseline architecture
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32          # set to 0 to disable second LSTM layer
LSTM_DROPOUT = 0.2
LSTM_REC_DROPOUT = 0.0

# Training controls
LSTM_VALIDATION_SPLIT = 0.1
LSTM_EARLY_STOPPING = True
LSTM_EARLY_STOPPING_PATIENCE = 5

LSTM_REDUCE_LR_ON_PLATEAU = True
LSTM_REDUCE_LR_PATIENCE = 3
LSTM_REDUCE_LR_FACTOR = 0.5
LSTM_MIN_LR = 1e-5

# Scaling
LSTM_SCALER_TYPE = "minmax"   # "minmax" or "standard"

TASK2_LSTM_MODEL_PATH = f"{TASK2_MODELS_DIR}/lstm_model.keras"
TASK2_LSTM_ARCH_PATH = f"{TASK2_METRICS_DIR}/lstm_architecture.json"
TASK2_LSTM_FORECAST_PATH = f"{TASK2_FORECASTS_DIR}/tsla_lstm_forecast.csv"


# -----------------------------------------------------------------------------
# Task 2: model comparison outputs
# -----------------------------------------------------------------------------
TASK2_FORECASTS_MERGED_PATH = f"{TASK2_FORECASTS_DIR}/tsla_forecasts_merged.csv"
TASK2_MODEL_COMPARISON_PATH = f"{TASK2_METRICS_DIR}/model_comparison.csv"

TASK2_FORECAST_PLOT_PATH = f"{TASK2_FIGURES_DIR}/forecast_test_period.png"
TASK2_ACF_PACF_PLOT_PATH = f"{TASK2_FIGURES_DIR}/acf_pacf_adj_close.png"
