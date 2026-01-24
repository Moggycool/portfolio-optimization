"""
Configuration constants for the financial analysis project.

Task 1: data extraction, cleaning, returns, stationarity, risk metrics
Task 2: TSLA time series forecasting (ARIMA/SARIMA + multivariate LSTM)

Key design choice (Task 2):
- Primary modeling + evaluation target is next-day log returns (logret_1d).
- Price (adj_close) is used for feature engineering and for reconstructing
  stakeholder-friendly price paths from predicted returns.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Global / Task 1 settings
# -----------------------------------------------------------------------------
TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-01-15"

PRICE_COL = "adj_close"          # Use adjusted close for returns/risk metrics
ANNUALIZATION_FACTOR = 252       # Trading days per year

RISK_FREE_RATE_ANNUAL = 0.02     # 2% annual (user choice)
DEFAULT_VAR_LEVEL = 0.95


# -----------------------------------------------------------------------------
# Paths (Task 1 + Task 2)
# NOTE: Your repo stores task1 artifacts under data/task1/processed (per user).
# -----------------------------------------------------------------------------
TASK1_PROCESSED_DIR = "data/task1/processed"
PRICES_PATH = f"{TASK1_PROCESSED_DIR}/prices.parquet"
RETURNS_PATH = f"{TASK1_PROCESSED_DIR}/returns.parquet"

TASK1_ADF_PATH = f"{TASK1_PROCESSED_DIR}/task1_adf_results.csv"
TASK1_RISK_PATH = f"{TASK1_PROCESSED_DIR}/task1_risk_metrics.csv"
TASK1_OUTLIERS_PATH = f"{TASK1_PROCESSED_DIR}/task1_outliers.csv"

# Scaling evidence + viz outputs (rubric-friendly)
TASK1_SCALED_PRICES_PATH = f"{TASK1_PROCESSED_DIR}/scaled_task1_prices.parquet"
TASK1_VIZ_DIR = "outputs/task1/viz"

# -----------------------------------------------------------------------------
# Task 2 directories
# -----------------------------------------------------------------------------
TASK2_DATA_DIR = "data/task2"
TASK2_SPLITS_DIR = f"{TASK2_DATA_DIR}/splits"
TASK2_FEATURES_DIR = f"{TASK2_DATA_DIR}/features"

TASK2_OUTPUTS_DIR = "outputs/task2"
TASK2_MODELS_DIR = f"{TASK2_OUTPUTS_DIR}/models"
TASK2_FORECASTS_DIR = f"{TASK2_OUTPUTS_DIR}/forecasts"
TASK2_METRICS_DIR = f"{TASK2_OUTPUTS_DIR}/metrics"
TASK2_FIGURES_DIR = f"{TASK2_OUTPUTS_DIR}/figures"

# -----------------------------------------------------------------------------
# Task 2: forecasting scope + column conventions
# -----------------------------------------------------------------------------
TASK2_ASSET = "TSLA"
TASK2_DATE_COL = "date"

# Critical decoupling:
# - price is used for features + reconstruction
# - target is used for modeling/evaluation (returns-primary)
TASK2_PRICE_COL = "adj_close"
TASK2_RETURN_COL = "logret_1d"
TASK2_TARGET_COL = TASK2_RETURN_COL

# Optional (nice for docs / downstream)
TASK2_TARGET_MODE = "log_return"

# Split rule:
# - TRAIN ends at last trading day in val_year
# - VAL ends at last trading day in split_year
# - TEST starts after split_year cutoff
TASK2_SPLIT_YEAR = 2024
TASK2_VAL_YEAR = 2023


# -----------------------------------------------------------------------------
# Task 2 split artifacts
# -----------------------------------------------------------------------------
TASK2_TRAIN_SPLIT_PATH = f"{TASK2_SPLITS_DIR}/tsla_train.parquet"
TASK2_VAL_SPLIT_PATH = f"{TASK2_SPLITS_DIR}/tsla_val.parquet"
TASK2_TEST_SPLIT_PATH = f"{TASK2_SPLITS_DIR}/tsla_test.parquet"
TASK2_SPLIT_INFO_PATH = f"{TASK2_METRICS_DIR}/split_info.json"

# Feature split paths (created by src/task2_data.py)
TASK2_FEATURES_TRAIN_PATH = f"{TASK2_FEATURES_DIR}/tsla_features_train.parquet"
TASK2_FEATURES_VAL_PATH = f"{TASK2_FEATURES_DIR}/tsla_features_val.parquet"
TASK2_FEATURES_TEST_PATH = f"{TASK2_FEATURES_DIR}/tsla_features_test.parquet"


# -----------------------------------------------------------------------------
# Task 2: multivariate features for LSTM
# -----------------------------------------------------------------------------
# Keep raw OHLCV + adj_close available for modeling.
TASK2_BASE_FEATURE_COLS = [
    "open", "high", "low", "close", "adj_close", "volume"
]

# Engineered features MUST be computed from TASK2_PRICE_COL (not target_col),
# so that switching target remains safe.
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
    # de-dup while preserving order
    return list(dict.fromkeys(cols))


# -----------------------------------------------------------------------------
# Task 2: ARIMA/SARIMA (pmdarima.auto_arima settings)
# -----------------------------------------------------------------------------
ARIMA_SEASONAL = False
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

# Forecast outputs:
# - Primary (returns)
TASK2_ARIMA_FORECAST_PATH = f"{TASK2_FORECASTS_DIR}/tsla_arima_forecast.csv"
TASK2_ARIMA_VAL_FORECAST_PATH = f"{TASK2_FORECASTS_DIR}/tsla_arima_forecast_val.csv"
# - Secondary (price reconstructed from predicted returns)
TASK2_ARIMA_FORECAST_PRICE_PATH = f"{TASK2_FORECASTS_DIR}/tsla_arima_forecast_price.csv"
TASK2_ARIMA_VAL_FORECAST_PRICE_PATH = f"{TASK2_FORECASTS_DIR}/tsla_arima_forecast_val_price.csv"


# -----------------------------------------------------------------------------
# Task 2: LSTM settings
# -----------------------------------------------------------------------------
TASK2_RANDOM_SEED = 42

LSTM_LOOKBACK = 60
LSTM_HORIZON = 1  # 1-step ahead

LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 1e-3

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
LSTM_DROPOUT = 0.2
LSTM_REC_DROPOUT = 0.0

# NOTE: We prefer explicit VAL split rather than validation_split, but we keep
# validation_split for internal training stability.
LSTM_VALIDATION_SPLIT = 0.1
LSTM_EARLY_STOPPING = True
LSTM_EARLY_STOPPING_PATIENCE = 5

LSTM_REDUCE_LR_ON_PLATEAU = True
LSTM_REDUCE_LR_PATIENCE = 3
LSTM_REDUCE_LR_FACTOR = 0.5
LSTM_MIN_LR = 1e-5

# Features scaled; target (log returns) NOT scaled by default (keeps interpretability)
LSTM_SCALER_TYPE = "minmax"  # "minmax" or "standard"

TASK2_LSTM_MODEL_PATH = f"{TASK2_MODELS_DIR}/lstm_model.keras"
TASK2_LSTM_ARCH_PATH = f"{TASK2_METRICS_DIR}/lstm_architecture.json"

# Forecast outputs:
# Keep existing filename for compatibility, but it contains returns (target_col).
TASK2_LSTM_FORECAST_PATH = f"{TASK2_FORECASTS_DIR}/tsla_lstm_forecast.csv"
TASK2_LSTM_VAL_FORECAST_PATH = f"{TASK2_FORECASTS_DIR}/tsla_lstm_forecast_val.csv"

# Reconstructed price paths from predicted returns
TASK2_LSTM_FORECAST_PRICE_PATH = f"{TASK2_FORECASTS_DIR}/tsla_lstm_forecast_price.csv"
TASK2_LSTM_VAL_FORECAST_PRICE_PATH = f"{TASK2_FORECASTS_DIR}/tsla_lstm_forecast_val_price.csv"


# -----------------------------------------------------------------------------
# Task 2: model comparison outputs
# -----------------------------------------------------------------------------
TASK2_FORECASTS_MERGED_PATH = f"{TASK2_FORECASTS_DIR}/tsla_forecasts_merged.csv"
TASK2_MODEL_COMPARISON_PATH = f"{TASK2_METRICS_DIR}/model_comparison.csv"
TASK2_ERROR_DIAGNOSTICS_PATH = f"{TASK2_METRICS_DIR}/error_diagnostics.csv"

TASK2_FORECAST_PLOT_PATH = f"{TASK2_FIGURES_DIR}/forecast_test_period.png"

# rename to reflect returns-primary (optional); keep old name if notebook expects it
TASK2_ACF_PACF_PLOT_PATH = f"{TASK2_FIGURES_DIR}/acf_pacf_logret_1d.png"


# -----------------------------------------------------------------------------
# Optional aliases (backward/forward compatibility)
# -----------------------------------------------------------------------------
# Some repos/notebooks use these shorter names; keep as aliases if helpful.
MODEL_COMPARISON_PATH = TASK2_MODEL_COMPARISON_PATH
ERROR_DIAGNOSTICS_PATH = TASK2_ERROR_DIAGNOSTICS_PATH

ARIMA_FORECAST_PATH = TASK2_ARIMA_FORECAST_PATH
LSTM_FORECAST_PATH = TASK2_LSTM_FORECAST_PATH
MERGED_FORECASTS_PATH = TASK2_FORECASTS_MERGED_PATH

FORECAST_PLOT_PATH = TASK2_FORECAST_PLOT_PATH

SPLIT_INFO_PATH = TASK2_SPLIT_INFO_PATH
ARIMA_PARAMS_PATH = TASK2_ARIMA_PARAMS_PATH
LSTM_ARCH_PATH = TASK2_LSTM_ARCH_PATH

# -----------------------------------------------------------------------------
# Clean up: remove conflicting/duplicate definitions from older drafts
# -----------------------------------------------------------------------------
# NOTE: Do not re-define TASK2_DATE_COL / TASK2_TARGET_COL again below this line.
# If you previously had:
#   TASK2_LSTM_RET_FORECAST_PATH / TASK2_LSTM_PRICE_FORECAST_PATH
# keep them as aliases to the canonical paths:
TASK2_LSTM_RET_FORECAST_PATH = TASK2_LSTM_FORECAST_PATH
TASK2_LSTM_PRICE_FORECAST_PATH = TASK2_LSTM_FORECAST_PRICE_PATH
