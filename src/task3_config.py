"# Configuration for Task 3: Forecast Evaluation and Visualization"
from pathlib import Path

# --- Core settings ---
ASSET = "TSLA"
DATE_COL = "date"
ASSET_COL = "asset"

# Long-format parquet columns
PRICE_COL = "adj_close"           # from prices.parquet
RET_COL_FALLBACK = "logret_1d"    # fallback if spec doesn't provide

# Forecast horizons (trading days approximation)
HORIZONS = {"6m": 126, "12m": 252}

# 95% prediction interval
ALPHA = 0.05

# Monte Carlo simulations for price fan chart
N_SIMS = 5000
RANDOM_SEED = 42

# --- Repo-relative paths ---
PRICES_PATH = Path("data/task1/processed/prices.parquet")
RETURNS_PATH = Path("data/task1/processed/returns.parquet")

ARIMA_SPEC_VAL_PATH = Path("outputs/task2/metrics/arima_params_val.json")
ARIMA_SPEC_PATH = Path("outputs/task2/metrics/arima_params.json")
SPLIT_INFO_PATH = Path("outputs/task2/metrics/split_info.json")  # optional

# --- Task 2 forecast artifacts (for test overlay) ---
# Prefer merged (often has actual + arima + lstm in one file)
TASK2_MERGED_FORECASTS_PATH = Path(
    "outputs/task2/forecasts/tsla_forecasts_merged.csv")

# If merged not available, fall back to ARIMA price forecasts
TASK2_ARIMA_TEST_FORECAST_PATH = Path(
    "outputs/task2/forecasts/tsla_arima_forecast.csv")
TASK2_ARIMA_VAL_FORECAST_PATH = Path(
    "outputs/task2/forecasts/tsla_arima_forecast_val.csv")

# Optional LSTM price-path forecasts (if you want overlay)
TASK2_LSTM_TEST_PRICE_FORECAST_PATH = Path(
    "outputs/task2/forecasts/tsla_lstm_forecast_price.csv")
TASK2_LSTM_VAL_PRICE_FORECAST_PATH = Path(
    "outputs/task2/forecasts/tsla_lstm_forecast_val_price.csv")

# We will try to infer these columns automatically in the loader:
FORECAST_DATE_CANDIDATES = ["date", "ds", "Date"]
ACTUAL_PRICE_CANDIDATES = [
    "actual_price", "y_true_price", "actual", "true", "adj_close", "price"]
ARIMA_PRICE_CANDIDATES = ["arima_price", "arima_pred_price",
                          "arima_pred", "yhat_arima", "arima_forecast_price"]
LSTM_PRICE_CANDIDATES = ["lstm_price", "lstm_pred_price",
                         "lstm_pred", "yhat_lstm", "lstm_forecast_price"]

# --- Outputs ---
OUT_DIR = Path("outputs/task3")
FORECAST_DIR = OUT_DIR / "forecasts"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_DIR = OUT_DIR / "summaries"


def ensure_output_dirs(repo_root: Path) -> None:
    for d in [repo_root / OUT_DIR, repo_root / FORECAST_DIR, repo_root / FIG_DIR, repo_root / SUMMARY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
