""" Configuration constants for Task 4. """
from pathlib import Path

ASSETS = ["TSLA", "SPY", "BND"]
DATE_COL = "date"
ASSET_COL = "asset"

# returns.parquet column name
RET_COL = "return"

TRADING_DAYS = 252
RISK_FREE_RATE_ANNUAL = 0.02

# Which TSLA forecast horizon to use for the "view"
TSLA_FORECAST_HORIZON = "6m"  # "6m" or "12m"

RETURNS_PATH = Path("data/task1/processed/returns.parquet")

TSLA_FORECAST_LOGRET_6M = Path(
    "outputs/task3/forecasts/tsla_logret_forecast_6m.csv")
TSLA_FORECAST_LOGRET_12M = Path(
    "outputs/task3/forecasts/tsla_logret_forecast_12m.csv")

OUT_DIR = Path("outputs/task4")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
SUMMARY_DIR = OUT_DIR / "summaries"


def ensure_output_dirs(repo_root: Path) -> None:
    """Ensure that all output directories exist."""
    for d in [repo_root / OUT_DIR, repo_root / FIG_DIR, repo_root / TABLE_DIR, repo_root / SUMMARY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
