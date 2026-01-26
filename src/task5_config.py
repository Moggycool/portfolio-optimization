from pathlib import Path

ASSETS = ["TSLA", "SPY", "BND"]
DATE_COL = "date"
ASSET_COL = "asset"
RET_COL = "return"

TRADING_DAYS = 252
RISK_FREE_RATE_ANNUAL = 0.02

# (a) Backtesting Period: Jan 2025 - Jan 2026 (out-of-sample)
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2026-01-31"   # inclusive

# (b) Benchmark: static 60/40
BENCHMARK_WEIGHTS = {"SPY": 0.60, "BND": 0.40, "TSLA": 0.0}

# (c) Strategy Simulation
# - "hold": hold initial weights entire backtest
# - "monthly": rebalance monthly (hold weights ~1 month between rebalances)
REBALANCE_MODE = "monthly"

# Optional cost model (keep 0 unless asked)
TRANSACTION_COST_BPS = 0

# Inputs
RETURNS_PATH = Path("data/task1/processed/returns.parquet")
TASK4_SUMMARY_PATH = Path("outputs/task4/task4_summary.json")
SPLIT_INFO_PATH = Path(
    "outputs/task2/metrics/split_info.json")  # optional evidence

# Outputs
OUT_DIR = Path("outputs/task5")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
SUMMARY_DIR = OUT_DIR / "summaries"


def ensure_output_dirs(repo_root: Path) -> None:
    for d in [repo_root / OUT_DIR, repo_root / FIG_DIR, repo_root / TABLE_DIR, repo_root / SUMMARY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
