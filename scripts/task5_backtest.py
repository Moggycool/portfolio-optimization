"""Backtesting script for Task 5.

Supports running as:
- python -m scripts.task5_backtest
- python scripts\task5_backtest.py
"""
from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root(start: Path | None = None) -> Path:
    """Auto-detect the repo root by looking for 'src' and 'data' directories."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        "Could not auto-detect repo root. Run from inside the repo.")


def main():
    """Main function to run backtest and generate outputs."""
    repo = find_repo_root(Path(__file__).resolve())

    # Ensure repo root is on sys.path so `import src...` works when running as a file path.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Local imports (after sys.path bootstrap)
    import pandas as pd
    import src.task5_config as cfg
    from src.task5_data import (
        load_parquet,
        pivot_returns_long_to_wide,
        slice_backtest_window,
        load_json,
    )
    from src.task5_strategy import (
        simulate_portfolio_hold,
        simulate_portfolio_monthly_rebalance,
    )
    from src.task5_metrics import (
        cumulative_from_returns,
        total_return,
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        max_drawdown,
    )
    from src.task5_plotting import plot_cumulative_returns
    from src.task5_report import write_text, write_json, conclusion_text

    def metrics_block(daily_returns: pd.Series) -> dict:
        """Compute performance metrics block from daily returns."""
        cum = cumulative_from_returns(daily_returns)
        return {
            "total_return": total_return(cum),
            "annualized_return": annualized_return(cum, trading_days=cfg.TRADING_DAYS),
            "annualized_volatility": annualized_volatility(
                daily_returns, trading_days=cfg.TRADING_DAYS
            ),
            "sharpe": sharpe_ratio(
                daily_returns,
                rf_annual=cfg.RISK_FREE_RATE_ANNUAL,
                trading_days=cfg.TRADING_DAYS,
            ),
            "max_drawdown": max_drawdown(cum),
        }

    cfg.ensure_output_dirs(repo)

    # (a) Backtesting window (out-of-sample)
    df_ret = load_parquet(repo / cfg.RETURNS_PATH)
    rets_wide = pivot_returns_long_to_wide(
        df_ret,
        cfg.DATE_COL,
        cfg.ASSET_COL,
        cfg.RET_COL,
        cfg.ASSETS,
    )
    bt = slice_backtest_window(rets_wide, cfg.BACKTEST_START, cfg.BACKTEST_END)
    if bt.empty:
        raise ValueError(
            "Backtest window produced empty dataframe. Check dates and data coverage."
        )

    # Strategy initial weights from Task 4
    task4 = load_json(repo / cfg.TASK4_SUMMARY_PATH)
    strategy_weights = task4["recommended"]["weights"]

    # (b) Benchmark
    bench_weights = cfg.BENCHMARK_WEIGHTS

    # (c) Strategy simulation
    if cfg.REBALANCE_MODE == "hold":
        strat_ret = simulate_portfolio_hold(
            bt,
            strategy_weights,
            transaction_cost_bps=cfg.TRANSACTION_COST_BPS,
        )
        turnover = None
        reb_dates = None
    elif cfg.REBALANCE_MODE == "monthly":
        strat_ret, turnover, reb_dates = simulate_portfolio_monthly_rebalance(
            bt,
            strategy_weights,
            transaction_cost_bps=cfg.TRANSACTION_COST_BPS,
        )
    else:
        raise ValueError("Unknown REBALANCE_MODE. Use 'hold' or 'monthly'.")

    bench_ret = simulate_portfolio_hold(
        bt,
        bench_weights,
        transaction_cost_bps=0.0,
    ).rename("benchmark_return")

    # (d) Cumulative returns plot
    strat_cum = cumulative_from_returns(strat_ret).rename("strategy_cum")
    bench_cum = cumulative_from_returns(bench_ret).rename("benchmark_cum")
    plot_cumulative_returns(
        strat_cum,
        bench_cum,
        out_path=str(repo / cfg.FIG_DIR /
                     "cumulative_returns_strategy_vs_benchmark.png"),
        title=(
            f"Backtest ({cfg.BACKTEST_START} to {cfg.BACKTEST_END}) — "
            f"Strategy ({cfg.REBALANCE_MODE}) vs 60/40 Benchmark"
        ),
    )

    # (e) Metrics
    strat_metrics = metrics_block(strat_ret)
    bench_metrics = metrics_block(bench_ret)

    metrics_df = pd.DataFrame(
        [
            {"portfolio": "strategy", **strat_metrics},
            {"portfolio": "benchmark_60_40", **bench_metrics},
        ]
    )
    metrics_df.to_csv(repo / cfg.TABLE_DIR /
                      "performance_metrics.csv", index=False)

    daily_df = pd.DataFrame(
        {
            "strategy_return": strat_ret,
            "benchmark_return": bench_ret,
            "strategy_cum": strat_cum,
            "benchmark_cum": bench_cum,
        }
    )
    daily_df.to_csv(repo / cfg.TABLE_DIR /
                    "backtest_daily_series.csv", index=True)

    # Extra audit (not required but helpful)
    if turnover is not None:
        turnover.to_csv(repo / cfg.TABLE_DIR / "turnover.csv", index=True)
    if reb_dates is not None:
        pd.Series(reb_dates.astype(str), name="rebalance_dates").to_csv(
            repo / cfg.TABLE_DIR / "rebalance_dates.csv",
            index=False,
        )

    # (f) Conclusion
    conclusion = conclusion_text(
        strat_metrics, bench_metrics, mode=cfg.REBALANCE_MODE)
    write_text(repo / cfg.SUMMARY_DIR / "conclusion.md", conclusion + "\n")

    summary = {
        "backtest_window": {
            "start": cfg.BACKTEST_START,
            "end": cfg.BACKTEST_END,
            "n_days": int(len(bt)),
        },
        "mode": cfg.REBALANCE_MODE,
        "transaction_cost_bps": cfg.TRANSACTION_COST_BPS,
        "strategy_weights_initial": strategy_weights,
        "benchmark_weights": bench_weights,
        "strategy_metrics": strat_metrics,
        "benchmark_metrics": bench_metrics,
    }
    write_json(repo / cfg.OUT_DIR / "task5_summary.json", summary)

    print("Task 5 complete. Outputs saved to outputs/task5/")


if __name__ == "__main__":
    main()
