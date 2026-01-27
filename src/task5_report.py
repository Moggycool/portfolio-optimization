from __future__ import annotations
import json
from pathlib import Path


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def conclusion_text(strategy_metrics: dict, bench_metrics: dict, mode: str) -> str:
    s_tr = strategy_metrics["total_return"]
    b_tr = bench_metrics["total_return"]
    better = "outperformed" if s_tr > b_tr else "underperformed"

    return (
        f"## Conclusion (Backtest: {mode})\n\n"
        f"Over the backtest window, the strategy **{better}** the 60/40 benchmark on total return "
        f"({s_tr:.2%} vs {b_tr:.2%}). The strategy’s Sharpe ratio and maximum drawdown provide additional context "
        f"for risk-adjusted performance and downside risk.\n\n"
        "Limitations: this is a simplified backtest using daily returns with (optional) stylized rebalancing and no/slim transaction costs. "
        "It does not model slippage, bid–ask spreads, taxes, or realistic execution timing. Results are sensitive to the chosen backtest window "
        "and to estimation error in expected returns and covariances (especially with a forecast-based view on TSLA). "
        "A more robust evaluation would include multiple windows, transaction-cost sensitivity, and stress/regime analysis."
    )
