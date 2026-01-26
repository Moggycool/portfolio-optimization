"""Analyze trends and uncertainties from forecasts and generate summaries."""
from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root directory."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        "Could not auto-detect repo root. Run from inside the repo.")


def main():
    """Main function to analyze trends and uncertainties from forecasts."""
    repo_root = find_repo_root(Path(__file__).resolve())

    # Ensure repo root is on sys.path so `import src...` works when running as a file path.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Local imports (after sys.path bootstrap)
    import pandas as pd
    import src.task3_config as config
    from src.task3_analysis import (
        trend_metrics,
        ci_width_series,
        ci_width_summary,
        opportunities_and_risks,
        reliability_assessment_text,
    )
    from src.task3_report import write_task3_outputs, write_summary_json

    config.ensure_output_dirs(repo_root)

    summary = {"asset": config.ASSET, "alpha": config.ALPHA, "horizons": {}}

    for label in config.HORIZONS.keys():
        price_fc = pd.read_csv(
            repo_root / config.FORECAST_DIR /
            f"tsla_prices_forecast_{label}.csv"
        )
        price_fc["date"] = pd.to_datetime(price_fc["date"])
        price_fc = price_fc.set_index("date").sort_index()

        trend = trend_metrics(price_fc["price_p50"])
        width = ci_width_series(price_fc)
        ci_sum = ci_width_summary(width)
        opp_risk = opportunities_and_risks(trend, ci_sum)
        reliability = reliability_assessment_text(ci_sum, horizon_label=label)

        write_task3_outputs(
            repo_root / config.SUMMARY_DIR, label, trend, ci_sum, opp_risk, reliability
        )

        summary["horizons"][label] = {
            "trend": trend,
            "ci_width_summary": ci_sum,
            "opportunities": opp_risk["opportunities"],
            "risks": opp_risk["risks"],
        }

    write_summary_json(repo_root / config.OUT_DIR /
                       "task3_summary.json", summary)
    print("Task 3 summaries saved to outputs/task3/summaries/ and task3_summary.json")


if __name__ == "__main__":
    main()
