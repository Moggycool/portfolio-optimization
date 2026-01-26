""" Reporting utilities for Task 4: Black-Litterman Model Implementation. """
from __future__ import annotations
import json
from pathlib import Path


def write_text(path: Path, text: str):
    """Write text content to a file."""
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict):
    """Write dictionary as JSON to a file."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_weights(assets, w):
    """Format weights array as a dictionary."""
    return {a: float(x) for a, x in zip(assets, w)}


def recommendation_paragraph(choice: str, ret: float, vol: float, sharpe: float) -> str:
    """Generate recommendation paragraph for the chosen portfolio."""
    return (
        f"We recommend the **{choice}** portfolio because it provides an attractive trade-off between return and risk "
        f"given the TSLA forecast-based view and the historical diversification structure across SPY and BND. "
        f"The portfolio’s expected annual return is approximately **{ret:.2%}**, with expected annual volatility of **{vol:.2%}** "
        f"and a Sharpe ratio of **{sharpe:.2f}** (rf=2% annual)."
    )


def write_recommendation(summary_dir: Path, paragraph: str) -> None:
    """Write the recommendation paragraph to the Task 4 summary output directory."""
    write_text(summary_dir / "task4_recommendation.md", paragraph + "\n")
