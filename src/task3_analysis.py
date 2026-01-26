""" Task 3: Analysis functions for trend and uncertainty metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_metrics(price_p50: pd.Series) -> dict:
    """Compute trend metrics from the median price forecast series."""
    start = float(price_p50.iloc[0])
    end = float(price_p50.iloc[-1])
    pct = (end / start - 1.0) * 100.0

    x = np.arange(len(price_p50))
    y = price_p50.to_numpy()
    slope = float(np.polyfit(x, y, 1)[0])

    if pct > 3:
        label = "upward"
    elif pct < -3:
        label = "downward"
    else:
        label = "stable"

    return {"start": start, "end": end, "pct_change": pct, "slope_per_step": slope, "label": label}


def ci_width_series(price_q: pd.DataFrame) -> pd.Series:
    """Compute the width of the confidence interval from price quantiles."""
    return (price_q["price_p95"] - price_q["price_p05"]).rename("ci_width")


def ci_width_summary(width: pd.Series) -> dict:
    """Summarize the confidence interval width series at key points."""
    def pick(i: int) -> float:
        i = min(i, len(width) - 1)
        return float(width.iloc[i])

    return {
        "day_1": pick(0),
        "day_21_~1m": pick(20),
        "day_63_~3m": pick(62),
        "day_126_~6m": pick(125),
        "day_252_~12m": pick(251),
        "end": float(width.iloc[-1]),
        "start": float(width.iloc[0]),
        "end_to_start_ratio": float(width.iloc[-1] / max(width.iloc[0], 1e-12)),
    }


def opportunities_and_risks(trend: dict, ci_sum: dict) -> dict:
    """Generate a list of opportunities and risks based on trend and CI summary."""
    opp, risks = [], []

    opp.append(
        f"Central forecast trend is **{trend['label']}** (p50 change ≈ {trend['pct_change']:.2f}% over horizon).")

    if trend["label"] == "upward":
        opp.append(
            "Potential opportunity: gradual upside scenario supports phased entry / rebalancing toward TSLA within risk limits.")
    elif trend["label"] == "downward":
        opp.append(
            "Potential opportunity: if declines are forecast, consider defensive positioning, hedges, or wait-for-better-entry triggers.")
    else:
        opp.append(
            "Potential opportunity: stable median path suggests focusing on scenario/range planning rather than strong directional bets.")

    risks.append(
        f"Uncertainty widens materially with horizon (band width end/start ratio ≈ {ci_sum['end_to_start_ratio']:.2f}×).")
    risks.append(
        "High forecast uncertainty implies long-horizon point forecasts should not be treated as precise targets.")
    risks.append(
        "Wide bands indicate substantial tail-risk; use scenarios for VaR/stress testing and set position limits accordingly.")

    return {"opportunities": opp, "risks": risks}


def reliability_assessment_text(ci_sum: dict, horizon_label: str) -> str:
    """Generate a reliability assessment text based on CI summary."""
    return (
        f"For the {horizon_label} forecast, the prediction band widens as the horizon increases "
        f"(e.g., width ≈ {ci_sum['day_21_~1m']:.2f} at ~1 month vs ≈ {ci_sum['end']:.2f} at the end). "
        "This widening reflects compounding uncertainty when forecasting volatile assets like TSLA. "
        "Short horizons are generally more actionable (tighter bands), while 6–12 month forecasts should be interpreted as broad scenario ranges "
        "rather than precise price targets."
    )
