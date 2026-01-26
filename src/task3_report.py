from __future__ import annotations

import json
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_task3_outputs(out_dir: Path, horizon_label: str, trend: dict, ci_sum: dict, opp_risk: dict, reliability_text: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    trend_md = (
        f"## Trend Analysis ({horizon_label})\n\n"
        f"The central forecast (median/p50) indicates an **{trend['label']}** trend over the forecast window, "
        f"with an estimated change of **{trend['pct_change']:.2f}%** from start to end. "
        "This summarizes the expected direction under the model’s central scenario.\n\n"
        "However, TSLA forecasts should be interpreted primarily through the uncertainty bands. "
        "The widening prediction interval over time indicates that confidence in specific long-horizon price levels diminishes substantially.\n"
    )

    opp_md = "## Market Opportunities\n" + \
        "\n".join([f"- {x}" for x in opp_risk["opportunities"]]) + "\n\n"
    risk_md = "## Market Risks\n" + \
        "\n".join([f"- {x}" for x in opp_risk["risks"]]) + "\n"

    rel_md = f"## Forecast Reliability Assessment ({horizon_label})\n\n{reliability_text}\n"

    _write_text(out_dir / f"trend_analysis_{horizon_label}.md", trend_md)
    _write_text(
        out_dir / f"opportunities_risks_{horizon_label}.md", opp_md + risk_md)
    _write_text(out_dir / f"reliability_assessment_{horizon_label}.md", rel_md)


def write_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
