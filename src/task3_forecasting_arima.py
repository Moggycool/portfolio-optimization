from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_arima_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_sarimax_on_returns(ret: pd.Series, spec: dict):
    order = tuple(spec.get("order", [0, 0, 0]))
    seasonal = bool(spec.get("seasonal", False))
    seasonal_order = tuple(
        spec.get("seasonal_order", [0, 0, 0, 0])) if seasonal else (0, 0, 0, 0)

    # Returns-space: include a constant mean
    model = SARIMAX(
        ret,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    return fit


def forecast_returns_with_pi(fit, steps: int, alpha: float) -> pd.DataFrame:
    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)

    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]

    out = pd.DataFrame(
        {"ret_mean": mean.values, "ret_lower": lower.values, "ret_upper": upper.values},
        index=mean.index,
    )
    return out
