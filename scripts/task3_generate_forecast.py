"""
Generate forecasts for Task 3 using ARIMA-like modeling on log returns.
Saves forecasted returns and price quantiles to CSV files.
"""
from __future__ import annotations

# Standard library imports
from pathlib import Path
import json
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
    """Main function to generate forecasts."""
    repo_root = find_repo_root(Path(__file__).resolve())

    # Ensure repo root is on sys.path so `import src...` works when running as a file path.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Local imports (after sys.path bootstrap)
    import src.task3_config as config
    from src.task3_data import (
        load_parquet,
        get_asset_series,
        align_on_intersection,
        make_future_bday_index,
        ensure_bday_frequency,
        to_log_return,
    )
    from src.task3_forecasting_arima import (
        load_arima_spec,
        fit_sarimax_on_returns,
        forecast_returns_with_pi,
    )
    from src.task3_intervals import (
        sigma_from_pi,
        simulate_price_paths_from_logrets,
        price_quantiles,
    )

    def _infer_return_col(returns_df):
        exclude = {config.DATE_COL, config.ASSET_COL}

        # Prefer return-like names
        ret_like = [
            c for c in returns_df.columns
            if ("ret" in str(c).lower()) and (c not in exclude)
        ]
        if ret_like:
            return ret_like[0]

        # Else: any numeric cols
        num = [
            c for c in returns_df.select_dtypes(include="number").columns
            if c not in exclude
        ]
        if not num:
            raise KeyError(
                f"No numeric return-like columns found. Available={list(returns_df.columns)}"
            )
        return num[0]

    config.ensure_output_dirs(repo_root)

    prices_df = load_parquet(repo_root / config.PRICES_PATH)
    returns_df = load_parquet(repo_root / config.RETURNS_PATH)

    price_s = get_asset_series(
        prices_df,
        config.ASSET,
        config.PRICE_COL,
        config.DATE_COL,
        config.ASSET_COL,
    )

    # Load ARIMA spec (val preferred)
    spec_path = (
        (repo_root / config.ARIMA_SPEC_VAL_PATH)
        if (repo_root / config.ARIMA_SPEC_VAL_PATH).exists()
        else (repo_root / config.ARIMA_SPEC_PATH)
    )
    spec = load_arima_spec(spec_path)

    requested_ret_col = spec.get("target_col", config.RET_COL_FALLBACK)

    # Build a *log return* series for modeling
    if requested_ret_col in returns_df.columns:
        ret_col_used = requested_ret_col
        logret_s = get_asset_series(
            returns_df,
            config.ASSET,
            ret_col_used,
            config.DATE_COL,
            config.ASSET_COL,
        )
        logret_source = f"used column '{ret_col_used}' directly"
    else:
        if "return" in returns_df.columns:
            ret_col_used = "return"
            simple_ret_s = get_asset_series(
                returns_df,
                config.ASSET,
                ret_col_used,
                config.DATE_COL,
                config.ASSET_COL,
            )
            logret_s = to_log_return(simple_ret_s)
            logret_source = "converted simple 'return' to log returns via log1p"
        else:
            ret_col_used = _infer_return_col(returns_df)
            simple_ret_s = get_asset_series(
                returns_df,
                config.ASSET,
                ret_col_used,
                config.DATE_COL,
                config.ASSET_COL,
            )
            logret_s = to_log_return(simple_ret_s)
            logret_source = f"converted inferred '{ret_col_used}' to log returns via log1p"

        print(
            f"[task3] Return column '{requested_ret_col}' not found; {logret_source}.")

    # Align price and log returns to same dates
    price_s, logret_s = align_on_intersection(price_s, logret_s)

    # Ensure business-day frequency for statsmodels stability
    logret_s_b = ensure_bday_frequency(logret_s, fill_value=0.0)

    # Fit ARIMA-like model on log returns
    fit = fit_sarimax_on_returns(logret_s_b, spec)

    # Audit spec used
    (repo_root / config.OUT_DIR).mkdir(parents=True, exist_ok=True)
    (repo_root / config.OUT_DIR / "task3_spec_used.json").write_text(
        json.dumps(
            {
                "spec_path": str(spec_path),
                "spec": spec,
                "alpha": config.ALPHA,
                "n_sims": config.N_SIMS,
                "asset": config.ASSET,
                "price_col": config.PRICE_COL,
                "requested_return_col": requested_ret_col,
                "return_col_used": ret_col_used,
                "logret_construction": logret_source,
                "last_observed_date": str(price_s.index.max()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    last_price = float(price_s.iloc[-1])
    last_date = price_s.index.max()

    for label, steps in config.HORIZONS.items():
        # Forecast *log returns* + PI
        ret_fc = forecast_returns_with_pi(fit, steps=steps, alpha=config.ALPHA)

        # Put future business-day dates on forecast
        future_idx = make_future_bday_index(last_date, steps)
        ret_fc.index = future_idx

        # Convert PI to sigma and simulate price fan chart
        sig = sigma_from_pi(ret_fc["ret_lower"],
                            ret_fc["ret_upper"], alpha=config.ALPHA)
        paths = simulate_price_paths_from_logrets(
            last_price,
            ret_fc["ret_mean"],
            sig,
            n_sims=config.N_SIMS,
            seed=config.RANDOM_SEED,
        )
        q = price_quantiles(paths, qs=(0.05, 0.5, 0.95))
        q.index = future_idx
        q.columns = ["price_p05", "price_p50", "price_p95"]

        # Save CSVs
        ret_out = ret_fc.reset_index().rename(columns={"index": "date"})
        price_out = q.reset_index().rename(columns={"index": "date"})

        ret_out.to_csv(
            repo_root / config.FORECAST_DIR /
            f"tsla_logret_forecast_{label}.csv",
            index=False,
        )
        price_out.to_csv(
            repo_root / config.FORECAST_DIR /
            f"tsla_prices_forecast_{label}.csv",
            index=False,
        )

    print("Task 3 forecasts saved to outputs/task3/forecasts/")


if __name__ == "__main__":
    main()
