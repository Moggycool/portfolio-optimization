"""Script to plot Task 3 forecasts and uncertainty intervals."""
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
    """Main function to plot Task 3 forecasts and uncertainty intervals."""
    repo_root = find_repo_root(Path(__file__).resolve())

    # Ensure repo root is on sys.path so `import src...` works when running as a file path.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Local imports (after sys.path bootstrap)
    import pandas as pd
    import src.task3_config as config
    from src.task3_data import (
        load_parquet,
        get_asset_series,
        load_split_info,
    )

    from src.task3_plotting import plot_price_forecast, plot_ci_width

    def load_task2_test_forecasts_for_overlay(
        path: Path,
        *,
        date_candidates: list[str],
        actual_candidates: list[str],
        arima_candidates: list[str],
        lstm_candidates: list[str],
    ) -> pd.DataFrame:
        """Load Task 2 test forecasts for overlay plots.

        Returns a DataFrame indexed by datetime with standardized columns:
        - actual (if present)
        - arima (if present)
        - lstm (if present)
        """
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col is None:
            raise ValueError(
                f"Could not find a date column in {path}. Tried: {date_candidates}"
            )

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        def pick_col(candidates: list[str]) -> str | None:
            return next((c for c in candidates if c in df.columns), None)

        actual_col = pick_col(actual_candidates)
        arima_col = pick_col(arima_candidates)
        lstm_col = pick_col(lstm_candidates)

        out = pd.DataFrame(index=df.index)
        if actual_col is not None:
            out["actual"] = df[actual_col]
        if arima_col is not None:
            out["arima"] = df[arima_col]
        if lstm_col is not None:
            out["lstm"] = df[lstm_col]

        if out.empty:
            raise ValueError(
                f"No usable forecast columns found in {path}. "
                f"Tried actual={actual_candidates}, arima={arima_candidates}, lstm={lstm_candidates}"
            )

        return out

    config.ensure_output_dirs(repo_root)

    # Historical prices
    prices_df = load_parquet(repo_root / config.PRICES_PATH)
    hist_price = get_asset_series(
        prices_df,
        config.ASSET,
        config.PRICE_COL,
        config.DATE_COL,
        config.ASSET_COL,
    )

    # Split info (optional)
    split_info = load_split_info(repo_root / config.SPLIT_INFO_PATH)

    # Load Task 2 test overlay forecasts (prefer merged)
    overlay = None
    merged_path = repo_root / config.TASK2_MERGED_FORECASTS_PATH
    if merged_path.exists():
        overlay = load_task2_test_forecasts_for_overlay(
            merged_path,
            date_candidates=config.FORECAST_DATE_CANDIDATES,
            actual_candidates=config.ACTUAL_PRICE_CANDIDATES,
            arima_candidates=config.ARIMA_PRICE_CANDIDATES,
            lstm_candidates=config.LSTM_PRICE_CANDIDATES,
        )
    else:
        # fallback: try ARIMA-only
        arima_path = repo_root / config.TASK2_ARIMA_TEST_FORECAST_PATH
        if arima_path.exists():
            overlay = load_task2_test_forecasts_for_overlay(
                arima_path,
                date_candidates=config.FORECAST_DATE_CANDIDATES,
                actual_candidates=config.ACTUAL_PRICE_CANDIDATES,
                arima_candidates=config.ARIMA_PRICE_CANDIDATES,
                lstm_candidates=[],
            )

    for label in config.HORIZONS.keys():
        price_fc = pd.read_csv(
            repo_root / config.FORECAST_DIR /
            f"tsla_prices_forecast_{label}.csv"
        )
        price_fc["date"] = pd.to_datetime(price_fc["date"])
        price_fc = price_fc.set_index("date").sort_index()

        out_fig = repo_root / config.FIG_DIR / \
            f"fig_task3_price_forecast_{label}.png"
        plot_price_forecast(
            history_price=hist_price,
            future_q=price_fc,
            split_info=split_info,
            test_overlay=overlay,
            out_path=str(out_fig),
            title=f"TSLA: Historical vs Test Predictions vs Future Forecast ({label})",
        )

        width = (price_fc["price_p95"] - price_fc["price_p05"]).rename("width")
        out_w = repo_root / config.FIG_DIR / f"fig_task3_ci_width_{label}.png"
        plot_ci_width(
            width,
            str(out_w),
            title=f"TSLA Uncertainty Width (p95 - p05) — {label}",
        )

    print("Task 3 figures saved to outputs/task3/figures/")


if __name__ == "__main__":
    main()
