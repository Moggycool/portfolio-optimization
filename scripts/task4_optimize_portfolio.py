"""Optimize portfolio based on expected returns and risk."""
from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root(start: Path | None = None) -> Path:
    """Auto-detect repo root by looking for 'src' and 'data' dirs."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        "Could not auto-detect repo root. Run from inside the repo.")


def main():
    """Main function to optimize portfolio and generate outputs."""
    repo = find_repo_root(Path(__file__).resolve())

    # Ensure repo root is on sys.path so `import src...` works when running as a file path.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Local imports (after sys.path bootstrap)
    import pandas as pd
    import src.task4_config as cfg
    from src.task4_data import load_parquet, pivot_returns_long_to_wide
    from src.task4_expected_returns import (
        load_tsla_forecast_logrets,
        build_expected_returns_vector,
    )
    from src.task4_risk import cov_annualized, portfolio_performance
    from src.task4_optimize import (
        min_vol_portfolio,
        max_sharpe_portfolio,
        efficient_frontier,
    )
    from src.task4_plotting import plot_cov_heatmap, plot_efficient_frontier
    from src.task4_report import (
        write_text,
        write_json,
        format_weights,
        recommendation_paragraph,
    )

    cfg.ensure_output_dirs(repo)

    # (1) Historical returns
    df_ret = load_parquet(repo / cfg.RETURNS_PATH)
    rets_wide = pivot_returns_long_to_wide(
        df_ret,
        date_col=cfg.DATE_COL,
        asset_col=cfg.ASSET_COL,
        ret_col=cfg.RET_COL,
        assets=cfg.ASSETS,
    )
    assets = list(rets_wide.columns)

    # (2) Expected returns vector
    fc_path = repo / (
        cfg.TSLA_FORECAST_LOGRET_6M
        if cfg.TSLA_FORECAST_HORIZON == "6m"
        else cfg.TSLA_FORECAST_LOGRET_12M
    )
    tsla_logrets = load_tsla_forecast_logrets(fc_path)
    mu = build_expected_returns_vector(
        rets_wide, tsla_logrets, trading_days=cfg.TRADING_DAYS)

    # (3) Covariance matrix
    cov = cov_annualized(rets_wide, trading_days=cfg.TRADING_DAYS)

    # Save tables
    mu.to_csv(repo / cfg.TABLE_DIR /
              "expected_returns_annual.csv", header=["mu_annual"])
    cov.to_csv(repo / cfg.TABLE_DIR / "covariance_annual.csv")

    # (4) Optimize + frontier
    mu_arr = mu.loc[assets].to_numpy()
    cov_arr = cov.loc[assets, assets].to_numpy()

    w_minvol = min_vol_portfolio(mu_arr, cov_arr, rf=cfg.RISK_FREE_RATE_ANNUAL)
    w_maxsh = max_sharpe_portfolio(
        mu_arr, cov_arr, rf=cfg.RISK_FREE_RATE_ANNUAL)

    r_min, v_min, s_min = portfolio_performance(
        w_minvol, mu_arr, cov_arr, cfg.RISK_FREE_RATE_ANNUAL
    )
    r_max, v_max, s_max = portfolio_performance(
        w_maxsh, mu_arr, cov_arr, cfg.RISK_FREE_RATE_ANNUAL
    )

    ef_rets, ef_vols, ef_w = efficient_frontier(
        mu_arr, cov_arr, rf=cfg.RISK_FREE_RATE_ANNUAL, n_points=60
    )

    # (5) Visualize
    plot_cov_heatmap(
        cov.loc[assets, assets],
        out_path=str(repo / cfg.FIG_DIR / "covariance_heatmap.png"),
    )
    plot_efficient_frontier(
        ef_rets,
        ef_vols,
        minvol_point={"ret": r_min, "vol": v_min},
        maxsharpe_point={"ret": r_max, "vol": v_max},
        out_path=str(repo / cfg.FIG_DIR / "efficient_frontier.png"),
    )

    # (6) Recommend portfolio (default: max Sharpe)
    VOL_CAP = 0.25  # 25% annual vol threshold (tune as needed)

    if v_max <= VOL_CAP:
        choice = "Maximum Sharpe Ratio"
        w_rec = w_maxsh
        r_rec, v_rec, s_rec = r_max, v_max, s_max
    else:
        choice = "Minimum Volatility"
        w_rec = w_minvol
        r_rec, v_rec, s_rec = r_min, v_min, s_min
    weights_table = pd.DataFrame(
        {
            "asset": assets,
            "w_min_vol": w_minvol,
            "w_max_sharpe": w_maxsh,
            "w_recommended": w_rec,
        }
    )
    weights_table.to_csv(repo / cfg.TABLE_DIR /
                         "portfolio_weights.csv", index=False)

    paragraph = recommendation_paragraph(choice, r_rec, v_rec, s_rec)
    write_text(repo / cfg.SUMMARY_DIR /
               "portfolio_recommendation.md", paragraph + "\n")

    summary = {
        "assets": assets,
        "tsla_forecast_horizon_used": cfg.TSLA_FORECAST_HORIZON,
        "risk_free_rate_annual": cfg.RISK_FREE_RATE_ANNUAL,
        "expected_returns_annual": mu.to_dict(),
        "min_vol": {
            "weights": format_weights(assets, w_minvol),
            "expected_return": r_min,
            "volatility": v_min,
            "sharpe": s_min,
        },
        "max_sharpe": {
            "weights": format_weights(assets, w_maxsh),
            "expected_return": r_max,
            "volatility": v_max,
            "sharpe": s_max,
        },
        "recommended": {
            "choice": choice,
            "weights": format_weights(assets, w_rec),
            "expected_return": r_rec,
            "volatility": v_rec,
            "sharpe": s_rec,
        },
    }
    write_json(repo / cfg.OUT_DIR / "task4_summary.json", summary)

    print("Task 4 complete. Outputs saved to outputs/task4/")


if __name__ == "__main__":
    main()
