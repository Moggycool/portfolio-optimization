# portfolio-optimization

Time Series Forecasting for Portfolio Management Optimization

## Project Objective

GMF Investments uses time-series forecasting to support portfolio optimization by forecasting **log returns** (primary) and reconstructing **price paths** (secondary) for interpretation.

## Data

- Source: YFinance
- Assets: TSLA (equity), SPY (market proxy), BND (bond proxy)
- Frequency: daily
- Period: 2015–present (configurable)

## Task 1 (EDA) Outputs

Artifacts created by scripts:

- Processed dataset(s): `data/task1/processed/...`
- Scaled dataset evidence: `data/task1/processed/scaled_task1.parquet`
- Visualizations (PNG):
  - `outputs/task1/viz/tsla_adj_close_timeseries.png`
  - `outputs/task1/viz/tsla_daily_pct_change.png`
  - `outputs/task1/viz/tsla_rolling_mean_std.png`

## Task 2 (Forecasting) Design

### Targets

- **Primary target:** `logret_1d` (next-day log return)
- **Price column for features/reporting:** `adj_close`

### Why returns-primary?

- closer to stationary → better statistical validity
- directly supports portfolio allocation and risk metrics
- EMH suggests price-level prediction is unreliable; returns/volatility factors are more actionable

### Outputs

- Returns forecasts:
  - `outputs/task2/forecasts/tsla_lstm_forecast_returns.csv`
  - `outputs/task2/forecasts/tsla_arima_forecast_returns.csv` (if enabled)
- Price reconstruction (from predicted returns):
  - `outputs/task2/forecasts/tsla_lstm_forecast_price.csv`

## How to run (script order)

1) Task 1 EDA + processing (your script name here)

- python scripts/01_task1_extract_and_process.py

2) Task 1 scaling + visualizations (added)

- python scripts/02_task1_scale_and_viz.py

3) Task 2 splits/features

- python scripts/04_task2_make_splits_and_features.py

4) Task 2 train LSTM on returns + reconstruct price

- python scripts/05_task2_train_lstm.py
