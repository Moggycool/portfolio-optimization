
# Portfolio Optimization & Time Series Forecasting (TSLA · BND · SPY)

This repository provides a **script-first, fully reproducible pipeline** for financial time series analysis and forecasting, designed to meet academic and rubric-driven requirements.

The project covers:

- **Task 1 — Exploratory Data Analysis & Risk Metrics**  
  Data extraction, cleaning, return computation, stationarity testing, outlier detection, and portfolio risk metrics for **TSLA, BND, and SPY**.

- **Task 2 — Time Series Forecasting**  
  Forecasting **TSLA next-day log returns (`logret_1d`)** using **ARIMA/SARIMA** and a **multivariate LSTM**, followed by model comparison using required evaluation metrics.  
  Predicted returns are optionally reconstructed into **price paths** for stakeholder-friendly interpretation.

> **Important**  
> Official submission outputs are the **script-generated artifacts** located under:  
> `data/task1/processed/`, `outputs/task1/`, and `outputs/task2/`  
>
> Notebooks are **for inspection and presentation only**.

---

## 1. Project Structure (Key Paths)

```text
portfolio-optimization/
├─ notebooks/
│  ├─ 01_task1_data_extraction_and_eda.ipynb
│  ├─ task1.ipynb
│  ├─ task2.ipynb
│  ├─ task2_04_model_comparison_and_discussion.ipynb
│  ├─ README.md
│  └─ __init__.py
│
├─ scripts/
│  ├─ 01_fetch_data.py
│  ├─ 02_make_returns_and_task1_tables.py
│  ├─ 03_task2_make_tsla_splits_and_features.py
│  ├─ 04_task2_train_arima.py
│  ├─ 05_task2_train_lstm.py
│  ├─ 06_task2_compare_models.py
│  ├─ 07_task2_run_all.py
│  └─ __init__.py
│
├─ src/
│  ├─ config.py
│  ├─ data_fetch.py
│  ├─ data_prep.py
│  ├─ eda.py
│  ├─ io.py
│  ├─ risk_metrics.py
│  ├─ stationarity.py
│  ├─ task2_arima.py
│  ├─ task2_data.py
│  ├─ task2_io.py
│  ├─ task2_lstm.py
│  ├─ task2_metrics.py
│  ├─ task2_plots.py
│  └─ __init__.py
│
├─ data/
├─ outputs/
├─ tests/
│  └─ __init__.py
│
├─ requirements.txt
└─ README.md
```

---

## 2. Environment Setup (Reproducible)

### Option A — venv (Recommended)

```bash
python -m venv .venv
```

**Activate environment**

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. How to Run (Script Execution Order)

Scripts must be run **in order**.

### 3.1 Task 1 — EDA & Risk Metrics

```bash
python scripts/01_fetch_data.py
python scripts/02_make_returns_and_task1_tables.py
python scripts/02_task1_scale_and_viz.py
```

### 3.2 Task 2 — TSLA Forecasting

```bash
python scripts/03_task2_make_tsla_splits_and_features.py
python scripts/04_task2_train_arima.py
python scripts/05_task2_train_lstm.py
python scripts/06_task2_compare_models.py
```

---

## 4. Key Design Choices (Rubric-Facing)

### Task 1

- Uses **Adjusted Close (`adj_close`)** for consistent return and risk calculations.
- Produces cleaned datasets, ADF stationarity tests, outlier tables, and portfolio risk metrics.

### Task 2 — Returns-Primary Modeling

- **Target:** `logret_1d` (stationary, portfolio-aligned).
- **Price reconstruction:** predicted returns accumulated with an anchor price.
- **Leakage prevention:**
  - Chronological TRAIN / VAL / TEST splits
  - Scalers fit on TRAIN only
  - Targets created using `shift(-1)`

---

## 5. Configuration

All constants and paths live in:

```text
src/config.py
```

Key settings:

```python
START_DATE = "2015-01-01"
END_DATE   = "2026-01-15"

TASK2_VAL_YEAR   = 2023
TASK2_SPLIT_YEAR = 2024

TASK2_TARGET_COL = "logret_1d"
TASK2_PRICE_COL  = "adj_close"
```

---

## 6. License

This project is for **educational and assessment purposes**.
