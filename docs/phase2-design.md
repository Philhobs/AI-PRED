# Phase 2: Ensemble Predictor — Design Spec

**Goal:** Extend the AI infrastructure predictor with an ensemble ML model (LightGBM + Random Forest + Ridge) that ranks all 24 watchlist tickers by expected 1-year annualized return, with confidence intervals, served via FastAPI and visualised in a Streamlit dashboard, updated daily by APScheduler.

**Prediction target:** 1-year forward annualized return (cross-sectional regression). At 5+ year investment horizons, research shows fundamentals dominate price signals; TFT is inappropriate for this horizon. The ensemble approach matches peer-reviewed findings for long-horizon equity return prediction.

**Output per ticker:** `rank`, `expected_annual_return`, `confidence_low` (10th pct), `confidence_high` (90th pct), `as_of_date`

---

## Architecture Overview

```
ingestion/                   → raw data → data/raw/
  ohlcv_ingestion.py         (updated: period="max" ~20y)
  fundamental_ingestion.py   (NEW: yfinance quarterly fundamentals)
  news_ingestion.py          (existing)
  financial_ingestion.py     (existing: EDGAR XBRL capex)

processing/
  feature_engineering.py     (existing: price features)
  fundamental_features.py    (NEW: join fundamentals onto price matrix)
  label_builder.py           (NEW: 1-year forward return labels)
  nlp_pipeline.py            (existing)

models/
  train.py                   (NEW: train ensemble, save artifacts)
  inference.py               (NEW: load artifacts, predict, write Parquet)
  artifacts/                 (gitignored: fitted model pickles + scaler)

api/
  main.py                    (NEW: FastAPI, 4 endpoints, reads Parquet)

dashboard/
  app.py                     (NEW: Streamlit, 2 pages)

orchestration/
  scheduler.py               (NEW: APScheduler, daily + weekly jobs)

data/
  raw/financials/ohlcv/      (existing, extended to max history)
  raw/financials/fundamentals/<TICKER>/quarterly.parquet  (NEW)
  features/daily/date=*/     (existing)
  predictions/date=*/predictions.parquet  (NEW)
```

---

## Section 1: Data & Features

### 1.1 OHLCV History Extension

`ingestion/ohlcv_ingestion.py` — change `period="2y"` default to `period="max"`. A bootstrap CLI flag (`--bootstrap`) triggers the full max-history download; daily updates use `period="5d"`.

### 1.2 Fundamental Ingestion (`ingestion/fundamental_ingestion.py`)

New module. Pulls quarterly fundamentals from yfinance for all 24 TICKERS.

**Source:** `yf.Ticker(ticker).quarterly_financials`, `.quarterly_balance_sheet`, `.info`

**Features extracted:**
| Feature | Source field | Rationale |
|---|---|---|
| `pe_ratio_trailing` | `info["trailingPE"]` | Valuation signal |
| `price_to_sales` | `info["priceToSalesTrailing12Months"]` | Valuation vs revenue |
| `price_to_book` | `info["priceToBook"]` | Asset valuation |
| `revenue_growth_yoy` | quarterly_financials "Total Revenue" YoY | Growth signal |
| `gross_margin` | quarterly_financials | Profitability |
| `operating_margin` | quarterly_financials | Profitability |
| `capex_to_revenue` | quarterly_financials "Capital Expenditures" / revenue | Capital intensity |
| `debt_to_equity` | quarterly_balance_sheet | Risk signal |
| `current_ratio` | quarterly_balance_sheet | Liquidity |

Written to: `data/raw/financials/fundamentals/<TICKER>/quarterly.parquet`

Schema: `ticker (string), period_end (date32), <feature_columns> (float64)`

Missing values (fields not available for a ticker/period) stored as `null` — LightGBM handles natively. For Ridge and RF, null values are imputed with the **training-set ticker median** for that feature. The imputation map (ticker → feature → median) is computed once during `train.py` on the training split only (no look-ahead) and saved to `models/artifacts/imputation_medians.json`. Inference loads this artifact and applies the same map — the live feature vector is never used to recompute medians.

### 1.3 Fundamental Features (`processing/fundamental_features.py`)

Joins the latest available quarterly fundamental snapshot onto each price-feature row by matching the most recent `period_end <= date`. Returns the combined ~20-feature Polars DataFrame.

**Function signature:**
```python
def join_fundamentals(
    price_df: pl.DataFrame,
    fundamentals_dir: Path = Path("data/raw/financials/fundamentals"),
) -> pl.DataFrame:
```

### 1.4 Label Builder (`processing/label_builder.py`)

Computes 1-year forward annualized return for each ticker×date row.

```
label = (close_price_t+252 / close_price_t) - 1
```

Rows where `t + 252` exceeds the latest available date are excluded (incomplete forward window). This prevents look-ahead leakage.

**Function signature:**
```python
def build_labels(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
) -> pl.DataFrame:
    """Returns DataFrame: ticker, date, label_return_1y (float64). No nulls."""
```

---

## Section 2: Model Architecture

### 2.1 Training (`models/train.py`)

**Input:** Joined feature+label DataFrame (all tickers, all dates with complete labels).

**Time-series cross-validation:** Walk-forward expanding window. Training set: all data up to split point. Validation set: next 252 days. 3 folds minimum. No shuffling — future data never enters training.

**Three models trained:**

**LightGBM** — quantile regression, three passes:
- `alpha=0.10` → `confidence_low`
- `alpha=0.50` → `lgbm_return` (point estimate)
- `alpha=0.90` → `confidence_high`
- Hyperparameters: `n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20`
- Handles null fundamentals natively

**Random Forest** — standard regression:
- `n_estimators=300, max_features="sqrt", min_samples_leaf=5`
- Features imputed with ticker median before fit
- Provides point estimate: `rf_return`

**Ridge Regression** — L2 regularized linear:
- Features standardized (StandardScaler, fitted on training set only)
- `alpha=1.0`
- Provides point estimate: `ridge_return`

**Ensemble weights** — learned on held-out validation set using non-negative least squares (scipy `nnls`). Weights sum to 1. Stored in `models/artifacts/ensemble_weights.json`.

**Point estimate:** `expected_annual_return = w_lgbm * lgbm_return + w_rf * rf_return + w_ridge * ridge_return`

**Artifacts saved to `models/artifacts/`:**
- `lgbm_q10.pkl`, `lgbm_q50.pkl`, `lgbm_q90.pkl`
- `rf_model.pkl`
- `ridge_model.pkl`
- `feature_scaler.pkl` (StandardScaler fitted on training set only — used by both Ridge and RF at inference time)
- `imputation_medians.json` (ticker → feature → median, computed on training set only)
- `feature_names.json` (ordered list — guards against column drift)
- `ensemble_weights.json`

### 2.2 Inference (`models/inference.py`)

```python
def run_inference(
    date_str: str,
    data_dir: Path = Path("data/raw"),
    artifacts_dir: Path = Path("models/artifacts"),
    output_dir: Path = Path("data/predictions"),
) -> pl.DataFrame:
```

Steps:
1. Build today's feature vector (price features + latest fundamentals) for all 24 tickers
2. Load all artifacts; verify `feature_names.json` matches current feature set (raises if mismatch)
3. Apply `imputation_medians.json` to fill nulls for RF and Ridge inputs; apply `feature_scaler.pkl` (fitted on training set) to standardize — the scaler is never refit at inference time
4. Run LightGBM q10/q50/q90 (on raw features with nulls), RF, Ridge (on imputed+scaled features)
5. Combine with ensemble weights from `ensemble_weights.json`
6. Rank by `expected_annual_return` descending (rank 1 = highest expected return)
7. Write to `data/predictions/date={date_str}/predictions.parquet`
8. Return Polars DataFrame

**Output schema:**
```
ticker (string), rank (int32), expected_annual_return (float64),
confidence_low (float64), confidence_high (float64),
lgbm_return (float64), rf_return (float64), ridge_return (float64),
as_of_date (date32)
```

---

## Section 3: API (`api/main.py`)

FastAPI app. Reads Parquet directly — no DB server.

```python
GET /health
→ {"status": "ok", "last_prediction_date": "2026-04-13", "ticker_count": 24}

GET /predictions/latest
→ [{"ticker": "MRVL", "rank": 1, "expected_annual_return": 0.18,
    "confidence_low": 0.04, "confidence_high": 0.31, "as_of_date": "2026-04-13"}, ...]

GET /predictions/{ticker}
→ [{"as_of_date": "2026-04-13", "rank": 1, "expected_annual_return": 0.18, ...}, ...]
   (all historical prediction dates for this ticker)

GET /features/{ticker}
→ [{"date": "2026-04-13", "close_price": 132.23, "return_5d": 0.208,
    "pe_ratio_trailing": 45.2, "revenue_growth_yoy": 0.34, ...}, ...]
```

No auth. Runs on port 8000 via `uvicorn api.main:app --reload`.

**Error handling:** 404 if ticker not in watchlist (`{"detail": "Ticker XYZ not in watchlist"}`). 503 if no predictions Parquet exists yet (`{"detail": "No predictions available yet. Run python models/inference.py to generate."}`). All errors use `{"detail": "<message>"}` format.

---

## Section 4: Dashboard (`dashboard/app.py`)

Streamlit app. Reads Parquet directly (same source as API).

### Page 1 — Top Picks

- **Ranked table:** all 24 tickers sorted by `expected_annual_return`. Columns: Rank, Ticker, Sector, Expected Annual Return (%), Confidence Low (%), Confidence High (%), Confidence Width.
- **Buy list box:** top 5 tickers highlighted with colour (green gradient). Shows the top 3 feature importances for each (from LightGBM `feature_importances_`).
- **Last updated** timestamp + staleness warning if predictions are >2 days old.

### Page 2 — Stock Drill-down

Sidebar ticker selector.

- **Price chart:** 2-year close price with 20-day SMA (Plotly).
- **Feature sparklines:** top 5 LightGBM feature importances for this ticker — one sparkline each, last 90 days.
- **Prediction history:** `expected_annual_return` + confidence band over all historical prediction dates (Plotly area chart).
- **Fundamentals table:** latest P/E, revenue growth, gross margin, capex-to-revenue vs sector median (colour-coded delta).

Runs on port 8501 via `streamlit run dashboard/app.py`.

---

## Section 5: Orchestration (`orchestration/scheduler.py`)

APScheduler `BlockingScheduler`. Runs as `python orchestration/scheduler.py`.

**Daily job** — weekdays, 16:30 ET (UTC-4/5 adjusted). Each step uses `date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")`:
```
1. ohlcv_ingestion.py        -- refresh last 5 days prices (period="5d")
2. news_ingestion.py         -- GDELT + RSS
3. nlp_pipeline.py           -- FinBERT score today's articles
4. feature_engineering.py    -- build today's feature matrix for date_str
5. inference.py run_inference(date_str)  -- ensemble predict → predictions Parquet
```

**Weekly job** — Sunday 08:00 ET:
```
1. fundamental_ingestion.py  -- refresh quarterly fundamentals
2. train.py                  -- retrain full ensemble, replace artifacts
```

**Bootstrap (one-time, manual) — must run in order, abort on any failure:**
```bash
python ingestion/ohlcv_ingestion.py --bootstrap   # max history (~20y)
python ingestion/fundamental_ingestion.py          # all available quarters
python processing/label_builder.py                 # build 1y forward labels
python models/train.py                             # initial ensemble train
python models/inference.py                         # first predictions
```
If any bootstrap step fails, fix the error before continuing — later steps depend on earlier outputs.

**Scheduled-job error policy:** each scheduled job wrapped in try/except. Failure logs `[Scheduler] ERROR <job>: <message>` and continues. A broken news scrape never blocks inference. Bootstrap failures are not caught — they abort immediately so errors are visible.

---

## Tech Stack

| Component | Library |
|---|---|
| ML models | `lightgbm>=4.0`, `scikit-learn>=1.5` (RF + Ridge + scaler + nnls) |
| Data | `yfinance>=0.2`, `polars>=1.0`, `pyarrow>=17.0` |
| API | `fastapi>=0.115`, `uvicorn>=0.32` |
| Dashboard | `streamlit>=1.40`, `plotly>=5.24` |
| Scheduling | `apscheduler>=3.10` |
| Existing | `duckdb>=1.1`, `pandas>=2.0`, `transformers`, `torch` |

`lightgbm` added to core `[project.dependencies]`. All others already in `phase2` optional group.

---

## Testing Strategy

**Unit tests (no network, no disk):**
- `tests/test_label_builder.py` — label computation correctness, look-ahead guard
- `tests/test_fundamental_features.py` — join logic, null handling
- `tests/test_train.py` — train on fixture data (10 tickers × 500 days), assert artifacts written
- `tests/test_inference.py` — load fixture artifacts, assert output schema and rank uniqueness
- `tests/test_api.py` — FastAPI TestClient against fixture predictions Parquet

**Integration tests (marked `integration`):**
- `tests/test_integration.py` additions — fundamental_ingestion for NVDA, inference end-to-end

---

## Phase 2 Optional-Dependency Install

```bash
pip install -e ".[phase2]"
pip install lightgbm>=4.0
```

