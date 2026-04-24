# EDGAR Fundamentals Expansion — Design Spec

**Date:** 2026-04-24
**Status:** Approved

---

## What We're Building

Wire `edgar_fundamentals_ingestion.py` into the refresh pipeline (currently never called) and expand the fundamental feature set from 9 → 14 columns by adding 5 new EDGAR-derived metrics. FEATURE_COLS grows **83 → 88**.

---

## Section 1: Features

Five new columns added to `FUNDAMENTAL_FEATURE_COLS`, all ticker-specific (backward asof join, most recent quarter ≤ query date):

| Column | Formula | Signal |
|--------|---------|--------|
| `net_income_margin` | TTM Net Income / TTM Revenue | Profitability — distinguishes mature earners (MSFT, GOOGL) from growth-mode (AMD) |
| `free_cash_flow_margin` | (TTM Operating Income − TTM CapEx) / TTM Revenue | Capacity to self-fund AI capex without dilution |
| `capex_growth_yoy` | TTM CapEx[t] / TTM CapEx[t−4q] − 1 | Direct AI hardware acceleration signal — surges ahead of GPU procurement cycles |
| `revenue_growth_accel` | YoY growth[t] − YoY growth[t−1q] | Second derivative of AI monetization — early signal that the ramp is steepening |
| `research_to_revenue` | TTM R&D Expense / TTM Revenue | R&D intensity — NVDA/AMD/ASML bet size on next-gen silicon |

**Existing 9 columns retained unchanged:**
`pe_ratio_trailing`, `price_to_sales`, `price_to_book`, `revenue_growth_yoy`, `gross_margin`, `operating_margin`, `capex_to_revenue`, `debt_to_equity`, `current_ratio`

**Tier routing:** medium + long only. Quarterly cadence (~90-day data refresh lag) is too coarse for 5d/20d horizons.

**Null handling:**
- `research_to_revenue` = `0.0` when R&D concept absent from SEC XBRL (energy/hardware companies)
- `revenue_growth_accel` = `0.0` when only one YoY data point available
- `capex_growth_yoy` = `null` when <8 quarters of history; downstream imputation handles it
- `net_income_margin`, `free_cash_flow_margin` = `0.0` when denominator (revenue) is zero

---

## Section 2: Architecture & Data Flow

### New XBRL fetch

**`ingestion/edgar_fundamentals_ingestion.py`** (modify)

One additional SEC XBRL API call per ticker per run:
```
GET https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/ResearchAndDevelopmentExpense.json
```
- Same `time.sleep(0.15)` rate-limit already in place between per-ticker fetches
- If the concept returns 404 or empty data → R&D recorded as `0.0` (graceful fallback)
- TTM R&D computed same way as all other TTM metrics (sum of 4 most recent quarters)

**Expanded output schema (11 → 16 columns):**
```python
{
    "ticker": pl.Utf8,
    "period_end": pl.Date,
    # Existing 9 ratios
    "pe_ratio_trailing": pl.Float64,
    "price_to_sales": pl.Float64,
    "price_to_book": pl.Float64,
    "revenue_growth_yoy": pl.Float64,
    "gross_margin": pl.Float64,
    "operating_margin": pl.Float64,
    "capex_to_revenue": pl.Float64,
    "debt_to_equity": pl.Float64,
    "current_ratio": pl.Float64,
    # 5 new metrics
    "net_income_margin": pl.Float64,
    "free_cash_flow_margin": pl.Float64,
    "capex_growth_yoy": pl.Float64,     # nullable
    "revenue_growth_accel": pl.Float64,
    "research_to_revenue": pl.Float64,
}
```

**TTM computation for new metrics:**

```python
# net_income_margin
ttm_net_income = sum of NetIncomeLoss for 4 most recent quarters
net_income_margin = ttm_net_income / ttm_revenue  # ttm_revenue already computed

# free_cash_flow_margin
ttm_fcf = ttm_operating_income - ttm_capex  # both already computed
free_cash_flow_margin = ttm_fcf / ttm_revenue

# capex_growth_yoy
# current_ttm_capex = sum of 4 most recent quarters
# prior_ttm_capex = sum of quarters 5-8 (one year ago)
capex_growth_yoy = (current_ttm_capex / prior_ttm_capex) - 1.0
# → null if <8 quarters of CapEx history

# revenue_growth_accel
# revenue_growth_yoy already computed for current quarter
# prior_quarter_yoy = revenue_growth_yoy from the immediately preceding quarter
revenue_growth_accel = current_yoy - prior_quarter_yoy
# → 0.0 if only one quarter of YoY data

# research_to_revenue
ttm_rd = sum of ResearchAndDevelopmentExpense for 4 most recent quarters
research_to_revenue = ttm_rd / ttm_revenue
# → 0.0 if R&D concept unavailable
```

### Feature module

**`processing/fundamental_features.py`** (modify)

Expand `FUNDAMENTAL_FEATURE_COLS` from 9 to 14:
```python
FUNDAMENTAL_FEATURE_COLS: list[str] = [
    # Existing
    "pe_ratio_trailing", "price_to_sales", "price_to_book",
    "revenue_growth_yoy", "gross_margin", "operating_margin",
    "capex_to_revenue", "debt_to_equity", "current_ratio",
    # New
    "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
    "revenue_growth_accel", "research_to_revenue",
]
```

The backward asof join logic is unchanged — it joins all columns from the parquet onto the input DataFrame by (ticker, period_end ≤ date).

Zero-fill backstop: all 14 fundamental columns filled with `0.0` on null (existing behavior extended to new columns).

### Modified files

| File | Change |
|------|--------|
| `ingestion/edgar_fundamentals_ingestion.py` | Add R&D XBRL fetch + 5 new computed columns + expand schema 11→16 |
| `processing/fundamental_features.py` | Expand `FUNDAMENTAL_FEATURE_COLS` 9→14 |
| `models/train.py` | FEATURE_COLS 83→88, medium tier comment update, long tier append |
| `models/inference.py` | No change — fundamental join already dynamic from parquet columns |
| `tools/run_refresh.sh` | Add step 2/16 (EDGAR after OHLCV), renumber 2–15 → 3–16 |
| `tests/test_edgar_fundamentals_ingestion.py` | New — 5 ingestion tests |
| `tests/test_fundamental_features.py` | New — 5 feature tests for new columns |
| `tests/test_train.py` | Count 83→88 + 5 new fundamental tier/names tests |

**`run_refresh.sh` step order:**

EDGAR fundamentals must run after OHLCV (step 1) because it reads price data from `data/raw/ohlcv/` to compute P/E, P/S, P/B valuation ratios.

```
1/16  OHLCV price data
2/16  EDGAR fundamentals (NEW — reads OHLCV for valuation ratios)
3/16  Short interest
4/16  Earnings surprises
5/16  News articles
6/16  NLP sentiment scoring
7/16  Sentiment features
8/16  Graph features
9/16  13F institutional holdings
10/16 Ownership features
11/16 SAM.gov government contracts
12/16 FERC interconnection queue
13/16 USPTO patents
14/16 USAJOBS federal AI/ML postings
15/16 BLS JOLTS tech job openings
16/16 Census international trade
```

---

## Section 3: Implementation Details & Testing

### SEC XBRL API

- Base URL: `https://data.sec.gov/api/xbrl/companyconcept/`
- R&D concept: `us-gaap/ResearchAndDevelopmentExpense.json`
- Full URL: `https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/ResearchAndDevelopmentExpense.json`
- Rate limit: `time.sleep(0.15)` between ticker fetches (already implemented)
- Timeout: `30` seconds (already implemented)
- On 404 or empty response: log info and use `0.0` for all R&D quarters

### Staleness guard

Unchanged from existing implementation — same calendar-quarter check. Since fundamentals are published quarterly, re-downloading within the same quarter is skipped.

### Test plan

**Ingestion tests (5 tests) — `tests/test_edgar_fundamentals_ingestion.py`:**

1. Output schema has exactly 16 columns with correct Polars dtypes (11 existing + 5 new)
2. `net_income_margin` = TTM Net Income / TTM Revenue (±1e-6)
3. `free_cash_flow_margin` = (TTM Operating Income − TTM CapEx) / TTM Revenue
4. `capex_growth_yoy` correct from two consecutive 4-quarter windows; `null` when <8q
5. `research_to_revenue` = `0.0` when R&D concept returns empty/404

**Feature tests (5 tests) — `tests/test_fundamental_features.py`:**

1. `net_income_margin` backward asof picks most recent quarter ≤ query date (not future quarters)
2. `free_cash_flow_margin` correct value via asof join
3. `capex_growth_yoy` correct (positive when capex accelerating)
4. `revenue_growth_accel` = this quarter YoY − prior quarter YoY; `0.0` when only one YoY point
5. Missing fundamentals directory → all 14 columns zero-filled in output

**Train tests (5 new tests) — `tests/test_train.py`:**
- `test_feature_cols_includes_edgar_expanded`: count 83→88, all 14 FUNDAMENTAL cols present
- `test_edgar_expanded_cols_absent_from_short_tier`
- `test_edgar_expanded_cols_in_medium_tier`
- `test_edgar_expanded_cols_in_long_tier`
- `test_edgar_expanded_col_names_correct`: exact set of 14 names

---

## Out of Scope

- `fundamental_ingestion.py` (yfinance-based, 24 tickers) — not added to run_refresh.sh; superseded by edgar_fundamentals for all 83 tickers. File retained but unused.
- Additional XBRL concepts (operating cash flow, asset turnover, inventory) — can add later
- Per-segment revenue breakdown — requires XBRL extension taxonomy, complex parsing
- Annual filing data (10-K) — quarterly (10-Q) data is sufficient for the signal cadence
