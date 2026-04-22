# Government Behavioral Data — Design Spec

**Date:** 2026-04-23
**Status:** Approved
**Follow-up:** USPTO patents spec (separate)

---

## What We're Building

Two new ingestion modules (SAM.gov federal contract awards, FERC interconnection queue) and one feature module that derives 6 `GOV_BEHAVIORAL_FEATURE_COLS` from them. FEATURE_COLS grows from 61 → 67. Features route to medium + long tiers only.

---

## Section 1: Features

Six new columns in `GOV_BEHAVIORAL_FEATURE_COLS`:

| Column | Source | Join type | Description |
|--------|--------|-----------|-------------|
| `gov_contract_value_90d` | SAM.gov | ticker-specific | Rolling 90-day sum of USD contract awards to the company |
| `gov_contract_count_90d` | SAM.gov | ticker-specific | Rolling 90-day count of contract awards |
| `gov_contract_momentum` | SAM.gov | ticker-specific | Recent 30d award value minus prior 60d (momentum signal) |
| `gov_ai_spend_30d` | SAM.gov | market-wide | Total AI/DC NAICS spend across all awardees in 30-day window |
| `ferc_queue_mw_30d` | FERC queue | market-wide | MW of new interconnection requests filed in DC power states in 30 days |
| `ferc_grid_constraint_score` | FERC queue | market-wide | Ratio of queued MW to historical capacity additions (grid congestion proxy) |

NAICS codes for AI/datacenter contracts: `541511`, `541512`, `541519` (custom software/IT services), `518210` (data processing/hosting), `334413` (semiconductor manufacturing).

DC power states for FERC: VA, TX, OH, AZ, NV, OR, GA, WA.

**Tier routing:** medium + long only. Short tier is unchanged (options + price signals).

**Null handling:** all 6 features zero-filled when no data is available.

---

## Section 2: Architecture & Data Flow

### New ingestion modules

**`ingestion/sam_gov_ingestion.py`**
- Calls SAM.gov awards API with NAICS filter + rolling date range
- Auth: `SAM_GOV_API_KEY` env var (free, self-service at sam.gov/content/duns-sam)
- Output: `data/raw/gov_contracts/date=YYYY-MM-DD/awards.parquet`
- Schema: `{date: Date, awardee_name: Utf8, uei: Utf8, contract_value_usd: Float64, naics_code: Utf8, agency: Utf8}`
- Raises `RuntimeError` if `SAM_GOV_API_KEY` not set

**`ingestion/ferc_queue_ingestion.py`**
- Downloads Lawrence Berkeley Lab "Queued Up" semi-annual Excel (~3MB)
- URL configurable via `FERC_QUEUE_URL` env var; hardcoded LBL URL as default
- Staleness check: skips re-download if existing parquet has same half-year snapshot date
- Output: `data/raw/ferc_queue/date=YYYY-MM-DD/queue.parquet`
- Schema: `{snapshot_date: Date, queue_date: Date, project_name: Utf8, mw: Float64, state: Utf8, fuel: Utf8, status: Utf8, iso: Utf8}`

### Feature module

**`processing/gov_behavioral_features.py`**

Ticker matching pipeline:
1. Check `GOV_TICKER_OVERRIDE_MAP` (known mismatches)
2. Strip legal suffixes (Inc, LLC, Corp, Ltd)
3. `difflib.get_close_matches(threshold=0.85)` against awardee names
4. No match → ticker gets zero values

```python
GOV_TICKER_OVERRIDE_MAP = {
    "GOOGL": "Alphabet",
    "META":  "Meta Platforms",
    "MSFT":  "Microsoft Corporation",
    "AMZN":  "Amazon Web Services",
    "TSM":   "Taiwan Semiconductor",
}
```

Mixed join strategy:
- 3 ticker-specific columns joined on `(ticker, date)`
- 3 market-wide columns joined on `date` only

### Modified files

| File | Change |
|------|--------|
| `models/train.py` | Import + join call + FEATURE_COLS 61→67 |
| `models/inference.py` | Add join call |
| `tests/test_train.py` | Update count assertion + 6 new GOV tests |
| `tools/run_refresh.sh` | Add 2 ingestion lines |

---

## Section 3: Implementation Details & Testing

### SAM.gov API

- Base URL: `https://api.sam.gov/opportunities/v2/search`
- Auth: `X-Api-Key` header
- Date filter: `awardDateRange=YYYY-MM-DD,YYYY-MM-DD` (rolling 90-day window)
- NAICS filter: `naicsCode=541511,541512,541519,518210,334413`
- Pagination: `limit=100`, `offset` cursor; stop when `totalRecords` reached
- Rate limit: 10 req/min → `time.sleep(6.0)` between pages

### FERC/LBL dataset

- Source: Lawrence Berkeley National Lab "Queued Up" Excel
- URL hardcoded in module, overridable via `FERC_QUEUE_URL` env var
- Parsed with `openpyxl` (transitive dep via pandas)
- Staleness: compare `snapshot_date` in existing parquet vs. current half-year (Jan/Jul update cycle)

### Test plan

**SAM ingestion (7 tests):**
- Schema columns and dtypes correct
- Pagination followed when `totalRecords > limit`
- `time.sleep` called between pages (rate limit)
- Empty awards page → empty DataFrame (no file written)
- Missing `SAM_GOV_API_KEY` → `RuntimeError`
- NAICS filter present in request params
- Date-range param covers 90 days

**FERC ingestion (5 tests):**
- Schema columns and dtypes correct
- Same half-year snapshot → download skipped
- State filter returns only DC states
- Empty queue sheet → empty DataFrame
- Bad URL → `RuntimeError`

**Feature tests (10 tests):**
- `gov_contract_value_90d` sums correctly over window
- `gov_contract_count_90d` count matches fixture rows
- `gov_contract_momentum` positive when recent 30d > prior 60d
- `gov_ai_spend_30d` sums across all tickers
- `ferc_queue_mw_30d` sums only DC-state MW
- `ferc_grid_constraint_score` positive when queue exceeds threshold
- Join adds exactly 6 columns
- Ticker with no contract history → zero-fill
- Date with no FERC snapshot → zero-fill
- GOV cols absent from short tier, present in medium + long

### `run_refresh.sh` additions

```bash
python ingestion/sam_gov_ingestion.py
python ingestion/ferc_queue_ingestion.py
```

Added after existing ingestion calls, before `processing/feature_engineering.py`.

---

## Out of Scope

- USPTO patent ingestion (separate follow-up spec)
- Historical backfill beyond the rolling windows
- Paid SAM.gov data feeds
