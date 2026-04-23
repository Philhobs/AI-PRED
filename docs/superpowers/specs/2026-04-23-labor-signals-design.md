# Labor Market Signals — Design Spec

**Date:** 2026-04-23
**Status:** Approved

---

## What We're Building

Two new ingestion modules and one feature module deriving 4 `LABOR_FEATURE_COLS` from federal job posting data (USAJOBS) and BLS tech-sector job openings (JOLTS). FEATURE_COLS grows from 73 → 77. All 4 features are market-wide (joined on date only); features route to medium + long tiers only.

---

## Section 1: Features

Four new columns in `LABOR_FEATURE_COLS`, all market-wide (no ticker-specific columns — no free API provides company-specific job posting counts):

| Column | Source | Window | Description |
|--------|--------|--------|-------------|
| `gov_ai_hiring_30d` | USAJOBS | 30d | Count of federal AI/ML job postings in 30d rolling window |
| `gov_ai_hiring_momentum` | USAJOBS | 30d vs prior 30d | Recent 30d count minus prior 30d count (government AI investment signal) |
| `tech_job_openings_index` | BLS JOLTS | Latest month ≤ query date | Computer & electronic products (NAICS 334) job openings, thousands |
| `tech_job_openings_momentum` | BLS JOLTS | MoM | Current month openings minus previous month (hiring acceleration) |

**Signal interpretation:**
- `gov_ai_hiring_*`: government AI talent investment — complements `gov_ai_spend_30d`; leads procurement by weeks to months
- `tech_job_openings_*`: private sector tech hiring cycle — leading indicator for AI infrastructure capex

**AI/ML keywords for USAJOBS** (OR filter, case-insensitive):
- `"artificial intelligence"`, `"machine learning"`, `"deep learning"`, `"GPU computing"`, `"semiconductor"`

**BLS JOLTS series:** `JTS510000000000000JOL` — Computer and Electronic Products (NAICS 334), Job Openings Level (thousands, seasonally adjusted)

**Tier routing:** medium + long only. Monthly JOLTS data and 30-day USAJOBS aggregates are too slow for 5d/20d horizons.

**Null handling:** all 4 features zero-filled when no data is available.

---

## Section 2: Architecture & Data Flow

### New ingestion modules

**`ingestion/usajobs_ingestion.py`**

- API: `GET https://data.usajobs.gov/api/search`
- No API key required — needs only `User-Agent` header (email string, configurable via `USAJOBS_USER_AGENT` env var)
- 5 keyword queries per run (one per AI/ML term), deduplicated on `posting_id`
- `DatePosted=30` returns postings from the last 30 days; `ResultsPerPage=500`
- Pagination: `Page` cursor; stop when all results fetched
- Rate limit: `time.sleep(1.0)` between keyword queries
- Staleness: skip if same-calendar-week snapshot exists
- Output: `data/raw/usajobs/date=YYYY-MM-DD/postings.parquet`
- Schema: `{date: Date, posting_id: Utf8, title: Utf8, posted_date: Date, keyword: Utf8}`
- Empty results → no file written

**`ingestion/bls_jolts_ingestion.py`**

- API: `POST https://api.bls.gov/publicAPI/v2/timeseries/data/`
- Optional free key: `BLS_API_KEY` env var (raises registered API rate limits; falls back gracefully if absent)
- Series: `JTS510000000000000JOL`
- 12-month lookback on each run (`startyear`, `endyear` derived from `date_str`)
- Staleness: skip if most recent parquet's max `(year, period)` matches current year-month
- Output: `data/raw/bls_jolts/date=YYYY-MM-DD/openings.parquet`
- Schema: `{date: Date, series_id: Utf8, year: Int32, period: Utf8, value: Float64}`
- `period` format: `M01`–`M12`; store as string (used for staleness check and ordering)
- Empty series → no file written

### Feature module

**`processing/labor_features.py`**

Single public function:
```python
def join_labor_features(
    df: pl.DataFrame,
    usajobs_dir: Path,
    jolts_dir: Path,
) -> pl.DataFrame:
    """Left-join labor market features to df. Missing rows zero-filled."""
```

Rolling aggregations via DuckDB cross-join (same pattern as `gov_behavioral_features.py`).

**`gov_ai_hiring_30d`** formula:
```
COUNT postings WHERE posted_date IN [query_date - 30d, query_date]
```

**`gov_ai_hiring_momentum`** formula:
```
recent_30d_count - prior_30d_count  (prior window: 31d–60d before query_date)
```

**`tech_job_openings_index`** formula:
```
value of most recent JOLTS month WHERE period_date <= query_date
```
(Convert `period` M01–M12 to `datetime.date(year, month, 1)` for date comparison.)

**`tech_job_openings_momentum`** formula:
```
current_month_value - previous_month_value
```
(Previous month = second most recent JOLTS month ≤ query_date.)

All 4 features joined on `date` only (market-wide). Zero-fill backstop at end.

### Modified files

| File | Change |
|------|--------|
| `models/train.py` | Import + FEATURE_COLS 73→77 + medium comment + long tier + join call |
| `models/inference.py` | Import + join call |
| `tests/test_train.py` | Count 73→77 + 5 new LABOR tests |
| `tools/run_refresh.sh` | Add 2 new steps (13/14 USAJOBS, 14/14 BLS JOLTS), renumber prior steps to X/14 |

---

## Section 3: Implementation Details & Testing

### USAJOBS API

- Base URL: `https://data.usajobs.gov/api/search`
- Required headers: `Host: data.usajobs.gov`, `User-Agent: {USAJOBS_USER_AGENT}`
- Keywords queried (5 sequential GET requests):
  1. `"artificial intelligence"`
  2. `"machine learning"`
  3. `"deep learning"`
  4. `"GPU computing"`
  5. `"semiconductor"`
- Parameters per request: `Keyword={term}`, `DatePosted=30`, `ResultsPerPage=500`, `Page={n}`
- Dedup: after collecting all pages for all keywords, deduplicate on `PositionID`
- `posted_date` parsed from `PublicationStartDate` (ISO 8601 string)

### BLS JOLTS API

- Method: POST, `Content-Type: application/json`
- Body:
```json
{
  "seriesid": ["JTS510000000000000JOL"],
  "startyear": "YYYY-1",
  "endyear": "YYYY",
  "registrationkey": "optional"
}
```
- Response path: `data["Results"]["series"][0]["data"]` — list of `{year, period, value, ...}`
- `period`: `"M01"` through `"M12"` → `datetime.date(int(year), int(period[1:]), 1)`
- Staleness: compare `(max_year, max_period)` from existing parquet vs `(today.year, f"M{today.month:02d}")`; same → skip

### Staleness checks

- USAJOBS: same ISO week (weekly posting cycles) — same logic as `ingestion/uspto_ingestion.py`
- BLS JOLTS: same calendar month — compare `(year, period)` of most recent row vs current month

### Test plan

**USAJOBS ingestion (5 tests):**
- Schema columns and dtypes correct
- Dedup removes duplicate `posting_id` across keyword queries
- Same-week snapshot → download skipped
- Empty results → no file written
- `time.sleep(1.0)` called between keyword queries

**BLS JOLTS ingestion (4 tests):**
- Schema columns and dtypes correct
- `period` field stored as string `M01`–`M12`
- Same-month snapshot → download skipped
- Empty series → no file written

**Feature tests (9 tests):**
- `gov_ai_hiring_30d` counts only postings within 30d window (not older)
- `gov_ai_hiring_momentum` positive when recent 30d > prior 30d
- `tech_job_openings_index` uses most recent JOLTS month ≤ query date (not future months)
- `tech_job_openings_momentum` equals current minus previous month value
- Join adds exactly 4 columns
- Missing `usajobs_dir` → zero-fill all 4
- Missing `jolts_dir` → zero-fill all 4
- All 4 zero-filled when no data falls within window
- LABOR cols absent from short tier, present in medium + long

### `run_refresh.sh` additions

```bash
echo ""
echo "=== 13/14  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 14/14  BLS JOLTS tech sector job openings ==="
python ingestion/bls_jolts_ingestion.py
```

Added after step 12 (USPTO), before "Refresh complete" message. All prior steps renumbered X/14.

---

## Out of Scope

- Company-specific private sector job postings (no free API available)
- BLS OES (Occupational Employment Statistics) — annual frequency, too infrequent
- LinkedIn, Indeed, Glassdoor job data (paid APIs)
- International labor market data
- Job posting text NLP / skills extraction
