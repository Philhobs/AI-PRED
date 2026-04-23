# USPTO Patent Signals — Design Spec

**Date:** 2026-04-23
**Status:** Approved

---

## What We're Building

One new ingestion module (`ingestion/uspto_ingestion.py`) fetching published patent applications and granted patents from the PatentsView v2 API, and one feature module (`processing/patent_features.py`) that derives 6 `USPTO_PATENT_FEATURE_COLS` from them. FEATURE_COLS grows from 67 → 73. All 6 features are ticker-specific; features route to medium + long tiers only.

---

## Section 1: Features

Six new columns in `USPTO_PATENT_FEATURE_COLS`, all ticker-specific (no market-wide columns — patent filings are company-specific signals):

| Column | Window | Description |
|--------|--------|-------------|
| `patent_app_count_90d` | 90d | Count of published AI/semiconductor patent applications filed by the company |
| `patent_app_momentum` | 90d vs prior 90d | Recent 90d application count minus prior 90d (R&D acceleration signal) |
| `patent_grant_count_365d` | 365d | Count of granted AI/semiconductor patents |
| `patent_grant_rate_365d` | 365d | `grants_365d / max(apps_365d, 1)` — quality ratio (high = strong IP portfolio) |
| `patent_ai_cpc_share_90d` | 90d | Fraction of 90d applications in G06N (AI/ML) vs all covered CPC codes |
| `patent_citation_count_365d` | 365d | Forward citations received on patents granted in the 365d window |

**CPC codes covered:**
- `G06N` — Computing; Artificial intelligence / machine learning
- `H01L` — Semiconductor devices
- `G06F` — Electric digital data processing (computing hardware)
- `G11C` — Static information storage (memory)

**AI-specific CPC codes** (for `patent_ai_cpc_share_90d`): `G06N` only.

**Tier routing:** medium + long only. Patent cycles are too slow for 5d/20d horizons.

**Null handling:** all 6 features zero-filled when no data is available.

---

## Section 2: Architecture & Data Flow

### New ingestion module

**`ingestion/uspto_ingestion.py`**

- Two PatentsView v2 API calls (free, no API key required):
  - Applications: `POST https://api.patentsview.org/applications/query`
  - Grants: `POST https://api.patentsview.org/patents/query`
- Rolling 365-day lookback on each run (covers all three window sizes at query time)
- Staleness check: skip re-fetch if existing parquet was written within the same calendar week (PatentsView updates weekly; compare ISO week number of existing snapshot vs. today)
- Output:
  - `data/raw/patents/applications/date=YYYY-MM-DD/apps.parquet`
  - `data/raw/patents/grants/date=YYYY-MM-DD/grants.parquet`
- Schema (applications): `{date: Date, assignee_name: Utf8, app_id: Utf8, cpc_group: Utf8, filing_date: Date}`
- Schema (grants): `{date: Date, assignee_name: Utf8, patent_id: Utf8, cpc_group: Utf8, grant_date: Date, forward_citation_count: Int32}`
- Rate limit: 45 req/min → `time.sleep(1.5)` between pages
- Pagination: `per_page=100`, `page` cursor; stop when `total_patent_count` (or `total_app_count`) reached

**PatentsView API query structure (applications):**
```json
{
  "q": {"_and": [
    {"_gte": {"app_date": "YYYY-MM-DD"}},
    {"_lte": {"app_date": "YYYY-MM-DD"}},
    {"_or": [
      {"cpc_group_id": "G06N"}, {"cpc_group_id": "H01L"},
      {"cpc_group_id": "G06F"}, {"cpc_group_id": "G11C"}
    ]}
  ]},
  "f": ["app_id", "assignee_organization", "cpc_group_id", "app_date"],
  "o": {"per_page": 100, "page": 1}
}
```

Grants query uses the same structure against `/patents/query` with `patent_date` field; additionally requests `cited_by_count` for forward citations.

### Feature module

**`processing/patent_features.py`**

Ticker matching pipeline (same as `gov_behavioral_features.py`):
1. Check `_TICKER_NAME_MAP` (imported from `gov_behavioral_features.py` — no duplication)
2. Strip legal suffixes
3. `difflib.get_close_matches(cutoff=0.85)` against assignee names
4. No match → ticker gets zero values

Rolling aggregations via DuckDB cross-join (same pattern as GOV behavioral — handles query dates not present in raw data).

Single public function:
```python
def join_patent_features(
    df: pl.DataFrame,
    apps_dir: Path,
    grants_dir: Path,
) -> pl.DataFrame:
    """Left-join USPTO patent features to df. Missing rows zero-filled."""
```

`patent_grant_rate_365d` formula:
```python
grants_365d / GREATEST(apps_365d, 1)
```
(GREATEST prevents division by zero without clipping the ratio.)

`patent_ai_cpc_share_90d` formula:
```python
g06n_apps_90d / GREATEST(total_apps_90d, 1)
```

### Modified files

| File | Change |
|------|--------|
| `models/train.py` | Import + join call + FEATURE_COLS 67→73 + long tier update |
| `models/inference.py` | Import + join call + docstring 67→73 |
| `tests/test_train.py` | Update count assertion 67→73 + 6 new USPTO tests |
| `tools/run_refresh.sh` | Add USPTO ingestion step (12/12) |

---

## Section 3: Implementation Details & Testing

### PatentsView v2 API

- Base URLs:
  - Applications: `https://api.patentsview.org/applications/query`
  - Patents (grants): `https://api.patentsview.org/patents/query`
- Method: POST, `Content-Type: application/json`
- No authentication required (v2 API)
- Date filter fields: `app_date` (applications), `patent_date` (grants)
- Lookback: rolling 365-day window from `date_str`
- CPC filter: `_or` across `G06N`, `H01L`, `G06F`, `G11C`
- Pagination: `per_page=100`, `page` increments; stop when total records reached
- Rate limit: 45 req/min → `time.sleep(1.5)` between pages

### Staleness check

Compare ISO week of the most recent existing parquet's `date` column vs. `date.isocalendar().week` of today. Same ISO week → skip re-download.

### Ticker matching

Import `_TICKER_NAME_MAP` directly from `processing.gov_behavioral_features` to avoid duplication. Same `_normalize_name` logic applies. PatentsView assignee names follow similar patterns to SAM.gov (e.g., "NVIDIA Corporation", "Microsoft Technology Licensing LLC").

### Test plan

**Ingestion (6 tests):**
- Applications schema columns and dtypes correct
- Grants schema columns and dtypes correct
- Pagination followed when total > per_page
- Same-week snapshot → download skipped
- Empty results → empty DataFrame, no file written
- Rate limit: `time.sleep` called between pages

**Feature tests (10 tests):**
- `patent_app_count_90d` sums only within 90d window (not 180d or 365d)
- `patent_app_momentum` positive when recent 90d > prior 90d
- `patent_grant_count_365d` sums correctly over full year
- `patent_grant_rate_365d` equals grants/apps (zero-safe denominator)
- `patent_ai_cpc_share_90d` isolates G06N fraction correctly (non-G06N excluded)
- `patent_citation_count_365d` sums forward citations within 365d grant window
- Join adds exactly 6 columns
- Ticker with no patents → zero-fill all 6
- Missing apps_dir → zero-fill all 6
- USPTO cols absent from short tier, present in medium + long

### `run_refresh.sh` addition

```bash
echo ""
echo "=== 12/12  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py
```

Added after step 11 (FERC queue), before "Refresh complete" message. All prior steps renumbered X/12.

---

## Out of Scope

- USPTO patent litigation / IPR proceedings
- Patent valuation or monetization signals
- International patent offices (EPO, WIPO)
- Paid PatentsView premium data feeds
