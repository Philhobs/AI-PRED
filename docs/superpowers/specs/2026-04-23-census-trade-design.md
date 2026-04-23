# Census Trade Signals — Design Spec

**Date:** 2026-04-23
**Status:** Approved

---

## What We're Building

One new ingestion module and one feature module deriving 6 `CENSUS_TRADE_FEATURE_COLS` from US International Trade data (Census Bureau). FEATURE_COLS grows from 77 → 83. All 6 features are market-wide (joined on date only); features route to medium + long tiers only.

---

## Section 1: Features

Six new columns in `CENSUS_TRADE_FEATURE_COLS`, all market-wide:

| Column | Source | Description |
|--------|--------|-------------|
| `semicon_import_value` | Census imports, HS 8541+8542, all partners | US semiconductor import value, most recent month ≤ query date (USD millions) |
| `semicon_import_momentum` | Same | MoM change: current month minus previous month (USD millions) |
| `dc_equipment_import_value` | Census imports, HS 8471+8473, all partners | Data center equipment import value, most recent month (USD millions) |
| `dc_equipment_import_momentum` | Same | MoM change (USD millions) |
| `china_semicon_export_share` | Census exports, HS 8541+8542, China (CTY_CODE=5700) vs total | US semiconductor exports to China / total US semiconductor exports (0–1 ratio) |
| `taiwan_semicon_import_share` | Census imports, HS 8541+8542, Taiwan (CTY_CODE=5830) vs total | Taiwan's share of US semiconductor imports (0–1 ratio, supply concentration risk) |

**Signal interpretation:**
- `semicon_import_*`: chip supply flowing into US — leading indicator for GPU/AI hardware availability and capex cycle
- `dc_equipment_import_*`: server/data center hardware capex — hyperscaler spend proxy
- `china_semicon_export_share`: drops sharply when export controls tighten — risk signal for NVDA, ASML, LRCX
- `taiwan_semicon_import_share`: supply concentration — rises mean higher geopolitical vulnerability for the whole supply chain

**HS codes:**
- Semiconductors: `8541` (semiconductor devices, LEDs, solar cells), `8542` (electronic integrated circuits — GPUs, CPUs, memory)
- Data center equipment: `8471` (computers, servers, ADP machines), `8473` (parts and accessories for ADP machines)

**Country codes (Census CTY_CODE):**
- China: `5700`
- Taiwan: `5830`
- All partners: omit `CTY_CODE` parameter

**Tier routing:** medium + long only. Monthly Census data (~6–8 week publication lag) is too slow for 5d/20d horizons.

**Null handling:** all 6 features zero-filled when no data is available.

---

## Section 2: Architecture & Data Flow

### New ingestion module

**`ingestion/census_trade_ingestion.py`**

- Imports API: `GET https://api.census.gov/data/timeseries/intltrade/imports`
- Exports API: `GET https://api.census.gov/data/timeseries/intltrade/exports`
- Optional free key: `CENSUS_API_KEY` env var (raises rate limits; falls back gracefully if absent)
- 10 targeted queries per run:
  1. Imports, HS 8541, all partners
  2. Imports, HS 8542, all partners
  3. Imports, HS 8471, all partners
  4. Imports, HS 8473, all partners
  5. Imports, HS 8541, Taiwan (CTY_CODE=5830)
  6. Imports, HS 8542, Taiwan (CTY_CODE=5830)
  7. Exports, HS 8541, all partners
  8. Exports, HS 8542, all partners
  9. Exports, HS 8541, China (CTY_CODE=5700)
  10. Exports, HS 8542, China (CTY_CODE=5700)
- 12-month lookback per run: `time=from+{year-1}-{month:02d}+to+{year}-{month:02d}`
- Rate limit: `time.sleep(0.5)` between queries
- Staleness: skip if same-calendar-month snapshot exists (identical to BLS JOLTS guard)
- Output: `data/raw/census_trade/date=YYYY-MM-DD/trade.parquet`
- Schema: `{date: Date, direction: Utf8, hs_code: Utf8, partner_code: Utf8, year: Int32, month: Int32, value_usd: Float64}`
  - `direction`: `"import"` or `"export"`
  - `partner_code`: `"ALL"` for all-partner queries, `"5700"` for China, `"5830"` for Taiwan
  - `value_usd`: raw USD value (converted to millions in feature module)
- Empty results → no file written

### Feature module

**`processing/census_trade_features.py`**

Single public function:
```python
def join_census_trade_features(
    df: pl.DataFrame,
    census_trade_dir: Path,
) -> pl.DataFrame:
    """Left-join Census trade features to df. Missing rows zero-filled."""
```

DuckDB window function approach (same as `labor_features.py` JOLTS queries):
- `ROW_NUMBER() OVER (PARTITION BY q.date ORDER BY period_date DESC)` to pick most recent month ≤ query date
- `MAKE_DATE(year, month, 1)` to convert year/month integers to comparable dates
- Values divided by 1,000,000 in SQL to produce USD millions
- Share features: filtered sum / total sum, with `GREATEST(total, 1.0)` denominator to avoid division by zero
- Zero-fill backstop at end via `fill_null(0.0)`

**`semicon_import_value` / `semicon_import_momentum` formula:**
```sql
WITH monthly AS (
    SELECT year, month, SUM(value_usd)/1e6 AS value_m
    FROM trade
    WHERE direction='import' AND hs_code IN ('8541','8542') AND partner_code='ALL'
    GROUP BY year, month
),
dated AS (SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly),
ranked AS (
    SELECT q.date, d.value_m,
        ROW_NUMBER() OVER (PARTITION BY q.date ORDER BY d.period_date DESC) AS rn
    FROM query_dates q CROSS JOIN dated d WHERE d.period_date <= q.date
)
SELECT date,
    COALESCE(MAX(CASE WHEN rn=1 THEN value_m END), 0.0) AS semicon_import_value,
    CASE WHEN MAX(CASE WHEN rn=2 THEN value_m END) IS NULL THEN 0.0
         ELSE MAX(CASE WHEN rn=1 THEN value_m END) - MAX(CASE WHEN rn=2 THEN value_m END)
    END AS semicon_import_momentum
FROM ranked GROUP BY date
```

`dc_equipment_import_value` / `dc_equipment_import_momentum`: identical pattern with `hs_code IN ('8471','8473')`.

**`china_semicon_export_share` formula:**
```sql
WITH monthly AS (
    SELECT year, month,
        SUM(CASE WHEN partner_code='5700' THEN value_usd ELSE 0 END) AS china_val,
        SUM(value_usd) AS total_val
    FROM trade
    WHERE direction='export' AND hs_code IN ('8541','8542')
    GROUP BY year, month
),
dated AS (SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly),
ranked AS (
    SELECT q.date, d.china_val, d.total_val,
        ROW_NUMBER() OVER (PARTITION BY q.date ORDER BY d.period_date DESC) AS rn
    FROM query_dates q CROSS JOIN dated d WHERE d.period_date <= q.date
)
SELECT date,
    COALESCE(MAX(CASE WHEN rn=1 THEN china_val END), 0.0)
    / GREATEST(COALESCE(MAX(CASE WHEN rn=1 THEN total_val END), 1.0), 1.0)
    AS china_semicon_export_share
FROM ranked GROUP BY date
```

`taiwan_semicon_import_share`: identical pattern with `direction='import'` and `partner_code='5830'`.

All 6 features joined on `date` only (market-wide). Zero-fill backstop at end.

### Modified files

| File | Change |
|------|--------|
| `models/train.py` | Import + FEATURE_COLS 77→83 + medium comment update + long tier + join call |
| `models/inference.py` | Import + join call |
| `tests/test_train.py` | Count 77→83 + 5 new CENSUS tests |
| `tools/run_refresh.sh` | Renumber X/14 → X/15, add step 15/15 Census |

---

## Section 3: Implementation Details & Testing

### Census API

- Import value field: `GEN_VAL_MO` (general imports, monthly USD)
- Export value field: `ALL_VAL_MO` (all exports, monthly USD)
- `COMM_LVL=HS4` — 4-digit HS heading level
- Time range parameter: `time=from+YYYY-MM+to+YYYY-MM`
- Query parameters: `get=GEN_VAL_MO` (or `ALL_VAL_MO`), `COMM_LVL=HS4`, `E_COMMODITY={hs_code}`, optionally `CTY_CODE={country}`, `time=...`, optionally `key={CENSUS_API_KEY}`
- Response format: JSON array, first row is headers, subsequent rows are data
- `timeout=30` on all requests
- `time.sleep(0.5)` between queries (10 queries × 0.5s = conservative, well within Census rate limits)

### Staleness check

Same calendar month as BLS JOLTS: check if most recent `date=*/trade.parquet` snapshot directory date shares year+month with today. If so, skip all 10 queries.

### Test plan

**Ingestion tests (5 tests):**
1. Schema columns and dtypes correct (`date`, `direction`, `hs_code`, `partner_code`, `year`, `month`, `value_usd`)
2. `direction`, `hs_code`, `partner_code` stored correctly for import and export rows
3. Same-month snapshot → download skipped (requests.get not called)
4. Empty API response → no file written
5. `time.sleep(0.5)` called 9 times (between 10 queries, not after the last)

**Feature tests (9 tests):**
1. `semicon_import_value` uses most recent month ≤ query date (not future months)
2. `semicon_import_momentum` = current minus previous month value
3. `semicon_import_momentum` = 0.0 when only one month available (no spurious index-as-momentum)
4. `dc_equipment_import_value` correct (HS 8471+8473 summed, not semiconductor codes)
5. `dc_equipment_import_momentum` correct
6. `china_semicon_export_share` = China exports / total exports, value in [0, 1]
7. `china_semicon_export_share` = 0.0 when no export data (denominator guard)
8. `taiwan_semicon_import_share` correct ratio
9. Join adds exactly 6 columns; missing `census_trade_dir` → zero-fill all 6

**Train tests (5 new):**
- `test_feature_cols_includes_census` (count 77→83, all 6 present)
- `test_census_cols_absent_from_short_tier`
- `test_census_cols_in_medium_tier`
- `test_census_cols_in_long_tier`
- `test_census_col_names_correct`

### `run_refresh.sh` addition

```bash
echo ""
echo "=== 15/15  Census international trade (semiconductors + DC equipment) ==="
python ingestion/census_trade_ingestion.py
```

Added after step 14 (BLS JOLTS), before "Refresh complete" message. All prior steps renumbered X/15.

---

## Out of Scope

- Sub-4-digit HS granularity (individual subheadings like 854230 — adds complexity without meaningful signal improvement)
- Country pairs beyond China and Taiwan (EU, South Korea, Japan — can add later if needed)
- Critical minerals (HS 2846, 2615) — lower signal quality for AI infra stocks; deferred
- Import/export quantity (kg) — value in USD is the right unit for financial signals
- Bilateral trade deficits or trade balance calculations
- WTO/tariff rate data
