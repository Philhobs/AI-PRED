# Physical AI Signals — Design Spec (Robotics Spec 2)

**Date:** 2026-04-25
**Status:** Approved
**Sequencing:** Spec 2 of 2 in the Robotics expansion. Spec 1 (Robotics Layer Expansion, 2026-04-25) merged first.

## Goal

Build a **physical-AI signals layer** capturing the macro + innovation backdrop driving robotics, autonomy, and AI-vision growth. Output: 21 new feature columns (`FEATURE_COLS` 88 → 109) applied uniformly across all 149 tickers — the model decides per-ticker relevance.

## Motivation

The robotics-pillar tickers (industrial automation, medical/humanoid, MCU chips) plus key chip suppliers (NVDA, TSM, AVGO) and hyperscalers (MSFT/AMZN/GOOGL with their robotics platforms) all benefit from the same underlying growth: physical AI replacing labor across manufacturing, aerial logistics, autonomous vehicles, and surgery. This spec captures three feature classes that represent leading and coincident signals of that growth:

1. **Macro demand** — capital-goods orders, manufacturing PMI, industrial machinery output and pricing power.
2. **Labor demand** — manufacturing job openings (NAICS 333) as adoption proxy.
3. **Innovation pace** — quarterly USPTO patent filings across 6 CPC classes spanning manipulators, drones, AVs, motion control, programme control, and AI vision.

No assignee whitelist on patents — capture industry-wide growth (a Waymo or Skydio patent surge benefits NVDA and TXN even though those companies aren't in our registry).

## Data sources (all free, no key required)

### FRED (4 series)

| Series ID | Description | Cadence | Pub lag |
|---|---|---|---|
| `NEWORDER` | Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft | Monthly | ~5 weeks |
| `NAPM` | ISM Manufacturing PMI (headline) | Monthly | ~1st of month |
| `IPG3331S` | Industrial Production: Industrial Machinery | Monthly | ~5 weeks |
| `WPU114` | PPI: Industrial Machinery | Monthly | ~mid-month |

**Fallback for `NAPM`:** if not reachable, try `USAPMI`. If neither, log warning and ship 3 FRED series; tests adjust to 108 features.

### BLS JOLTS (1 new series)

| Series ID | Description |
|---|---|
| `JTS333000000000000JOL` | NAICS 333 — Machinery Manufacturing — Job openings level |

Existing `bls_jolts_ingestion.py` already fetches NAICS 51 (Information). This adds NAICS 333 alongside, no schema change.

### USPTO PatentsView — 6 CPC class buckets

| Bucket | CPC classes | Coverage |
|---|---|---|
| `B25J` | B25J* | Manipulators / industrial robots |
| `B64` | B64C*, B64U* | Aircraft + unmanned aerial vehicles (drones) |
| `B60W` | B60W* | Vehicle dynamic / autonomous control |
| `G05D1` | G05D1* | Motion / position control |
| `G05B19` | G05B19* | Programme control of machinery (CNC + robotics software) |
| `G06V` | G06V* | Image / video AI (perception) |

**No filters:** no assignee whitelist, no co-classification requirement. Industry-wide quarterly counts.

## Modules — new and modified

| Module | Role |
|---|---|
| **`ingestion/robotics_signals_ingestion.py`** *(new)* | Fetches the 4 FRED series. Saves to `data/raw/robotics_signals/{series_id}.parquet`. Schema: `date (Date), value (Float64)`. |
| **`ingestion/bls_jolts_ingestion.py`** *(modify)* | Add NAICS 333 alongside NAICS 51 — no schema change. |
| **`ingestion/uspto_ingestion.py`** *(modify)* | Add a "physical AI" mode that pulls quarterly filing counts per CPC class for 6 buckets. New parquet path: `data/raw/uspto/physical_ai/cpc_class={B25J\|B64\|B60W\|G05D1\|G05B19\|G06V}/quarter=YYYY-Qn.parquet`. Schema: `quarter_end (Date), cpc_class (Utf8), filing_count (Int64)`. |
| **`processing/physical_ai_features.py`** *(new)* | Single feature module covering all signals from this spec. Loads FRED + JOLTS NAICS 333 + patent counts; joins to ticker spine via as-of date. Public entry: `join_physical_ai_features(spine: pl.DataFrame, raw_dir: Path) -> pl.DataFrame`. |
| **`models/train.py`** *(modify)* | Append the 21 new feature names to `FEATURE_COLS`. Add the join_physical_ai_features call to the spine assembly. |
| **`tools/run_refresh.sh`** *(modify)* | Insert `robotics_signals_ingestion.py` between FRED energy step and BLS JOLTS step. Existing patent step picks up physical-AI mode automatically (same script invocation). |

Each new file has one responsibility:
- Ingestion modules: fetch + save raw data, no derived computation.
- Feature module: load raw + compute features + join to spine, no fetching.

## Feature columns added (21 total — `FEATURE_COLS` 88 → 109)

| # | Feature name | Source | Type |
|---:|---|---|---|
| 1 | `phys_ai_capgoods_orders_level` | FRED NEWORDER | Level (USD) |
| 2 | `phys_ai_capgoods_orders_yoy` | FRED NEWORDER | yoy % change |
| 3 | `phys_ai_pmi_level` | FRED NAPM | Level (0–100 index) |
| 4 | `phys_ai_machinery_prod_level` | FRED IPG3331S | Level (index) |
| 5 | `phys_ai_machinery_prod_yoy` | FRED IPG3331S | yoy % change |
| 6 | `phys_ai_machinery_ppi_level` | FRED WPU114 | Level (index) |
| 7 | `phys_ai_machinery_ppi_yoy` | FRED WPU114 | yoy % change |
| 8 | `phys_ai_machinery_jobs_level` | BLS NAICS 333 | Level (thousands) |
| 9 | `phys_ai_machinery_jobs_yoy` | BLS NAICS 333 | yoy % change |
| 10 | `phys_ai_patents_manipulators_count` | USPTO B25J | Quarterly count |
| 11 | `phys_ai_patents_manipulators_yoy` | USPTO B25J | yoy % change |
| 12 | `phys_ai_patents_aerial_count` | USPTO B64 | Quarterly count |
| 13 | `phys_ai_patents_aerial_yoy` | USPTO B64 | yoy % change |
| 14 | `phys_ai_patents_avs_count` | USPTO B60W | Quarterly count |
| 15 | `phys_ai_patents_avs_yoy` | USPTO B60W | yoy % change |
| 16 | `phys_ai_patents_motion_count` | USPTO G05D1 | Quarterly count |
| 17 | `phys_ai_patents_motion_yoy` | USPTO G05D1 | yoy % change |
| 18 | `phys_ai_patents_progcontrol_count` | USPTO G05B19 | Quarterly count |
| 19 | `phys_ai_patents_progcontrol_yoy` | USPTO G05B19 | yoy % change |
| 20 | `phys_ai_patents_vision_count` | USPTO G06V | Quarterly count |
| 21 | `phys_ai_patents_vision_yoy` | USPTO G06V | yoy % change |

**Notes on choices:**
- PMI gets level only — yoy of a 0–100 index is noise.
- All other macro/labor signals get level + yoy because absolute level (current activity) and rate of change (momentum) are independently informative.
- Patents get count + yoy because filing-count level represents industry size (NVDA cares about the slope; small-cap robotics players care about absolute volume of competitive activity).

## Join semantics

- All features are **as-of joined** to the ticker spine using the **publication date** (when the data became publicly known), not the period-end date.
  - FRED: use `realtime_start` field.
  - BLS JOLTS: existing module behaviour (period_end + 6-week pub lag).
  - USPTO: use `application_date` (publicly disclosed at filing).
- **Forward-fill within tolerance:**
  - FRED & JOLTS monthly → 60-day window
  - Patents quarterly → 120-day window
- Beyond tolerance → emit `null`. No imputation.
- All features apply uniformly across all 149 tickers — model decides per-ticker weight.

## Error handling & data quality

- **API failures (fail-soft per source):** log warning, write empty parquet with correct schema. Existing data persists; daily refresh retries next day.
- **Missing data values:** FRED uses `.` for missing — convert to `null` at ingestion. Polars/pytorch-forecasting handle nulls natively.
- **Schema drift:** each parquet path has a fixed schema dict at the top of its module; load-time assertion fails fast if a source returns unexpected columns.
- **Look-ahead leakage:** every join uses publication date, not period end.
- **Zero-baseline yoy:** if prior period's value is 0, yoy returns `null` (no division by zero).

## Test plan

### New: `tests/test_robotics_signals_ingestion.py`
- `test_fetch_fred_series_schema` — mocked FRED response, assert 4 series with correct schema.
- `test_fetch_fred_series_handles_missing_dot` — `.` → null conversion.
- `test_fetch_fred_series_failure_returns_empty` — mocked 500 → empty DataFrame, no exception.
- `test_save_robotics_signals_writes_parquet` — round-trip schema and snappy compression.

### Modified: `tests/test_bls_jolts_ingestion.py`
- Update existing tests: fetch BOTH NAICS 51 and NAICS 333 (count assertion 1 → 2 series).
- New: `test_naics_333_series_id_is_correct` — asserts `JTS333000000000000JOL` is in the fetched series list.

### Modified: `tests/test_uspto_ingestion.py`
- New: `test_physical_ai_mode_fetches_six_classes` — mocked PatentsView response, assert 6 CPC class buckets each get a parquet file.
- New: `test_physical_ai_quarterly_aggregation` — assert filings are bucketed by `quarter_end` (Q1=Mar 31, Q2=Jun 30, Q3=Sep 30, Q4=Dec 31).
- New: `test_physical_ai_b64_combines_b64c_and_b64u` — asserts B64 bucket sums both subclass filings.

### New: `tests/test_physical_ai_features.py`
- `test_join_macro_features_forward_fills_within_tolerance` — last FRED obs within 60d → propagates; beyond → null.
- `test_join_patent_features_quarterly_tolerance` — 120-day tolerance for patent counts.
- `test_yoy_handles_zero_baseline` — yoy with prior value = 0 → null.
- `test_phys_ai_features_apply_to_all_tickers` — every one of 149 tickers gets a value (or null) for each new feature column on a given test date.
- `test_publication_date_no_leakage` — feature value at date T uses only data publicly available by T.

### Modified: `tests/test_train.py`
- Update `FEATURE_COLS` count assertion: 88 → 109.
- Add the 21 new feature names to the membership assertion list.

### Acceptance gate
`pytest tests/ -m 'not integration'` — current baseline 435 passing. After this spec lands, expect ~445+ (10 new tests, none broken).

## Files touched

| File | Change |
|---|---|
| `ingestion/robotics_signals_ingestion.py` | NEW |
| `ingestion/bls_jolts_ingestion.py` | +NAICS 333 series fetch |
| `ingestion/uspto_ingestion.py` | +physical-AI CPC mode |
| `processing/physical_ai_features.py` | NEW |
| `models/train.py` | +21 feature names, +join_physical_ai_features call |
| `tools/run_refresh.sh` | +1 step (robotics_signals_ingestion before BLS JOLTS) |
| `tests/test_robotics_signals_ingestion.py` | NEW |
| `tests/test_bls_jolts_ingestion.py` | +NAICS 333 test, count update |
| `tests/test_uspto_ingestion.py` | +physical-AI mode tests |
| `tests/test_physical_ai_features.py` | NEW |
| `tests/test_train.py` | +21 feature names, count 88→109 |

## Files NOT touched (verified)

- `ingestion/ticker_registry.py` — no new tickers in this spec
- `models/inference.py` — picks up `FEATURE_COLS` dynamically from `train.py`
- `processing/feature_engineering.py` — `physical_ai_features` is its own module joined into the spine

## Rollout

Single PR (`feature/physical-ai-signals`), 5 atomic commits, each leaves the suite green:

1. **`feat: add robotics_signals_ingestion.py for 4 FRED series`** — new ingestion module + tests
2. **`feat: extend BLS JOLTS to NAICS 333 machinery manufacturing`** — modify ingestion + tests
3. **`feat: extend USPTO patents ingestion to 6 physical-AI CPC classes`** — modify ingestion + tests
4. **`feat: add physical_ai_features module, FEATURE_COLS 88→109`** — new feature module, train wiring, tests
5. **`chore: wire physical-AI signals into run_refresh.sh`** — refresh script + final acceptance gate

## Risks & out-of-scope

- **Backtest staleness**: `data/backtest/walk_forward_results.json` references the old 88-feature run. A fresh walk-forward run after merging is required but **out-of-scope** for this spec.
- **Patent CPC code drift**: USPTO occasionally retires/renames CPC classes. Mitigated by load-time schema assertions; manual update if a class is removed.
- **NAPM availability on FRED**: Spec assumes `NAPM` is reachable; fallback chain is `NAPM` → `USAPMI` → ship with 3 FRED series instead of 4 (FEATURE_COLS would be 108 instead of 109; tests adjust).
- **No model retrain inside this spec** — features land, but a fresh walk-forward run is the user's call, not part of this PR.
- **Out-of-scope for this spec** (deferred):
  - BLS CES employment by sub-industry (separate spec if needed)
  - USAJOBS keyword expansion for robotics terms (separate spec)
  - OEWS occupation-code data (annual cadence too slow)
  - Per-ticker assignee-filtered patent counts (deferred — current spec captures industry-wide signal)
