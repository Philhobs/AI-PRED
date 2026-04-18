# Phase C — Energy Signals Design

**Goal:** Add 4 new features to the model (39 → 43 `FEATURE_COLS`) capturing US grid tightness, international energy capacity tailwinds, and energy deal quality — turning the AI infrastructure buildout's power constraint into a quantifiable edge.

**Architecture:** Standard `ingestion/ → processing/ → FEATURE_COLS` pipeline. One new ingestion module (`eia_ingestion.py`), one new processing module (`energy_geo_features.py`), two targeted extensions to existing modules (`deal_ingestion.py`, `graph_features.py`), and one manual mapping file for per-ticker geographic exposure.

**Tech Stack:** Python 3.11, Polars, requests, openpyxl (PJM Excel), EIA open API (free key), yfinance already a dep.

---

## Context: What Already Exists

- `ingestion/energy_geo_ingestion.py` — pulls country-level energy stats (nuclear %, renewables %, electricity demand, carbon intensity) from Our World in Data for 14 AI-infra countries. Data collected but **not yet in FEATURE_COLS**.
- `ingestion/deal_ingestion.py` — ingests SEC 8-K Item 1.01 filings and classifies deal type (PPA, supply_agreement, etc.). Already extracts counterparty names. **Missing: MW capacity and buyer type (hyperscaler vs other).**
- `processing/graph_features.py` — produces `graph_deal_count_90d`, `graph_partner_momentum_30d`, `graph_hops_to_hyperscaler`. **Missing: energy-deal-specific features.**

Phase C fills these gaps.

---

## New Features Added (39 → 43)

| Feature | Source | Target layer |
|---|---|---|
| `us_power_moat_score` | EIA capacity + PJM queue | Power (CEG, VST, NRG, TLN, NEE, etc.) |
| `geo_weighted_tailwind_score` | OWID geo data + ticker exposure weights | Datacenter REITs + hyperscalers |
| `energy_deal_mw_90d` | Extended deal_ingestion.py | All (power + datacenter layers most signal) |
| `hyperscaler_ppa_count_90d` | Extended deal_ingestion.py | Power layer sellers |

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `ingestion/eia_ingestion.py` | Create | Fetch EIA monthly capacity + PJM weekly queue |
| `processing/energy_geo_features.py` | Create | Convert grid/geo data into per-ticker features |
| `data/manual/ticker_geo_exposure.csv` | Create | Per-ticker region weights (heuristic defaults + overrides) |
| `ingestion/deal_ingestion.py` | Extend | Add `deal_mw` + `buyer_type` columns to deal schema |
| `processing/graph_features.py` | Extend | Add `energy_deal_mw_90d` + `hyperscaler_ppa_count_90d` |
| `models/train.py` | Extend | Add 4 new features to `FEATURE_COLS` |
| `tests/test_eia_ingestion.py` | Create | Schema + filter tests |
| `tests/test_energy_geo_features.py` | Create | Feature logic + edge case tests |
| `tests/test_deal_enrichment.py` | Create | MW regex + buyer type classification tests |

---

## 1. EIA + PJM Ingestion (`ingestion/eia_ingestion.py`)

### 1a. EIA National Capacity (monthly)

**Endpoint:** `https://api.eia.gov/v2/electricity/electric-power-operational-data/data/`

**Auth:** Free API key — register at `api.eia.gov`. Store as `EIA_API_KEY` in `.env`.

**Parameters:**
```
frequency=monthly
data[]=capacity
facets[fueltypeid][]=NG   (natural gas)
facets[fueltypeid][]=NUC  (nuclear)
facets[fueltypeid][]=SUN  (solar)
facets[fueltypeid][]=WND  (wind)
facets[sectorid][]=99     (all sectors)
```

**Output schema:** `data/raw/energy/eia_capacity.parquet`
```
date          date     (first of month)
fuel_type     str      (natural_gas | nuclear | solar | wind)
capacity_gw   float64  (installed capacity in GW)
```

**Cadence:** Monthly. Fetch last 36 months on first run, then incremental.

### 1b. PJM Virginia Queue (weekly)

**Source:** PJM publishes a public Excel file — no auth required.
**URL:** `https://www.pjm.com/-/media/planning/rtep/rtep-documents/active-interconnection-requests-report.ashx`

**Processing:**
- Load Excel, filter to zones: `MAAC`, `AECO`, `SWVA` (Northern Virginia data center corridor)
- Keep rows where `Status != 'Withdrawn'`
- Aggregate: `queue_backlog_gw = sum(MW Requested) / 1000`

**Output schema:** `data/raw/energy/pjm_queue.parquet`
```
date              date     (download date)
zone              str      (MAAC | AECO | SWVA | ALL_VIRGINIA)
queue_backlog_gw  float64
project_count     int32
```

**Cadence:** Weekly (PJM updates this file weekly).

**Failure handling:** If EIA key missing or PJM URL unreachable → log warning, write nothing, pipeline continues. Same pattern as `ais_ingestion.py`.

---

## 2. Energy Geo Features (`processing/energy_geo_features.py`)

Produces 2 features on the daily training spine using `join_asof` with `tolerance=timedelta(days=45)` (monthly data).

### 2a. `us_power_moat_score`

**Formula:**
```python
raw = pjm_virginia_queue_backlog_gw / (nuclear_capacity_gw + gas_capacity_gw)
us_power_moat_score = (raw - rolling_3yr_min) / (rolling_3yr_max - rolling_3yr_min)
# Clipped to [0, 1]
```

**Interpretation:** High = demand far exceeds baseload supply = strong moat for existing generators.

**Per-ticker assignment:**
- Power layer tickers (CEG, VST, NRG, TLN, NEE, SO, EXC, ETR, GEV, BWX, OKLO, SMR, FSLR): full signal value
- All other tickers: `0.0`

### 2b. `geo_weighted_tailwind_score`

**Formula:**
```python
for each region in ticker_geo_exposure[ticker]:
    region_tailwind = 0.6 * renewable_growth_yoy[region] + 0.4 * (1 - carbon_intensity_norm[region])
geo_weighted_tailwind_score = sum(weight[region] * region_tailwind[region])
```

**Geographic exposure** from `data/manual/ticker_geo_exposure.csv`:

```csv
ticker,region,weight
EQIX,north_america,0.55
EQIX,emea,0.12
EQIX,nordics,0.18
EQIX,asia_pacific,0.15
DLR,north_america,0.60
DLR,emea,0.25
DLR,asia_pacific,0.15
AMT,north_america,0.70
AMT,emea,0.20
AMT,asia_pacific,0.10
APLD,north_america,1.00
IREN,north_america,1.00
MSFT,north_america,0.55
MSFT,emea,0.25
MSFT,asia_pacific,0.20
AMZN,north_america,0.60
AMZN,emea,0.20
AMZN,asia_pacific,0.20
GOOGL,north_america,0.55
GOOGL,emea,0.25
GOOGL,asia_pacific,0.20
META,north_america,0.65
META,emea,0.20
META,nordics,0.15
```

**Default for tickers not in CSV:** `north_america = 1.0` (neutral — no region penalty).

**Region → OWID country mapping:**
```
north_america → United States, Canada
emea          → Germany, United Kingdom, France, Netherlands
nordics       → Norway, Sweden, Iceland
asia_pacific  → Japan, South Korea, Singapore, Malaysia
```

---

## 3. Deal Enrichment

### 3a. Extend `ingestion/deal_ingestion.py`

Add two columns to the existing deal parsing logic (inside `_parse_deal_text()`):

**`deal_mw` (float64, nullable):**
```python
_MW_PATTERNS = [
    r"(\d[\d,]*)\s*(?:MW|megawatt)",
    r"(\d[\d,]*)\s*(?:GW|gigawatt)",   # multiply by 1000
    r"(\d[\d,]*)-megawatt",
]
# Returns None if no match found (~40% of PPAs don't name capacity)
```

**`buyer_type` (str):**
```python
_HYPERSCALERS = {"microsoft", "amazon", "google", "alphabet", "meta", "apple"}
_CRYPTO_MINERS = {"iren", "applied digital", "apld", "marathon", "riot", "core scientific"}
_UTILITIES     = {"duke energy", "dominion", "southern company", "exelon", "entergy", "nextera"}

# Map counterparty_name.lower() → hyperscaler | crypto_miner | utility | other
```

Updated `deals.parquet` schema adds these two nullable columns. Existing columns unchanged.

### 3b. Extend `processing/graph_features.py`

Add 2 features to the per-ticker rolling window calculation (alongside existing `graph_deal_count_90d`):

**`energy_deal_mw_90d`:**
```python
# Sum of deal_mw for all deals involving this ticker in last 90 days
# Null deal_mw treated as 0 (don't penalize deals that omit capacity)
```

**`hyperscaler_ppa_count_90d`:**
```python
# Count of deals where deal_type == 'power_purchase_agreement'
#   AND buyer_type == 'hyperscaler'
#   AND ticker is the SELLER (power company)
# For non-seller tickers: 0
```

---

## 4. Model Integration

In `models/train.py`, extend `FEATURE_COLS`:

```python
# Energy signals (Phase C) — 4 new features
"us_power_moat_score",
"geo_weighted_tailwind_score",
"energy_deal_mw_90d",
"hyperscaler_ppa_count_90d",
```

These are added to `build_training_dataset()` via the same `join_asof` pattern as other slow-moving features.

---

## 5. Testing

### `tests/test_eia_ingestion.py`
- `test_eia_capacity_schema` — mock HTTP response → assert parquet columns `[date, fuel_type, capacity_gw]`
- `test_pjm_queue_schema` — mock Excel file → assert columns `[date, zone, queue_backlog_gw, project_count]`
- `test_pjm_filters_virginia_zones_only` — mixed zones → only MAAC/AECO/SWVA kept

### `tests/test_energy_geo_features.py`
- `test_us_power_moat_score_range` — synthetic data → score in [0, 1]
- `test_power_moat_zero_for_non_power_layer` — NVDA → `us_power_moat_score == 0.0`
- `test_geo_tailwind_uses_exposure_weights` — EQIX with 50% nordics / 50% NA → correct weighted average
- `test_missing_ticker_defaults_to_north_america` — unknown ticker → `north_america = 1.0`

### `tests/test_deal_enrichment.py`
- `test_mw_extraction_standard_format` — "500 MW" → `deal_mw == 500.0`
- `test_mw_extraction_with_commas` — "1,200 megawatts" → `deal_mw == 1200.0`
- `test_mw_extraction_returns_null_when_absent` — no capacity mention → `deal_mw is None`
- `test_buyer_type_hyperscaler` — "Microsoft Corporation" → `buyer_type == "hyperscaler"`
- `test_buyer_type_other` — unknown counterparty → `buyer_type == "other"`
- `test_hyperscaler_ppa_count_feature` — synthetic deals parquet → correct count per ticker per 90d window

---

## 6. Environment

Add to `.env.example`:
```
EIA_API_KEY=your_key_here   # Register free at https://www.eia.gov/opendata/
```

---

## 7. Cadence Summary

| Data source | Update frequency | Stale threshold (pipeline_health.py) |
|---|---|---|
| EIA capacity | Monthly | 45 days |
| PJM queue | Weekly | 10 days |
| Energy geo (OWID) | Annual | 400 days |
| Deal enrichment | On each 8-K ingest | 7 days |
