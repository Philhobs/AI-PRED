# AI Infrastructure Supply Chain Expansion — Design Spec

**Goal:** Expand the prediction model from 24 tickers to 83 tickers across 10 supply chain layers, add a knowledge graph of confirmed deals and partnerships, and introduce per-layer ensemble models with a unified global ranking.

**Architecture:** Ten per-layer ensembles (LightGBM + RF + Ridge), each trained only on its layer's tickers. Global ranking by raw predicted annual return. Three graph features added to every ticker's feature vector, derived from a NetworkX deal graph built from SEC 8-K filings + manual curation.

**Tech Stack:** Python 3.11, Polars, DuckDB, NetworkX, LightGBM, scikit-learn, yfinance, SEC EDGAR EFTS API, pdfplumber, existing project stack.

---

## 1. Supply Chain Taxonomy

Ten layers. Each ticker included only where confirmed deal, partnership, or direct revenue dependency exists with the AI infrastructure build-out.

### Layer 1 — Hyperscalers / Cloud
`MSFT AMZN GOOGL META ORCL IBM`

Demand root of the entire supply chain. ORCL: $10B GPU cluster deal with NVDA. IBM: watsonx + hybrid cloud partnerships.

### Layer 2 — AI Compute / Chips
`NVDA AMD AVGO MRVL TSM ASML INTC ARM MU SNPS CDNS`

MU: HBM3E memory shipped inside NVDA H100/H200. SNPS + CDNS: EDA software used to tape out every chip at TSMC — direct design-time dependency.

### Layer 3 — Semiconductor Equipment & Materials
`AMAT LRCX KLAC ENTG MKSI UCTT ICHR TER ONTO APD LIN`

ENTG: ultra-pure materials to TSMC and Samsung. APD + LIN: specialty gases (NF3, SiH4) at every wafer fab. ICHR: fluid delivery systems inside AMAT/LRCX tools.

### Layer 4 — Networking / Interconnect
`ANET CSCO CIEN COHR LITE INFN NOK VIAV`

ANET: ~40% revenue from Meta + Microsoft Ethernet switches. COHR: optical transceivers inside every 400G/800G data center link.

### Layer 5 — Servers / Storage / Systems
`SMCI DELL HPE NTAP PSTG STX WDC`

NTAP: "ONTAP AI" certified with NVDA DGX. PSTG: NVIDIA-Certified storage, signed joint go-to-market with NVDA.

### Layer 6 — Data Center Operators / REITs
`EQIX DLR AMT CCI IREN APLD`

IREN + APLD: pure-play AI GPU-as-a-service operators — direct dependency on NVDA GPU supply and hyperscaler demand.

### Layer 7 — Power / Energy / Nuclear
`CEG VST NRG TLN NEE SO EXC ETR GEV BWX OKLO SMR FSLR`

CEG: Three Mile Island restart, 20-year Microsoft PPA. TLN: nuclear deal with Amazon. OKLO: micro-reactor data center power deals. GEV: gas turbines and grid equipment for data center power interconnects. BWX: SMR component manufacturing.

### Layer 8 — Cooling / Facilities / Backup Power
`VRT NVENT JCI TT CARR GNRC HUBB`

VRT: existing model ticker. GNRC: ~35% of US hyperscale data center backup generators. HUBB: electrical distribution equipment in every large DC build.

### Layer 9 — Grid / Construction / Electrical Contracting
`PWR MTZ EME MYR IESC AGX`

PWR (Quanta Services): largest US power grid contractor, building transmission infrastructure for AI data center power interconnects — signed contracts with utilities serving AI campuses. MTZ + EME: electrical infrastructure for DC campuses.

### Layer 10 — Metals / Materials
`FCX SCCO AA NUE STLD MP UUUU ECL`

FCX: copper for busbars + liquid cooling loops (~50kg copper per H100 rack). MP Materials: rare-earth magnets in every server fan and power supply motor. ECL: water treatment chemistry for hyperscale cooling towers.

**Total: 83 tickers across 10 layers** (up from 24 currently in the model).

---

## 2. Architecture

### Storage

All data follows existing Parquet/DuckDB pattern (snappy compression). Two new stores:

```
data/raw/
  graph/
    deals.parquet          ← all confirmed deals (automated + manual merged)
    edges.parquet          ← company→company relationship graph
    ticker_layers.parquet  ← {ticker, layer_name, layer_id} lookup

data/manual/
  deals_override.csv       ← editorial curation file (user edits directly)
```

### Per-Layer Model Artifacts

```
models/artifacts/
  layer_01_cloud/
    lgbm_q10.pkl  lgbm_q50.pkl  lgbm_q90.pkl
    rf_model.pkl  ridge_model.pkl  feature_scaler.pkl
    imputation_medians.json  feature_names.json  ensemble_weights.json
  layer_02_compute/
  ...
  layer_10_metals/
```

### Global Ranking

`models/inference.py` runs all 10 layer models, collects predicted returns for all 83 tickers, sorts globally by `expected_annual_return` descending. Output schema unchanged — one unified ranked list per day.

---

## 3. Deal Ingestion Pipeline

### 3a. Automated — SEC 8-K Filings (`ingestion/deal_ingestion.py`)

- Polls EDGAR EFTS full-text search for 8-K Item 1.01 (material definitive agreements) for all 130 tickers.
- Extracts counterparty company names from filing text using regex + fuzzy match against ticker watchlist.
- Maps counterparty names → tickers using a name→ticker lookup table.
- Assigns `deal_type` from keyword matching (see taxonomy below).
- Sets `confidence = 0.7` for 8-K extracted deals.
- Runs nightly via existing scheduler pattern.

### 3b. Manual Curation (`data/manual/deals_override.csv`)

CSV format — user edits directly, no code changes required:

```csv
date,party_a,party_b,deal_type,description,source_url
2023-09-25,MSFT,CEG,power_purchase_agreement,"20yr 835MW nuclear PPA TMI restart",https://...
2024-03-18,AMZN,TLN,power_purchase_agreement,"Talen nuclear deal Susquehanna",https://...
2024-05-20,NVDA,TSM,supply_agreement,"CoWoS advanced packaging capacity reservation",https://...
```

Sets `confidence = 1.0`. Manual entries always override automated on `(party_a, party_b, date)` conflicts.

### Deal Type Taxonomy

```
power_purchase_agreement    ← utility signs with cloud/DC operator
supply_agreement            ← component/material supplier with OEM
manufacturing_agreement     ← fab capacity reservation (TSM/INTC)
joint_venture               ← co-investment in facility/project
customer_contract           ← large revenue commitment (NVDA GPUs to CSP)
investment                  ← equity stake or financing
licensing_agreement         ← IP/technology licensing
construction_contract       ← PWR/MTZ building for utility/DC operator
```

---

## 4. Knowledge Graph Data Model

### `deals.parquet` Schema

| Column | Type | Notes |
|--------|------|-------|
| `deal_id` | str | `{party_a}-{party_b}-{date}` |
| `date` | date | Filing or announcement date |
| `party_a` | str | Ticker |
| `party_b` | str | Ticker |
| `deal_type` | str | From taxonomy above |
| `description` | str | Human-readable summary |
| `value_usd` | float | Often null (undisclosed) |
| `duration_years` | float | Null if not specified |
| `source` | str | `8-K` / `manual` / `news` |
| `source_url` | str | EDGAR URL or news URL |
| `layer_a` | str | Layer name of party_a |
| `layer_b` | str | Layer name of party_b |
| `confidence` | float | 1.0=manual, 0.7=8-K, 0.5=news |

### `edges.parquet` Schema

Bidirectional — one row per unique ticker pair with at least one deal.

| Column | Type | Notes |
|--------|------|-------|
| `ticker_from` | str | |
| `ticker_to` | str | |
| `edge_weight` | float | Sum of confidence scores, decayed 50%/year |
| `deal_count` | int | Distinct deals between pair |
| `last_deal_date` | date | Most recent deal |
| `edge_types` | str | Pipe-separated list of deal types |

**Edge weight decay:** `weight = confidence × 0.5^(years_since_deal)`. Old deals matter less; a 3-year-old PPA contributes 0.125× a fresh one.

### Graph Construction

Built in memory using NetworkX at training/inference time from `edges.parquet`. Nodes = tickers, edges = weighted by `edge_weight`. Recomputed on each run — graph is small (~130 nodes, ~300–500 edges).

---

## 5. Graph Features

Three new features added to every ticker's 34-feature vector (up from 31):

### `graph_partner_momentum_30d`
Weighted average of direct deal partners' 30-day price returns, weighted by `edge_weight`. Partners' momentum leads own momentum — copper supplier surging often precedes your own orders accelerating.

```
graph_partner_momentum_30d(ticker) =
    Σ(edge_weight(ticker→p) × return_30d(p)) / Σ(edge_weight(ticker→p))
    for all direct neighbors p in graph
```

Returns `null` for tickers with no confirmed deal partners yet.

### `graph_deal_count_90d`
Count of new deals filed by direct partners in the past 90 days. A burst of partner deal activity is a leading indicator of revenue growth upstream or downstream.

### `graph_hops_to_hyperscaler`
Encoded as `1 / shortest_path_length` to nearest {MSFT, AMZN, GOOGL, META}.
- Hyperscalers themselves: `1.0`
- Direct partners (CEG, NVDA, ANET): `0.5`
- Partners' suppliers (FCX supplying VRT supplying MSFT): `0.33`
- Returns `0.0` if no path found (disconnected from hyperscalers)

---

## 6. Extended Data Pipelines

All existing ingestors extended to cover all 130 tickers with no architecture changes — just a larger `TICKER_LIST`:

| Pipeline | Change |
|----------|--------|
| `ohlcv_ingestion.py` | 130 tickers instead of 24 |
| `fundamental_ingestion.py` | 130 tickers |
| `earnings_ingestion.py` | 130 tickers |
| `short_interest_ingestion.py` | 130 tickers |
| `insider_trading_ingestion.py` | Add SEC CIK entries for new tickers |
| `news_ingestion.py` | Expand alias dictionary for new company names |
| `deal_ingestion.py` | **New** — 8-K automated deal discovery |

Congressional trades ingestion unchanged — House PTR data covers all US-listed equities already.

---

## 7. Training Changes

`models/train.py` updated to:
1. Load `ticker_layers.parquet` to get layer assignments.
2. Build graph features from `edges.parquet` + OHLCV prices.
3. For each layer (1–10): filter dataset to layer tickers, fit ensemble, save artifacts to `models/artifacts/layer_NN_name/`.
4. Save a global `ticker_layers.parquet` artifact for inference routing.

`models/inference.py` updated to:
1. Run all 10 layer models.
2. Collect predictions into single DataFrame.
3. Sort globally by `expected_annual_return` descending.
4. Write unified output — schema unchanged.

---

## 8. Testing Strategy

- **Unit tests per new module:** `test_deal_ingestion.py`, `test_graph_features.py`, `test_per_layer_train.py`
- **Graph tests:** synthetic 5-node graph, verify partner momentum and hop distance calculations.
- **Deal parsing tests:** synthetic 8-K text fixture, verify counterparty extraction and deal type assignment.
- **Per-layer model tests:** synthetic labeled data per layer, verify each layer trains and predicts without error.
- **Integration guard:** verify final inference output has ~130 rows, all tickers present, no nulls in rank column.

---

## 9. Files Created / Modified

**New files:**
```
ingestion/deal_ingestion.py
processing/graph_features.py
data/manual/deals_override.csv        ← seeded with known deals
data/raw/graph/ticker_layers.parquet  ← generated at first run
```

**Modified files:**
```
ingestion/insider_trading_ingestion.py   ← expand CIK_MAP to 130 tickers
ingestion/news_ingestion.py              ← expand alias dict
models/train.py                          ← per-layer training loop
models/inference.py                      ← multi-layer inference + global rank
models/backtest.py                       ← layer-aware backtest
```

**Test files:**
```
tests/test_deal_ingestion.py
tests/test_graph_features.py
tests/test_per_layer_models.py
```
