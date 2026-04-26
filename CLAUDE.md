# AI Infrastructure Predictor — Claude Code Instructions

## Context

Financial prediction tool for the **full AI infrastructure supply chain** (compute,
power, cooling, networking, storage, datacenter REITs, robotics, cyber, enterprise
SaaS) plus government/behavioural data signals. Predicts directional returns at
8 horizons (5d through 5040d ≈ 20y).

## Current state (as of 2026-04-26)

- **165 tickers** across **16 layers** (registry: `ingestion/ticker_registry.py`)
- **112 features** in `models/train.py::FEATURE_COLS` — engineered from 18 feature modules
- **517 tests passing** (`pytest -m 'not integration'`)
- **25-step pipeline** via `tools/run_refresh.sh` (best-effort: per-step failures logged, never abort)
- **Per-layer ensemble**: LGBM + RF + Ridge with NNLS-learned weights, one model per (layer, horizon) → 16×8 = up to 128 model triples

## Architecture

- **Language**: Python 3.11+
- **Storage**: Parquet (snappy) under `data/raw/`. Hive-partitioned where appropriate (e.g. `date=YYYY-MM-DD/`)
- **Query layer**: DuckDB (no server) for all analytical reads
- **NLP**: FinBERT (ProsusAI/finbert) for sentiment scoring
- **Models**: LightGBM + RandomForest + Ridge ensemble (NNLS weights from 3-fold walk-forward CV)
- **Single source of truth**: `ingestion/ticker_registry.py` — TICKERS_INFO drives layer assignment, currency, country, exchange. All consumers read it generically.

## Layer pillars (16)

```
1  cloud                       MSFT, AMZN, GOOGL, META, ORCL, IBM, SAP.DE, CAP.PA, OVH.PA
2  compute                     NVDA, AMD, AVGO, MRVL, TSM, ASML, MU, SK Hynix, Samsung, ...
3  semi_equipment              AMAT, LRCX, KLAC, ENTG, MKSI, ASMI.AS, BESI.AS, ...
4  networking                  ANET, CSCO, CIEN, COHR, LITE, ERIC, JNPR, ...
5  servers                     SMCI, DELL, HPE, NTAP, PSTG, STX, WDC, ...
6  datacenter                  EQIX, DLR, AMT, CCI, IREN, APLD, 9432.T, CLNX.MC
7  power                       CEG, VST, NEE, GEV, OKLO, SMR, CCJ, ENR.DE, ...
8  cooling                     VRT, ETN, NVENT, JCI, TT, CARR, GNRC, HUBB, SU.PA, ...
9  grid                        PWR, MTZ, EME, MYR, PRY.MI, NG.L, TRN.MI, ...
10 metals                      FCX, SCCO, AA, NUE, STLD, MP, UUUU, ECL, RIO.L, ...
11 robotics_industrial         ROK, ZBRA, CGNX, SYM, EMR, FANUC (6954.T), ABBN.SW, ...
12 robotics_medical_humanoid   ISRG, TSLA, 1683.HK (UBTECH), 005380.KS (Hyundai/BD)
13 robotics_mcu_chips          TXN, MCHP, ADI, 6723.T (Renesas)
14 cyber_pureplay              CRWD, ZS, S, DARK.L, VRNS, NET (Cloudflare)
15 cyber_platform              PANW, FTNT, CHKP, CYBR, TENB, QLYS, OKTA, AKAM, RPD
16 enterprise_saas             PLTR, NOW, CRM, ADBE, INTU, DDOG, SNOW, GTLB, TEAM, PATH, MNDY
```

## Feature module map

Each module in `processing/` exposes a `*_FEATURE_COLS` constant + a `join_*_features()` function. `models/train.py` imports them and aggregates into `FEATURE_COLS`. `models/inference.py` mirrors the join sequence. **Both must stay in sync** — enforced by `tests/test_train.py::test_train_and_inference_import_same_join_functions`.

| Module | Cols |
|---|---:|
| price_features | 6 |
| fundamental_features | 14 |
| insider_features | 5 |
| sentiment_features | 4 |
| short_interest_features | 4 |
| earnings_features | 4 |
| graph_features | 4 |
| ownership_features | 4 |
| energy_geo_features | 4 |
| supply_chain_features | 4 |
| fx_features | 1 |
| cyber_threat_features | 7 |
| options_features | 6 |
| gov_behavioral_features | 6 |
| patent_features | 6 |
| labor_features | 4 |
| census_trade_features | 6 |
| physical_ai_features | 21 |
| ai_economics_features | 3 |
| **Total** | **112** |

## Key rules (operational)

1. **Never write files to project root** — use `data/`, `ingestion/`, `processing/`, `models/`, `tools/`, `tests/`
2. Each pipeline module is **self-contained** — check imports before adding new deps
3. Always write Parquet with **snappy compression**
4. Use **DuckDB** for analytical queries — never load full datasets into Python memory
5. Free APIs: `time.sleep(1)` between calls; SEC: `time.sleep(0.1)` (10 req/s limit)
6. **`.env` must never be committed** — use `.env.example` only. Required keys: `FRED_API_KEY`, `CENSUS_API_KEY`. Optional: `BLS_API_KEY`. Pending (not yet acquired): `SAM_GOV_API_KEY`, `USAJOBS_USER_AGENT`.
7. Vectorised operations only (Polars or DuckDB) — never iterrows/apply on large data
8. **Always run tests** before marking work complete: `pytest tests/ -m 'not integration' -q`
9. **Fail-soft pattern** for ingestion: try/except → log warning → return empty parquet (never raise from ingestion)
10. **Train/inference parity**: every join function added to train.py must be mirrored in inference.py (regression-tested)

## Pipeline orchestration

`bash tools/run_refresh.sh` runs all 25 ingestion + processing steps. Best-effort: each step's failure is logged but the script continues. At the end, `tools/pipeline_health.py` reports per-source freshness with cadence-aware thresholds (daily=3d, weekly=10d, monthly=60d, quarterly=999d).

## Tooling

- **`tools/run_refresh.sh`** — full daily refresh (25 steps, best-effort)
- **`tools/pipeline_health.py`** — per-source freshness check, exit code 1 if anything stale/missing
- **`tools/score_predictions.py`** — IC + hit-rate + top-decile-return scoring of saved predictions vs realized OHLCV. PENDING when future hasn't elapsed.
- **`models/train.py [--horizon Ntag]`** — train per-layer ensembles (default: all 8 horizons)
- **`models/inference.py [--horizon Ntag]`** — predict for a date, write to `data/predictions/date=*/horizon=*/predictions.parquet`

## Workflow conventions

- **Branches**: `feature/<name>` for substantive work; small fixes can land on master directly with descriptive commits
- **Commits**: include a "why" sentence and a "what changed" summary; reference relevant agent flags
- **Specs + plans**: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` + `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
- **Planning seeds** (for deferred work): `docs/superpowers/specs/_planning_seed_<topic>.md` with explicit triggers
- **Backtest artifacts** (`data/backtest/`, `models/artifacts/`) are gitignored — regenerated per training run

## Outstanding planning seeds (8)

Triggered next session via `/brainstorm`:

| Seed | Trigger phrase | Effort |
|---|---|---|
| `_biosecurity_pillar` | "lets do biosecurity" | 4-6h |
| `_ferc_resurrection` | "fix FERC" | 30min - 2h |
| `_model_benchmark_gap` | "lets do model benchmarks" | 2-3h |
| `_inference_pricing` | "track cost per token" | 2-3h (do AFTER model_benchmark_gap) |
| `_dc_reit_prelease` | "track REIT occupancy" | 4-6h |
| `_regulatory_feed` | "EU AI Act tracking" | 3-4h |

## Recorded follow-ups (deferred minor work)

- Vectorize `join_physical_ai_features` via Polars `join_asof` (perf, currently row-by-row Python)
- Vectorize `_aggregate_physical_ai` via Polars expressions (perf, currently `map_elements`)
- USPTO pagination total-count termination (defense-in-depth)
- Consolidate FRED helper between `robotics_signals_ingestion` and `financial_ingestion`
- Per-layer trainer behavior with n=4 ticker cohorts (small layers like robotics_medical_humanoid)
