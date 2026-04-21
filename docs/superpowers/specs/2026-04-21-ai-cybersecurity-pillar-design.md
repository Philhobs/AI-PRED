# AI Cybersecurity Pillar Design

## Problem

The current 11-layer model covers the AI infrastructure supply chain from hyperscalers to
metals. Cybersecurity is a critical adjacent sector: AI spending directly drives cybersecurity
demand (larger attack surfaces, AI-powered threats), and cybersecurity companies are among
the primary beneficiaries of AI infrastructure growth. Neither the stock universe nor the
feature set currently captures this relationship.

## Goal

1. Add **Layer 12 (`cyber_pureplay`)** and **Layer 13 (`cyber_platform`)** to the ticker
   registry — 14 new tickers flowing through the existing pipeline unchanged.
2. Add **7 cybersecurity threat features** derived from free public sources (NVD, CISA,
   AlienVault OTX) that serve as market-wide regime signals for all tickers.
3. Route threat features into the `short` and `medium` tiers; exclude from `long`
   (noise at year+ horizons).
4. Design the threat ingestion layer as pluggable so paid sources can be added later
   without pipeline changes.

## Tickers

### Layer 12: `cyber_pureplay` — AI-native, product is the AI

| Ticker | Company | Exchange | Currency |
|---|---|---|---|
| CRWD | CrowdStrike | US | USD |
| ZS | Zscaler | US | USD |
| S | SentinelOne | US | USD |
| DARK.L | Darktrace | London | GBP |
| VRNS | Varonis | US | USD |

### Layer 13: `cyber_platform` — Enterprise platforms with AI security

| Ticker | Company | Exchange | Currency |
|---|---|---|---|
| PANW | Palo Alto Networks | US | USD |
| FTNT | Fortinet | US | USD |
| CHKP | Check Point | US (NASDAQ) | USD |
| CYBR | CyberArk | US (NASDAQ) | USD |
| TENB | Tenable | US | USD |
| QLYS | Qualys | US | USD |
| OKTA | Okta | US | USD |
| AKAM | Akamai | US | USD |
| RPD | Rapid7 | US | USD |

The split reflects fundamentally different trading drivers: pure-plays trade on ARR/NRR
growth and AI product differentiation; platforms trade on enterprise contract renewals,
government contracts, and competitive moat.

## Threat Data Ingestion

### Architecture: pluggable sources

```python
# ingestion/cyber_threat_ingestion.py

class CyberThreatSource(Protocol):
    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Return rows: (date, source, metric, value)."""
        ...

class NVDSource:      # NIST CVE API — no key required
class CISASource:     # CISA KEV feed — no key required
class OTXSource:      # AlienVault OTX — free key (OTX_API_KEY in .env, optional)
```

Adding a paid source (GreyNoise, Shodan) later = one new class implementing
`CyberThreatSource`. No ingestor changes required.

### Output

```
data/raw/cyber_threat/date=YYYY-MM-DD/threats.parquet
```

Schema: `date (Date), source (Utf8), metric (Utf8), value (Float64)`

### Source details

| Source | Endpoint | Key needed | Rate limit |
|---|---|---|---|
| NVD CVE API | `https://services.nvd.nist.gov/rest/json/cves/2.0` | None | 5 req/30s |
| CISA KEV | `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` | None | No limit |
| AlienVault OTX | `https://otx.alienvault.com/api/v1/pulses/subscribed` | Free (OTX_API_KEY) | 10k/day |

OTX gracefully degrades to zero-fill if `OTX_API_KEY` is absent in `.env`.

`time.sleep(1)` between NVD calls per project rate-limit rules.

## Cyber Threat Features

### New file: `processing/cyber_threat_features.py`

Reads raw threat parquet, produces rolling-window aggregates joined by **date**
(market-wide regime signal — not ticker-specific).

```python
CYBER_THREAT_FEATURE_COLS: list[str] = [
    "cve_critical_7d",
    "cve_high_7d",
    "cisa_kev_7d",
    "otx_pulse_7d",
    "cyber_threat_index_7d",
    "cve_critical_30d",
    "cisa_kev_30d",
]
```

| Feature | Description | Window |
|---|---|---|
| `cve_critical_7d` | CVSS ≥9.0 CVEs published | 7-day rolling sum |
| `cve_high_7d` | CVSS 7.0–8.9 CVEs published | 7-day rolling sum |
| `cisa_kev_7d` | CISA KEV entries added | 7-day rolling sum |
| `otx_pulse_7d` | AlienVault OTX threat pulses | 7-day rolling sum |
| `cyber_threat_index_7d` | Composite: `weighted = cve_critical_7d×3 + cve_high_7d + cisa_kev_7d×2`; divided by the 30-day rolling max of `weighted` (min denominator = 1), clamped to [0, 1] | 7-day |
| `cve_critical_30d` | CVSS ≥9.0 CVEs published | 30-day rolling sum |
| `cisa_kev_30d` | CISA KEV entries added | 30-day rolling sum |

### Join strategy

`join_cyber_threat_features(df: pl.DataFrame, features_dir: Path) -> pl.DataFrame`

Joins on `date` column (left join). Missing dates zero-fill. All tickers on a given date
receive the same threat feature values.

## Feature Tier Routing

`CYBER_THREAT_FEATURE_COLS` is added to `FEATURE_COLS` (the 48-feature master list grows
to 55 features).

| Tier | Change |
|---|---|
| `short` | Add `cve_critical_7d`, `cve_high_7d`, `cisa_kev_7d`, `otx_pulse_7d`, `cyber_threat_index_7d` |
| `medium` | Add all 7 `CYBER_THREAT_FEATURE_COLS` (medium = all FEATURE_COLS) |
| `long` | No change — threat features excluded (stale at year+ horizons) |

## File Changes

### `ingestion/ticker_registry.py`
- Add `cyber_pureplay` (Layer 12) and `cyber_platform` (Layer 13) tickers to `TICKERS_INFO`
- Add entries to `LAYER_IDS` and `LAYER_LABELS`
- No other changes — existing lookups auto-update from `TICKERS_INFO`

### `ingestion/cyber_threat_ingestion.py` (new)
- `CyberThreatSource` protocol
- `NVDSource`, `CISASource`, `OTXSource` implementations
- `ingest_cyber_threats(start_date, end_date, output_dir, sources)` — main entry point
- `__main__`: run last 30 days by default; accepts `--start` / `--end` flags

### `processing/cyber_threat_features.py` (new)
- `build_cyber_threat_features(threats_dir, date_str) -> pl.DataFrame`
- `join_cyber_threat_features(df, features_dir) -> pl.DataFrame`
- Exports `CYBER_THREAT_FEATURE_COLS`

### `models/train.py`
- Import `CYBER_THREAT_FEATURE_COLS` from `processing.cyber_threat_features`
- Append to `FEATURE_COLS`
- Update `TIER_FEATURE_COLS["short"]` and implicitly `["medium"]`

### `models/inference.py`
- Import `join_cyber_threat_features`
- Add call in `_build_feature_df` (after existing joins, before return)
- Gracefully zero-fills if `data/raw/cyber_threat/` doesn't exist

### `tests/`
- `tests/ingestion/test_cyber_threat_ingestion.py`
  - NVDSource, CISASource, OTXSource each return correct schema
  - OTX degrades gracefully when key absent
  - ingest writes parquet with correct partition
- `tests/processing/test_cyber_threat_features.py`
  - 7-day rolling sums correct given known input
  - `cyber_threat_index_7d` is in [0, 1]
  - join adds columns to df; missing dates zero-fill
  - empty threats dir returns zero-filled columns
- `tests/ingestion/test_ticker_registry.py`
  - `cyber_pureplay` and `cyber_platform` present in `LAYER_IDS`
  - All new tickers appear in `TICKERS`
  - `tickers_in_layer("cyber_pureplay")` returns 5 tickers
  - `tickers_in_layer("cyber_platform")` returns 9 tickers

## Architecture Constraints

- `DARK.L` follows the existing non-USD ticker pattern (`exchange="L"`, `currency="GBP"`)
  — FX conversion handled by `fx_features.py` already
- `CHKP` and `CYBR` are NASDAQ-listed Israeli companies — `country="IL"`, `exchange="US"`
- Threat features are market-wide (date-keyed) not ticker-specific. Ticker-specific breach
  data is a future paid-source extension via the pluggable architecture.
- `FEATURE_COLS` grows from 48 → 55. Existing 252d flat-layout artifacts remain valid
  (they were trained on 48 features; `feature_names.json` governs what each model uses).

## What This Does NOT Cover

- Ticker-specific breach/incident data (requires paid feeds — GreyNoise, SecurityScorecard)
- Private AI security companies (Vectra AI, Abnormal Security, Deep Instinct)
- Options-derived signals for cyber tickers (Sub-project 3)
- Scheduled daily threat ingestion via APScheduler (Phase 2)

## Success Criteria

1. `pytest tests/ -m 'not integration'` passes with no regressions
2. `tickers_in_layer("cyber_pureplay")` returns `["CRWD", "DARK.L", "S", "VRNS", "ZS"]`
3. `tickers_in_layer("cyber_platform")` returns `["AKAM", "CHKP", "CYBR", "FTNT", "OKTA", "PANW", "QLYS", "RPD", "TENB"]`
4. `build_cyber_threat_features(threats_dir, date_str)` returns a DataFrame with all 7 columns
5. `FEATURE_COLS` contains all 7 `CYBER_THREAT_FEATURE_COLS`
6. `cyber_threat_index_7d` values are in [0, 1]
