# Options Signals Design

## Problem

The model has 55 features across 11 groups but no options-derived signals. Options markets price in forward-looking uncertainty and directional hedging demand — information not captured by any existing feature. For cybersecurity stocks in particular, IV spikes and bearish skew often precede or accompany breach-related sell-offs. Options data is the primary missing forward-looking signal.

## Goal

1. Add **6 ticker-specific options features** (`OPTIONS_FEATURE_COLS`) derived from the live options chain via `yfinance`.
2. Wire a **pluggable `OptionsSource` protocol** (same pattern as `CyberThreatSource`) so paid sources (Tradier, CBOE) can be added later without pipeline changes.
3. Route options features into `short` and `medium` tiers; exclude from `long` (noise at year+ horizons).
4. Apply to **all 141 tickers** in the registry. Tickers with no liquid options (e.g. DARK.L) zero-fill gracefully.

## Features

`OPTIONS_FEATURE_COLS: list[str]` — 6 features, all ticker-specific, joined by `(ticker, date)`:

| Feature | Description | Source |
|---|---|---|
| `iv_rank_30d` | ATM IV percentile vs. 52-week high/low, scaled 0–100. High = elevated fear premium. | Options chain + rolling history |
| `iv_hv_spread` | Near-term ATM IV minus 30-day realized historical vol (HV30). Positive = vol premium. | Options chain + OHLCV |
| `put_call_oi_ratio` | Put OI / call OI for near-term expiry (≤45 days). High = institutional hedging. | Options chain |
| `put_call_vol_ratio` | Put volume / call volume for near-term expiry. More reactive than OI — reflects today's flow. | Options chain |
| `skew_otm` | OTM put IV minus OTM call IV at ~5–10% moneyness. Positive = bearish skew. | Options chain |
| `iv_term_slope` | 30-day ATM IV minus 90-day ATM IV. Positive = inverted term structure = near-term fear spike. | Options chain |

**FEATURE_COLS grows from 55 → 61.**

### IV Rank Computation

Requires 252 daily ATM IV snapshots per ticker. Processing module reads all historical `date=*/` parquets, computes:

```
iv_rank = (today_iv - min_52w) / (max_52w - min_52w) × 100
```

Clamped to [0, 100]. Falls back to 50 (neutral) if fewer than 30 days of history exist.

### HV30 Computation

Computed from OHLCV parquets already at `data/raw/financials/ohlcv/`. 30-day rolling std of log returns × √252. No new ingestion required.

### ATM Strike Selection

For each expiry: find the strike closest to the current spot price. OTM put = nearest strike ~5–10% below spot. OTM call = nearest strike ~5–10% above spot.

## Feature Tier Routing

| Tier | Change |
|---|---|
| `short` | Add all 6 `OPTIONS_FEATURE_COLS` |
| `medium` | Add all 6 (via `FEATURE_COLS`) |
| `long` | No change — options signals excluded (noise at year+ horizons) |

## Ingestion Architecture

### Pluggable Source Protocol

```python
# ingestion/options_ingestion.py

class OptionsSource(Protocol):
    def fetch(self, ticker: str, date: str) -> pl.DataFrame:
        """Return raw options contracts: (ticker, date, expiry, option_type, strike, iv, oi, volume)"""
        ...

class YFinanceOptionsSource:  # free, no key required
    ...
```

Adding a paid source (Tradier, CBOE) = one new class implementing `OptionsSource`. No ingestor changes required.

### Raw Storage

```
data/raw/options/date=YYYY-MM-DD/{ticker}.parquet
```

Schema: `(ticker: Utf8, date: Date, expiry: Date, option_type: Utf8, strike: Float64, iv: Float64, oi: Int64, volume: Int64)`

One row per options contract. If a ticker has no options data (e.g. DARK.L), nothing is written — zero-filled downstream by `join_options_features`.

### Rate Limiting

`time.sleep(0.5)` between tickers. yfinance does not publish a strict rate limit but 0.5s is safe for 141 tickers (~70 seconds total per daily run).

### `__main__` Block

`python ingestion/options_ingestion.py` — fetches today for all 141 tickers. Accepts `--date` flag for historical backfill.

## Processing Architecture

### New File: `processing/options_features.py`

```python
OPTIONS_FEATURE_COLS: list[str] = [
    "iv_rank_30d",
    "iv_hv_spread",
    "put_call_oi_ratio",
    "put_call_vol_ratio",
    "skew_otm",
    "iv_term_slope",
]

def build_options_features(options_dir: Path, ohlcv_dir: Path) -> pl.DataFrame:
    """Aggregate raw options contracts into (ticker, date) feature rows."""
    ...

def join_options_features(df: pl.DataFrame, options_dir: Path, ohlcv_dir: Path) -> pl.DataFrame:
    """Left-join options features to df by (ticker, date). Missing rows zero-fill."""
    ...
```

Joins on `["ticker", "date"]` (left join). Zero-fills on miss. All feature columns are `Float64`.

## File Changes

### `ingestion/options_ingestion.py` (new)
- `OptionsSource` protocol
- `YFinanceOptionsSource` implementation
- `ingest_options(tickers, date_str, output_dir, sources=None)`
- `__main__`: fetch all 141 tickers for today; `--date` flag for backfill

### `processing/options_features.py` (new)
- `OPTIONS_FEATURE_COLS`
- `build_options_features(options_dir, ohlcv_dir) -> pl.DataFrame`
- `join_options_features(df, options_dir, ohlcv_dir) -> pl.DataFrame`

### `models/train.py`
- Import `OPTIONS_FEATURE_COLS`, `join_options_features`
- Append to `FEATURE_COLS` (55 → 61)
- Update `TIER_FEATURE_COLS["short"]` to include all 6 options features
- Add `join_options_features` call in `build_training_dataset` (after `join_fx_features`, before `join_cyber_threat_features`)
- Pass `ohlcv_dir` to the call (already available as a parameter)

### `models/inference.py`
- Import `join_options_features`
- Add call in `_build_feature_df` (after `join_fx_features`)

### `tests/test_options_ingestion.py` (new)
- `YFinanceOptionsSource` returns correct schema with mocked yfinance
- ATM strike selected correctly given known chain
- OTM strikes selected at ~5–10% moneyness
- Graceful empty chain (no options data) returns empty DataFrame, no crash
- `ingest_options` writes parquet at correct path with correct partition

### `tests/test_options_features.py` (new)
- `iv_rank_30d` is in [0, 100]
- `iv_rank_30d` is 50 when fewer than 30 days of history exist
- `iv_hv_spread` sign is correct given known IV and HV inputs
- `join_options_features` adds all 6 columns; row count unchanged; original columns preserved
- Missing `options_dir` → all 6 columns zero-filled (not null)
- Ticker present in df but absent from options data → zero-filled (not null)

## Architecture Constraints

- `DARK.L` and other non-US tickers with no US options chain → zero-fill (handled by `join_options_features` graceful degradation)
- HV30 is computed from OHLCV data already in the pipeline — no additional ingestion module needed
- IV rank requires at least 30 days of daily options snapshots to be meaningful; falls back to 50 (neutral) before that
- `FEATURE_COLS` grows 55 → 61. Existing trained artifacts remain valid (`feature_names.json` governs per-model feature selection)
- Near-term expiry defined as: the options expiry closest to 30 days out (but ≤45 days). Mid-term expiry: closest to 90 days out.

## What This Does NOT Cover

- Greeks (delta, gamma, theta) — require Black-Scholes computation; marginal gain vs. complexity
- Intraday options flow (requires paid data — CBOE DataShop, Tradier streaming)
- Options-derived earnings play detection (future extension via pluggable source)
- Paid sources (Tradier, CBOE) — pluggable architecture leaves the door open

## Success Criteria

1. `pytest tests/ -m 'not integration'` passes with no regressions
2. `OPTIONS_FEATURE_COLS` has exactly 6 elements
3. `len(FEATURE_COLS) == 61`
4. `iv_rank_30d` values are in [0, 100]
5. `join_options_features` zero-fills (not null) when options data is absent
6. `TIER_FEATURE_COLS["long"]` contains no `OPTIONS_FEATURE_COLS`
