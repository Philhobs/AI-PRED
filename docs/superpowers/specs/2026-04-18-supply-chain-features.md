# Phase D — Supply Chain Relationship Features Design

**Goal:** Add 4 new features (43 → 47 `FEATURE_COLS`) capturing same-layer peer momentum, ecosystem-wide momentum, supply chain correlation, and peer earnings contagion — turning the 83-ticker layer taxonomy into quantifiable inter-company relationship signals.

**Architecture:** One new processing module (`processing/supply_chain_features.py`) drawing from existing OHLCV and earnings parquets + the ticker registry layer map. No new ingestion. Integrated into `models/train.py` and `models/inference.py` via `join_supply_chain_features()`.

---

## Context: What Already Exists

- `ingestion/ticker_registry.py` — maps every ticker to a named layer (cloud, compute, semi_equipment, networking, servers, datacenter, power, cooling, grid, metals). 83 tickers across 10 layers.
- `data/raw/financials/ohlcv/{ticker}/{year}.parquet` — daily OHLCV per ticker, columns include `date` and `close_price` (returns computed from close_price).
- `data/raw/fundamentals/earnings/{ticker}.parquet` — quarterly earnings per ticker, includes `date` and `eps_surprise_last`.
- `processing/graph_features.py` — existing supply chain graph features (deal-based). Phase D adds statistical features on top.
- `processing/energy_geo_features.py` — pattern to follow for `join_supply_chain_features()`.

---

## New Features (43 → 47)

| Feature | Source | Description |
|---|---|---|
| `own_layer_momentum_20d` | OHLCV + layer map | Avg 20d return of same-layer peers, excluding self |
| `ecosystem_momentum_20d` | OHLCV + layer map | Avg 20d return of all tickers in other layers |
| `supply_chain_correlation_60d` | OHLCV + layer map | Avg rolling 60d Pearson correlation with 20 fixed other-layer peers |
| `peer_eps_surprise_mean` | Earnings + layer map | Avg EPS surprise of same-layer peers that reported in last 90 days |

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `processing/supply_chain_features.py` | Create | All 4 feature computations + `join_supply_chain_features()` |
| `models/train.py` | Extend | Add `SUPPLY_CHAIN_FEATURE_COLS`, call `join_supply_chain_features()` |
| `models/inference.py` | Extend | Mirror train.py changes |
| `tests/test_supply_chain_features.py` | Create | 8 unit tests |

---

## 1. Processing Module (`processing/supply_chain_features.py`)

### 1a. Module-level setup

At module load time:
1. Import the ticker-to-layer mapping from `ingestion/ticker_registry.py` — build `_LAYER_MAP: dict[str, str]` (ticker → layer name) and `_LAYER_TICKERS: dict[str, list[str]]` (layer name → list of tickers).
2. Compute `_CORRELATION_PEERS: dict[str, list[str]]` — for each ticker, sample 20 tickers from other layers using `random.sample(seed=42)`. Fixed at module load; same peers used for every date.

### 1b. Returns matrix

`join_supply_chain_features(df)` builds a full returns matrix once at call time:
- Load all OHLCV parquets for tickers in the spine's date range
- Pivot to wide format: `date (rows) × ticker (columns)`, values = `pct_change(1)` daily returns
- Used for all momentum and correlation computations — no repeated file reads

### 1c. `own_layer_momentum_20d`

For ticker T on date D:
1. Get same-layer peers: `_LAYER_TICKERS[layer(T)] - {T}`
2. For each peer, compute the 20-trading-day cumulative return ending on D: `(close[D] / close[D-20 trading days]) - 1`
3. Return equal-weighted mean across peers

Returns `null` if fewer than 2 same-layer peers have data at D.

### 1d. `ecosystem_momentum_20d`

For ticker T on date D:
1. Get all tickers in layers other than `layer(T)`
2. For each, compute 20-trading-day cumulative return ending on D
3. Return equal-weighted mean

Returns `null` if fewer than 5 other-layer tickers have data at D.

### 1e. `supply_chain_correlation_60d`

For ticker T on date D:
1. Get T's 20 fixed other-layer peers from `_CORRELATION_PEERS[T]`
2. For each peer P, compute Pearson correlation between T's and P's daily returns over the 60 trading days ending on D (requires ≥ 30 non-null overlapping days)
3. Return mean of valid correlations

Returns `null` if fewer than 5 peers have sufficient data.

### 1f. `peer_eps_surprise_mean`

For ticker T on date D:
1. Get same-layer peers: `_LAYER_TICKERS[layer(T)] - {T}`
2. Load each peer's earnings parquet, filter to rows where `date >= D - 90 days` and `date <= D`
3. Return equal-weighted mean of `eps_surprise_last` across peers with qualifying reports

Returns `null` (not 0) if no same-layer peer reported in the 90-day window — absence of data is not a neutral signal.

### 1g. Public interface

```python
def join_supply_chain_features(
    df: pl.DataFrame,
    ohlcv_dir: Path | None = None,
    earnings_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Add own_layer_momentum_20d, ecosystem_momentum_20d,
    supply_chain_correlation_60d, peer_eps_surprise_mean to training spine.

    Args:
        df: Training spine with columns [ticker, date, ...].
        ohlcv_dir: Path to data/raw/financials/ohlcv/ (default: resolved from __file__).
        earnings_dir: Path to data/raw/fundamentals/earnings/ (default: resolved from __file__).

    Returns df with 4 new Float64 columns. Missing data → null (not 0).
    """
```

---

## 2. Model Integration (`models/train.py`)

Add after existing feature group definitions:

```python
SUPPLY_CHAIN_FEATURE_COLS = [
    "own_layer_momentum_20d",
    "ecosystem_momentum_20d",
    "supply_chain_correlation_60d",
    "peer_eps_surprise_mean",
]
```

Add `+ SUPPLY_CHAIN_FEATURE_COLS` to `FEATURE_COLS` assembly.

Add import:
```python
from processing.supply_chain_features import join_supply_chain_features
```

Add call in `build_training_dataset()` after existing joins:
```python
df = join_supply_chain_features(df)
```

Mirror same changes in `models/inference.py`.

---

## 3. Testing (`tests/test_supply_chain_features.py`)

### Momentum tests

- **`test_own_layer_momentum_excludes_self`** — ticker appears in its own layer; assert its own return is not included in the peer average. Synthetic returns: self returns 50%, peers return 10% → result is 10%, not biased toward 50%.
- **`test_ecosystem_momentum_excludes_own_layer`** — own-layer tickers all return 0%, other-layer tickers return 5% → `ecosystem_momentum_20d = 0.05`, `own_layer_momentum_20d = 0.0`.
- **`test_own_layer_momentum_null_when_insufficient_peers`** — ticker is in a layer with only 1 other peer that has no OHLCV data → result is `null`.

### Correlation tests

- **`test_supply_chain_correlation_range`** — synthetic returns with known positive correlation → result in [-1, 1].
- **`test_supply_chain_correlation_null_when_insufficient_data`** — fewer than 5 peers have 30+ days of data → result is `null`.

### Earnings contagion tests

- **`test_peer_eps_surprise_excludes_self`** — ticker's own EPS surprise is not included in its peer mean. Self reports +50% surprise, peers report +10% → result is +10%.
- **`test_peer_eps_surprise_uses_90d_window`** — peer earnings report dated 91 days before spine date → not included; report dated 89 days before → included.
- **`test_peer_eps_surprise_null_when_no_peers_reported`** — no same-layer peer has a report in the 90-day window → result is `null`, not `0.0`.

---

## 4. Null Handling Summary

| Feature | Null condition |
|---|---|
| `own_layer_momentum_20d` | < 2 same-layer peers have OHLCV data at the spine date |
| `ecosystem_momentum_20d` | < 5 other-layer tickers have OHLCV data at the spine date |
| `supply_chain_correlation_60d` | < 5 peers have ≥ 30 days of overlapping returns in 60d window |
| `peer_eps_surprise_mean` | No same-layer peer reported earnings in the last 90 days |

LightGBM and the other sub-models handle null natively — no imputation needed here (consistent with existing feature pipeline).

---

## 5. Cadence / Stale Thresholds (pipeline_health.py)

| Feature group | Update frequency | Stale threshold |
|---|---|---|
| Momentum + correlation | Daily (OHLCV) | 3 days |
| Peer EPS surprise | Quarterly (earnings) | 95 days |
