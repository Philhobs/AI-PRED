# Supply Chain Relationship Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new features to the model (43 → 47 `FEATURE_COLS`) capturing same-layer peer momentum, ecosystem-wide momentum, supply chain correlation, and peer earnings contagion.

**Architecture:** One new module `processing/supply_chain_features.py` draws from existing OHLCV and earnings parquets using `ingestion/ticker_registry.py`'s layer taxonomy. It pre-builds a wide close-price matrix once per call, then computes all 4 features. Wired into `models/train.py` and `models/inference.py` via `join_supply_chain_features()`.

**Tech Stack:** Python 3.11, Polars, NumPy — all already project dependencies.

---

## Context You Must Know

**Data sources (existing — no new ingestion):**
- OHLCV: `data/raw/financials/ohlcv/{ticker}/{year}.parquet` — columns: `ticker, date, open, high, low, close_price, volume`
- Earnings: `data/raw/financials/earnings/earnings_surprises.parquet` — columns: `ticker, quarter_end, eps_actual, eps_estimate, eps_surprise, eps_surprise_pct`
- Layer map: `ingestion/ticker_registry.py` — `TICKER_LAYERS: dict[str, str]`, `tickers_in_layer(layer)`, `layers()` — 83 tickers across 10 layers

**Pattern to follow:** `processing/energy_geo_features.py` — module-level constants, pure computation functions, one public `join_*()` function that takes a spine DataFrame and returns it with new columns.

**Where to extend train.py:**
- Feature col groups defined around line 44–95
- `FEATURE_COLS` assembled around line 96–101
- `build_training_dataset()` adds joins from line 140 onward — add supply chain join after `join_energy_geo_features(df)` at line 232
- `from processing.energy_geo_features import join_energy_geo_features` is the import pattern to follow (line 39)

**Where to extend inference.py:**
- Imports from `models.train` at lines 27–31 — add `SUPPLY_CHAIN_FEATURE_COLS` here
- `_build_feature_df()` adds joins from line 71 onward — add supply chain join after `join_energy_geo_features(df)` at line 120

**Null-fill fallback pattern (when a data dir doesn't exist):**
```python
for col in SUPPLY_CHAIN_FEATURE_COLS:
    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
```

**Spec:** `docs/superpowers/specs/2026-04-18-supply-chain-features.md`

---

## File Map

| File | Action |
|---|---|
| `processing/supply_chain_features.py` | Create |
| `tests/test_supply_chain_features.py` | Create |
| `models/train.py` | Extend |
| `models/inference.py` | Extend |

---

## Task 1: Supply Chain Features Module

**Files:**
- Create: `processing/supply_chain_features.py`
- Create: `tests/test_supply_chain_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supply_chain_features.py`:

```python
"""Tests for supply chain relationship features."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_close_wide(tickers_daily_ret: dict[str, float], as_of: date, n_days: int = 25) -> pl.DataFrame:
    """
    Build a wide close-price DataFrame where each ticker has a constant daily return.
    n_days rows before as_of, so 20d cumulative return is computable at as_of.
    """
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    data: dict = {"date": dates}
    for ticker, daily_ret in tickers_daily_ret.items():
        prices = [100.0]
        for _ in range(n_days):
            prices.append(prices[-1] * (1 + daily_ret))
        data[ticker] = prices
    return pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))


# ── Momentum tests ────────────────────────────────────────────────────────────

def test_own_layer_momentum_excludes_self():
    """Self's return should not influence own_layer_momentum (peers return ~10.5%)."""
    from processing.supply_chain_features import compute_layer_momentum, _compute_20d_returns

    as_of = date(2025, 6, 1)
    # NVDA in compute layer; AMD, AVGO also compute — give peers 0.5%/day, NVDA 2%/day
    close = _make_close_wide({"NVDA": 0.02, "AMD": 0.005, "AVGO": 0.005}, as_of, n_days=25)
    ret_20d = _compute_20d_returns(close)

    result = compute_layer_momentum("NVDA", as_of, ret_20d, exclude_own_layer=False)
    assert result is not None
    # AMD and AVGO 20d return: (1.005^20) - 1 ≈ 0.1049
    expected = (1.005 ** 20) - 1.0
    assert abs(result - expected) < 0.01, f"Expected ~{expected:.3f}, got {result:.3f}"


def test_ecosystem_momentum_excludes_own_layer():
    """Ecosystem uses only other-layer tickers; own-layer return does not affect it."""
    from processing.supply_chain_features import compute_layer_momentum, _compute_20d_returns

    as_of = date(2025, 6, 1)
    # CEG in power layer; NEE also power → own layer returns 0%
    # MSFT in cloud → other layer, returns ~22% over 20 days
    close = _make_close_wide({"CEG": 0.0, "NEE": 0.0, "MSFT": 0.01}, as_of, n_days=25)
    ret_20d = _compute_20d_returns(close)

    eco  = compute_layer_momentum("CEG", as_of, ret_20d, exclude_own_layer=True)
    own  = compute_layer_momentum("CEG", as_of, ret_20d, exclude_own_layer=False)

    assert eco is not None
    assert eco > 0.05, f"Ecosystem momentum should reflect MSFT's +22%, got {eco}"
    assert own is not None
    assert abs(own) < 0.001, f"Own-layer momentum should be ~0 (NEE returns 0%), got {own}"


def test_own_layer_momentum_null_when_insufficient_peers():
    """Returns null when fewer than 2 same-layer peers have data."""
    from processing.supply_chain_features import compute_layer_momentum, _compute_20d_returns

    as_of = date(2025, 6, 1)
    # Only NVDA in the close matrix — no compute-layer peers
    close = _make_close_wide({"NVDA": 0.01}, as_of, n_days=25)
    ret_20d = _compute_20d_returns(close)

    result = compute_layer_momentum("NVDA", as_of, ret_20d, exclude_own_layer=False)
    assert result is None, f"Expected None with 0 peers, got {result}"


# ── Correlation tests ─────────────────────────────────────────────────────────

def test_supply_chain_correlation_range():
    """Correlation result must be in [-1, 1]."""
    from processing.supply_chain_features import (
        compute_supply_chain_correlation, _CORRELATION_PEERS,
    )

    ticker = "NVDA"
    peers = _CORRELATION_PEERS[ticker]
    as_of = date(2025, 6, 1)
    n_days = 65

    # Alternating returns — all peers in phase with NVDA → correlation ≈ 1.0
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    data: dict = {
        "date": dates,
        ticker: [0.01 * ((-1) ** i) for i in range(n_days + 1)],
    }
    for peer in peers:
        data[peer] = [0.005 * ((-1) ** i) for i in range(n_days + 1)]

    ret_1d = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))

    result = compute_supply_chain_correlation(ticker, as_of, ret_1d)
    assert result is not None
    assert -1.0 <= result <= 1.0, f"Correlation out of range: {result}"


def test_supply_chain_correlation_null_when_insufficient_data():
    """Returns null when fewer than 30 overlapping days exist."""
    from processing.supply_chain_features import compute_supply_chain_correlation

    as_of = date(2025, 6, 1)
    # Only 20 days — not enough for the 30-day minimum
    n_days = 20
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    ret_1d = pl.DataFrame({
        "date": dates,
        "NVDA": [0.01] * (n_days + 1),
    }).with_columns(pl.col("date").cast(pl.Date))

    result = compute_supply_chain_correlation("NVDA", as_of, ret_1d)
    assert result is None, f"Expected None with only {n_days} days of data, got {result}"


# ── Earnings tests ────────────────────────────────────────────────────────────

def test_peer_eps_surprise_excludes_self():
    """Ticker's own EPS surprise must not be counted in peer mean."""
    from processing.supply_chain_features import compute_peer_eps_surprise

    as_of = date(2025, 6, 1)
    earnings = pl.DataFrame({
        "ticker":          ["NVDA",                          "AMD"],
        "quarter_end":     [as_of - timedelta(days=30),      as_of - timedelta(days=45)],
        "eps_surprise":    [0.5,                             0.1],
        "eps_surprise_pct":[0.50,                            0.10],
    })

    result = compute_peer_eps_surprise("NVDA", as_of, earnings)
    assert result is not None
    assert abs(result - 0.10) < 0.001, f"Expected 0.10 (AMD only), got {result}"


def test_peer_eps_surprise_uses_90d_window():
    """Earnings older than 90 days are excluded."""
    from processing.supply_chain_features import compute_peer_eps_surprise

    as_of = date(2025, 6, 1)
    earnings = pl.DataFrame({
        "ticker":          ["AMD",                        "AMD"],
        "quarter_end":     [as_of - timedelta(days=89),   as_of - timedelta(days=91)],
        "eps_surprise":    [0.2,                          0.8],
        "eps_surprise_pct":[0.20,                         0.80],
    })

    result = compute_peer_eps_surprise("NVDA", as_of, earnings)
    assert result is not None
    assert abs(result - 0.20) < 0.001, f"Expected 0.20 (only 89-day report), got {result}"


def test_peer_eps_surprise_null_when_no_peers_reported():
    """Returns null (not 0.0) when no peer reported in the 90-day window."""
    from processing.supply_chain_features import compute_peer_eps_surprise

    as_of = date(2025, 6, 1)
    earnings = pl.DataFrame({
        "ticker":          ["AMD"],
        "quarter_end":     [as_of - timedelta(days=120)],   # outside 90d
        "eps_surprise":    [0.5],
        "eps_surprise_pct":[0.50],
    })

    result = compute_peer_eps_surprise("NVDA", as_of, earnings)
    assert result is None, f"Expected None, got {result}"


def test_join_supply_chain_features_adds_four_columns(tmp_path):
    """join_supply_chain_features adds exactly 4 Float64 columns (nulls when no data)."""
    from processing.supply_chain_features import join_supply_chain_features

    ohlcv_dir    = tmp_path / "ohlcv";    ohlcv_dir.mkdir()
    earnings_dir = tmp_path / "earnings"; earnings_dir.mkdir()

    spine = pl.DataFrame({
        "ticker": ["NVDA", "AMD"],
        "date":   [date(2025, 6, 1), date(2025, 6, 1)],
    })

    result = join_supply_chain_features(spine, ohlcv_dir=ohlcv_dir, earnings_dir=earnings_dir)

    for col in ["own_layer_momentum_20d", "ecosystem_momentum_20d",
                "supply_chain_correlation_60d", "peer_eps_surprise_mean"]:
        assert col in result.columns, f"Missing column: {col}"
        assert result[col].dtype == pl.Float64, f"{col} should be Float64"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor"
pytest tests/test_supply_chain_features.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'processing.supply_chain_features'`

- [ ] **Step 3: Create `processing/supply_chain_features.py`**

```python
"""
Supply chain relationship features — peer layer momentum, ecosystem momentum,
supply chain correlation, and peer earnings contagion.

Produces 4 features per ticker per date:
  own_layer_momentum_20d       — avg 20-trading-day return of same-layer peers (excl. self)
  ecosystem_momentum_20d       — avg 20-trading-day return of all other-layer tickers
  supply_chain_correlation_60d — mean 60-day rolling Pearson corr with 20 fixed other-layer peers
  peer_eps_surprise_mean       — mean EPS surprise pct of same-layer peers in last 90 days

Called by models/train.py via join_supply_chain_features().
"""
from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ingestion.ticker_registry import TICKER_LAYERS, tickers_in_layer, layers as all_layers

_LOG = logging.getLogger(__name__)

# ── Layer maps (built at module load time) ──────────────────────────────────

_LAYER_MAP: dict[str, str] = dict(TICKER_LAYERS)  # ticker → layer name

_LAYER_TICKERS: dict[str, list[str]] = {
    layer: tickers_in_layer(layer) for layer in all_layers()
}

_ALL_TICKERS: list[str] = sorted(_LAYER_MAP.keys())


# ── Correlation peers (fixed at module load, seed=42) ──────────────────────

def _build_correlation_peers() -> dict[str, list[str]]:
    rng = random.Random(42)
    peers: dict[str, list[str]] = {}
    for ticker, layer in _LAYER_MAP.items():
        other = [t for t in _ALL_TICKERS if _LAYER_MAP[t] != layer]
        peers[ticker] = rng.sample(other, min(20, len(other)))
    return peers


_CORRELATION_PEERS: dict[str, list[str]] = _build_correlation_peers()


# ── Returns matrix helpers ──────────────────────────────────────────────────

def _build_close_matrix(
    ohlcv_dir: Path,
    min_date: date,
    max_date: date,
    extra_days: int = 100,
) -> pl.DataFrame:
    """
    Wide close-price matrix: date × ticker columns.
    Loads from (min_date - extra_days) to max_date for rolling window history.
    """
    load_start = min_date - timedelta(days=extra_days)

    frames = []
    for ticker in _ALL_TICKERS:
        ticker_dir = ohlcv_dir / ticker
        if not ticker_dir.exists():
            continue
        year_files = sorted(
            f for f in ticker_dir.glob("*.parquet")
            if int(f.stem) >= load_start.year
        )
        if not year_files:
            continue
        df = pl.concat([pl.read_parquet(f) for f in year_files])
        df = (
            df.filter((pl.col("date") >= load_start) & (pl.col("date") <= max_date))
            .sort("date")
            .select(["date", pl.col("close_price").alias(ticker)])
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="date", how="full", coalesce=True)
    return result.sort("date")


def _compute_20d_returns(close_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Wide DataFrame of 20-trading-day cumulative returns.
    ret_20d[ticker][d] = close[d] / close[d-20 rows] - 1
    """
    ticker_cols = [c for c in close_wide.columns if c != "date"]
    return close_wide.select(
        ["date"] + [
            (pl.col(t) / pl.col(t).shift(20) - 1).alias(t)
            for t in ticker_cols
        ]
    )


# ── Feature computation functions ──────────────────────────────────────────

def compute_layer_momentum(
    ticker: str,
    as_of: date,
    ret_20d_wide: pl.DataFrame,
    exclude_own_layer: bool = False,
) -> float | None:
    """
    Compute mean 20-trading-day return of:
      - same-layer peers (exclude_own_layer=False) — excludes self
      - all other-layer tickers (exclude_own_layer=True)

    Returns None when fewer than 2 (own-layer) or 5 (ecosystem) valid values exist.
    """
    layer = _LAYER_MAP.get(ticker)
    if layer is None:
        return None

    if exclude_own_layer:
        target_tickers = [t for t in _ALL_TICKERS if _LAYER_MAP.get(t) != layer]
        min_peers = 5
    else:
        target_tickers = [t for t in _LAYER_TICKERS.get(layer, []) if t != ticker]
        min_peers = 2

    row = ret_20d_wide.filter(pl.col("date") == as_of)
    if row.is_empty():
        return None

    vals = []
    for t in target_tickers:
        if t not in row.columns:
            continue
        v = row[t][0]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))

    return float(np.mean(vals)) if len(vals) >= min_peers else None


def compute_supply_chain_correlation(
    ticker: str,
    as_of: date,
    ret_1d_wide: pl.DataFrame,
) -> float | None:
    """
    Mean rolling 60-trading-day Pearson correlation between ticker and its 20
    fixed other-layer peers from _CORRELATION_PEERS.

    Returns None when fewer than 5 peers have 30+ overlapping return days.
    """
    peers = _CORRELATION_PEERS.get(ticker, [])
    if not peers or ticker not in ret_1d_wide.columns:
        return None

    all_dates = ret_1d_wide["date"].to_list()
    # Find last row index where date <= as_of
    idx = next(
        (i for i, d in reversed(list(enumerate(all_dates))) if d <= as_of),
        None,
    )
    if idx is None or idx < 1:
        return None

    start_idx = max(0, idx - 59)
    window = ret_1d_wide.slice(start_idx, idx - start_idx + 1)

    if window.height < 30:
        return None

    t_returns = window[ticker].to_numpy().astype(float)

    valid_corrs = []
    for peer in peers:
        if peer not in window.columns:
            continue
        p_returns = window[peer].to_numpy().astype(float)
        mask = ~(np.isnan(t_returns) | np.isnan(p_returns))
        if mask.sum() < 30:
            continue
        corr = float(np.corrcoef(t_returns[mask], p_returns[mask])[0, 1])
        if not np.isnan(corr):
            valid_corrs.append(corr)

    return float(np.mean(valid_corrs)) if len(valid_corrs) >= 5 else None


def compute_peer_eps_surprise(
    ticker: str,
    as_of: date,
    earnings_df: pl.DataFrame,
) -> float | None:
    """
    Mean EPS surprise pct of same-layer peers with earnings reported in last 90 days.

    Returns None (not 0) when no qualifying peer earnings exist — absence of
    earnings data is not a neutral signal.
    """
    layer = _LAYER_MAP.get(ticker)
    if layer is None:
        return None

    peers = [t for t in _LAYER_TICKERS.get(layer, []) if t != ticker]
    if not peers:
        return None

    if earnings_df.is_empty():
        return None

    window_start = as_of - timedelta(days=90)
    peer_earnings = earnings_df.filter(
        pl.col("ticker").is_in(peers)
        & (pl.col("quarter_end") >= window_start)
        & (pl.col("quarter_end") <= as_of)
    )

    if peer_earnings.is_empty():
        return None

    vals = peer_earnings["eps_surprise_pct"].drop_nulls().to_list()
    return float(np.mean(vals)) if vals else None


# ── Public join function ────────────────────────────────────────────────────

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
        earnings_dir: Path to data/raw/financials/earnings/ (default: resolved from __file__).

    Returns df with 4 new Float64 columns. Missing data → null (not 0).
    """
    _ROOT = Path(__file__).parent.parent
    if ohlcv_dir is None:
        ohlcv_dir = _ROOT / "data" / "raw" / "financials" / "ohlcv"
    if earnings_dir is None:
        earnings_dir = _ROOT / "data" / "raw" / "financials" / "earnings"

    _NULLS = [
        pl.lit(None).cast(pl.Float64).alias("own_layer_momentum_20d"),
        pl.lit(None).cast(pl.Float64).alias("ecosystem_momentum_20d"),
        pl.lit(None).cast(pl.Float64).alias("supply_chain_correlation_60d"),
        pl.lit(None).cast(pl.Float64).alias("peer_eps_surprise_mean"),
    ]

    min_date = df["date"].min()
    max_date = df["date"].max()

    close_wide = _build_close_matrix(ohlcv_dir, min_date, max_date)
    if close_wide.is_empty() or close_wide.height < 2:
        _LOG.warning("[SupplyChain] No OHLCV data found — supply chain features will be null")
        return df.with_columns(_NULLS)

    ret_20d_wide = _compute_20d_returns(close_wide)

    ticker_cols = [c for c in close_wide.columns if c != "date"]
    ret_1d_wide = close_wide.select(
        ["date"] + [pl.col(t).pct_change(1).alias(t) for t in ticker_cols]
    )

    earnings_path = earnings_dir / "earnings_surprises.parquet"
    earnings_df = (
        pl.read_parquet(earnings_path)
        if earnings_path.exists()
        else pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "quarter_end": pl.Date,
            "eps_surprise": pl.Float64,
            "eps_surprise_pct": pl.Float64,
        })
    )

    own_mom_vals: list[float | None]  = []
    eco_mom_vals: list[float | None]  = []
    corr_vals:    list[float | None]  = []
    eps_vals:     list[float | None]  = []

    for row in df.select(["ticker", "date"]).iter_rows(named=True):
        ticker = row["ticker"]
        as_of  = row["date"]

        own_mom_vals.append(compute_layer_momentum(ticker, as_of, ret_20d_wide, exclude_own_layer=False))
        eco_mom_vals.append(compute_layer_momentum(ticker, as_of, ret_20d_wide, exclude_own_layer=True))
        corr_vals.append(compute_supply_chain_correlation(ticker, as_of, ret_1d_wide))
        eps_vals.append(compute_peer_eps_surprise(ticker, as_of, earnings_df))

    return df.with_columns([
        pl.Series("own_layer_momentum_20d",       own_mom_vals, dtype=pl.Float64),
        pl.Series("ecosystem_momentum_20d",        eco_mom_vals, dtype=pl.Float64),
        pl.Series("supply_chain_correlation_60d",  corr_vals,    dtype=pl.Float64),
        pl.Series("peer_eps_surprise_mean",         eps_vals,     dtype=pl.Float64),
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _LOG.info("Supply chain features are computed on-demand in train.py — no standalone run needed.")
    _LOG.info("Ensure OHLCV data exists at data/raw/financials/ohlcv/ before running train.py.")
```

- [ ] **Step 4: Run the tests**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor"
pytest tests/test_supply_chain_features.py -v
```

Expected: 9 PASSED

- [ ] **Step 5: Run the full suite to check for regressions**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass (217+ tests).

- [ ] **Step 6: Commit**

```bash
git add processing/supply_chain_features.py tests/test_supply_chain_features.py
git commit -m "feat: supply chain features — peer momentum, ecosystem momentum, correlation, EPS contagion (Task 1)"
```

---

## Task 2: FEATURE_COLS Integration

**Files:**
- Modify: `models/train.py`
- Modify: `models/inference.py`

- [ ] **Step 1: Read the current state of both files**

Read `models/train.py` lines 83–101 and 206–237 to see the exact FEATURE_COLS assembly and build_training_dataset join block.

Read `models/inference.py` lines 27–31 and 103–122 to see the import block and `_build_feature_df` join block.

- [ ] **Step 2: Extend `models/train.py`**

After line 95 (end of `ENERGY_FEATURE_COLS`), add:

```python
SUPPLY_CHAIN_FEATURE_COLS = [
    "own_layer_momentum_20d",
    "ecosystem_momentum_20d",
    "supply_chain_correlation_60d",
    "peer_eps_surprise_mean",
]
```

Change line 96–101 (the `FEATURE_COLS` assembly) to append `+ SUPPLY_CHAIN_FEATURE_COLS`:

```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS + ENERGY_FEATURE_COLS
    + SUPPLY_CHAIN_FEATURE_COLS  # 43 → 47 features total
)
```

Add import at line 39 (with the other processing imports):

```python
from processing.supply_chain_features import join_supply_chain_features
```

In `build_training_dataset()`, after `df = join_energy_geo_features(df)` (line 232), add:

```python
    # Join supply chain relationship features (own-layer momentum, ecosystem momentum,
    # correlation, peer EPS contagion) — no external dir needed, uses ohlcv_dir.
    df = join_supply_chain_features(df, ohlcv_dir=ohlcv_dir)
```

- [ ] **Step 3: Extend `models/inference.py`**

Add `SUPPLY_CHAIN_FEATURE_COLS` to the import from `models.train` (around line 27–31):

```python
from models.train import (
    FEATURE_COLS, INSIDER_FEATURE_COLS, SENTIMENT_FEATURE_COLS,
    SHORT_INTEREST_FEATURE_COLS, EARNINGS_FEATURE_COLS, GRAPH_FEATURE_COLS,
    OWNERSHIP_FEATURE_COLS, ENERGY_FEATURE_COLS, SUPPLY_CHAIN_FEATURE_COLS,
)
```

Add import of the join function (with other processing imports):

```python
from processing.supply_chain_features import join_supply_chain_features
```

In `_build_feature_df()`, after `df = join_energy_geo_features(df)` (line 120), add:

```python
    df = join_supply_chain_features(df, ohlcv_dir=ohlcv_dir)
```

Also update the docstring comment on line 61 from `43-feature` to `47-feature`.

- [ ] **Step 4: Verify feature count**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor"
python -c "
from models.train import FEATURE_COLS
print(f'Feature count: {len(FEATURE_COLS)}')
print('Last 5:', FEATURE_COLS[-5:])
"
```

Expected:
```
Feature count: 47
Last 5: ['us_power_moat_score', 'geo_weighted_tailwind_score', 'energy_deal_mw_90d', 'hyperscaler_ppa_count_90d', 'own_layer_momentum_20d']
```

Wait — the last 5 should be the 4 supply chain features + the last energy feature. Verify the order is correct: ENERGY_FEATURE_COLS last item is `hyperscaler_ppa_count_90d`, then SUPPLY_CHAIN starts with `own_layer_momentum_20d`.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass (226+ tests).

- [ ] **Step 6: Run train.py end-to-end**

```bash
python models/train.py 2>&1 | tail -10
```

Expected: Completes without errors. Then verify:

```bash
python -c "
import json
r = json.load(open('data/backtest/walk_forward_results.json'))
print('feature_count:', r['feature_count'])
"
```

Expected: `feature_count: 47`

- [ ] **Step 7: Commit**

```bash
git add models/train.py models/inference.py
git commit -m "feat: wire supply chain features into FEATURE_COLS — model now trains on 47 features (Task 2)"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `own_layer_momentum_20d` — computed in `compute_layer_momentum(..., exclude_own_layer=False)`
- ✅ `ecosystem_momentum_20d` — computed in `compute_layer_momentum(..., exclude_own_layer=True)`
- ✅ `supply_chain_correlation_60d` — computed in `compute_supply_chain_correlation()`
- ✅ `peer_eps_surprise_mean` — computed in `compute_peer_eps_surprise()` using `eps_surprise_pct`
- ✅ Wide returns matrix built once — `_build_close_matrix()` + `_compute_20d_returns()`
- ✅ `_CORRELATION_PEERS` fixed at module load with seed=42
- ✅ Null conditions: < 2 own-layer peers → null; < 5 eco-tickers → null; < 5 correlation peers with 30d data → null; no peer earnings in 90d → null
- ✅ 8 tests (9 actually — one extra for the public join interface)
- ✅ train.py + inference.py both extended

**Placeholder scan:** No TBD, no vague steps. All code blocks complete.

**Type consistency:**
- `compute_layer_momentum` → `float | None` — used as `own_mom_vals` elements
- `compute_supply_chain_correlation` → `float | None`
- `compute_peer_eps_surprise` → `float | None`
- All stored in typed lists and cast to `pl.Float64` in the final `with_columns`
- `_compute_20d_returns` is imported in tests — it's a private function but tests importing private functions is established project pattern (see `test_deal_enrichment.py`)
