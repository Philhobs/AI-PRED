# Portfolio Metrics Post-Processor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-processing step after inference that enriches predictions with market cap (liquidity filter), model agreement score, and peer correlation — making the output actionable rather than just ranked.

**Architecture:** New `processing/portfolio_metrics.py` module. Reads raw predictions parquet, adds 4 columns (`market_cap_b`, `is_liquid`, `model_agreement`, `peer_correlation_90d`), writes `predictions_enriched.parquet` alongside the original. Called automatically from `models/inference.py` after predictions are written.

**Tech Stack:** Python 3.11, Polars, NumPy, yfinance (already a dep), time (stdlib)

---

## Context: Existing Prediction Schema

`data/predictions/date=YYYY-MM-DD/predictions.parquet` has these columns:

```
rank, ticker, layer, expected_annual_return,
confidence_low, confidence_high,
lgbm_return, rf_return, ridge_return,
as_of_date
```

The enriched parquet adds 4 columns to this. The original is kept unchanged.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `processing/portfolio_metrics.py` | Create | All enrichment logic |
| `models/inference.py` | Modify | Call `enrich()` at end of `run_inference()` |
| `tests/test_portfolio_metrics.py` | Create | 5 unit tests |

---

## Task 1: Model Agreement + Liquidity Filter

**Files:**
- Create: `processing/portfolio_metrics.py` (partial — agreement + liquidity)
- Create: `tests/test_portfolio_metrics.py` (partial)

- [ ] **Step 1: Write tests for model agreement and liquidity**

```python
# tests/test_portfolio_metrics.py
"""Tests for portfolio_metrics enrichment functions."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import tempfile

import numpy as np
import polars as pl
import pytest


def _make_predictions(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal predictions DataFrame for testing."""
    return pl.DataFrame({
        "rank": list(range(1, len(rows) + 1)),
        "ticker": [r["ticker"] for r in rows],
        "layer": [r.get("layer", "compute") for r in rows],
        "expected_annual_return": [r["ensemble"] for r in rows],
        "confidence_low": [0.0] * len(rows),
        "confidence_high": [1.0] * len(rows),
        "lgbm_return": [r["lgbm"] for r in rows],
        "rf_return": [r["rf"] for r in rows],
        "ridge_return": [r["ridge"] for r in rows],
        "as_of_date": [date(2026, 4, 15)] * len(rows),
    })


def test_model_agreement_all_agree():
    from processing.portfolio_metrics import _model_agreement
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.5, "lgbm": 0.6, "rf": 0.4, "ridge": 0.3},
    ])
    result = _model_agreement(df)
    assert result[0] == pytest.approx(1.0), "All three sub-models positive → agreement=1.0"


def test_model_agreement_one_disagrees():
    from processing.portfolio_metrics import _model_agreement
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.5, "lgbm": 0.6, "rf": 0.4, "ridge": -0.2},
    ])
    result = _model_agreement(df)
    assert result[0] == pytest.approx(2 / 3), "Two agree, one disagrees → agreement=0.667"


def test_model_agreement_none_agree():
    from processing.portfolio_metrics import _model_agreement
    df = _make_predictions([
        # ensemble is positive but all sub-models are negative
        {"ticker": "NVDA", "ensemble": 0.1, "lgbm": -0.6, "rf": -0.4, "ridge": -0.2},
    ])
    result = _model_agreement(df)
    assert result[0] == pytest.approx(0.0), "All sub-models disagree → agreement=0.0"


def test_is_liquid_threshold():
    from processing.portfolio_metrics import _apply_liquidity
    df = _make_predictions([
        {"ticker": "APLD", "ensemble": 3.0, "lgbm": 2.0, "rf": 3.5, "ridge": 3.5},
        {"ticker": "TSM",  "ensemble": 2.0, "lgbm": 1.5, "rf": 2.5, "ridge": 2.0},
    ])
    caps = {"APLD": 0.3, "TSM": 650.0}
    result = _apply_liquidity(df, caps)
    assert result.filter(pl.col("ticker") == "APLD")["is_liquid"][0] == False
    assert result.filter(pl.col("ticker") == "TSM")["is_liquid"][0] == True


def test_enrich_writes_enriched_parquet(tmp_path, monkeypatch):
    from processing import portfolio_metrics

    # Write a mock predictions parquet
    date_str = "2026-04-15"
    pred_dir = tmp_path / f"date={date_str}"
    pred_dir.mkdir()
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.9, "lgbm": 0.8, "rf": 0.9, "ridge": 1.0},
        {"ticker": "TSM",  "ensemble": 2.1, "lgbm": 1.5, "rf": 2.5, "ridge": 2.3},
    ])
    df.write_parquet(pred_dir / "predictions.parquet")

    # Mock _get_market_caps so no yfinance call
    monkeypatch.setattr(
        portfolio_metrics, "_get_market_caps",
        lambda tickers, as_of: {"NVDA": 2800.0, "TSM": 650.0},
    )
    # Mock _peer_correlation to return zeros (no OHLCV needed)
    # Must return a named Series so with_columns() works
    monkeypatch.setattr(
        portfolio_metrics, "_peer_correlation",
        lambda df, ohlcv_dir, as_of, top_n=10: pl.Series("peer_correlation_90d", [0.3, 0.4]),
    )

    portfolio_metrics.enrich(date_str, predictions_dir=tmp_path)

    enriched_path = pred_dir / "predictions_enriched.parquet"
    assert enriched_path.exists(), "Enriched parquet must be written"
    enriched = pl.read_parquet(enriched_path)
    assert "market_cap_b" in enriched.columns
    assert "is_liquid" in enriched.columns
    assert "model_agreement" in enriched.columns
    assert "peer_correlation_90d" in enriched.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_portfolio_metrics.py -v
```

Expected: All 5 FAIL (`processing.portfolio_metrics` doesn't exist yet).

- [ ] **Step 3: Write `processing/portfolio_metrics.py` — agreement + liquidity functions**

```python
# processing/portfolio_metrics.py
"""
Portfolio metrics post-processor.

Enriches raw predictions with:
  market_cap_b        — market cap in billions (yfinance, cached daily)
  is_liquid           — True if market_cap_b >= 1.0
  model_agreement     — fraction of sub-models agreeing with ensemble direction
  peer_correlation_90d — avg pairwise 90d return correlation with other top-10 picks

Usage:
  python processing/portfolio_metrics.py 2026-04-15   # enrich specific date
  python processing/portfolio_metrics.py              # enrich today
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)
_LIQUIDITY_THRESHOLD_B = 1.0  # $1B minimum market cap


def _model_agreement(df: pl.DataFrame) -> pl.Series:
    """Fraction of sub-models (lgbm, rf, ridge) agreeing with ensemble direction.

    1.0 = all three point same way as ensemble.
    0.67 = two of three agree.
    0.33 = one of three agrees.
    0.0 = none agree.
    """
    ensemble_sign = np.sign(df["expected_annual_return"].to_numpy())
    agreements = []
    for col in ["lgbm_return", "rf_return", "ridge_return"]:
        sub_sign = np.sign(df[col].to_numpy())
        agreements.append((sub_sign == ensemble_sign).astype(float))
    mean_agreement = (agreements[0] + agreements[1] + agreements[2]) / 3.0
    return pl.Series("model_agreement", mean_agreement.tolist())


def _apply_liquidity(df: pl.DataFrame, caps: dict[str, float | None]) -> pl.DataFrame:
    """Add market_cap_b and is_liquid columns using preloaded caps dict."""
    market_cap_b = pl.Series(
        "market_cap_b",
        [caps.get(t) for t in df["ticker"].to_list()],
        dtype=pl.Float64,
    )
    is_liquid = pl.Series(
        "is_liquid",
        [
            (c is not None and c >= _LIQUIDITY_THRESHOLD_B)
            for c in market_cap_b.to_list()
        ],
        dtype=pl.Boolean,
    )
    return df.with_columns([market_cap_b, is_liquid])
```

- [ ] **Step 4: Run the first 4 tests**

```bash
pytest tests/test_portfolio_metrics.py::test_model_agreement_all_agree \
       tests/test_portfolio_metrics.py::test_model_agreement_one_disagrees \
       tests/test_portfolio_metrics.py::test_model_agreement_none_agree \
       tests/test_portfolio_metrics.py::test_is_liquid_threshold -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit partial**

```bash
git add processing/portfolio_metrics.py tests/test_portfolio_metrics.py
git commit -m "feat: portfolio_metrics — model agreement and liquidity filter"
```

---

## Task 2: Market Cap Cache

**Files:**
- Modify: `processing/portfolio_metrics.py` (add `_get_market_caps`)

- [ ] **Step 1: Add `_get_market_caps` to `processing/portfolio_metrics.py`**

Append this function after `_apply_liquidity`:

```python
def _get_market_caps(tickers: list[str], as_of: date) -> dict[str, float | None]:
    """
    Return {ticker: market_cap_billions} for all tickers.

    Uses a daily cache at data/raw/financials/market_caps.parquet.
    Fetches from yfinance only for tickers missing from today's cache.
    """
    cache_path = Path("data/raw/financials/market_caps.parquet")

    # Load today's cached values
    cached: dict[str, float | None] = {}
    if cache_path.exists():
        try:
            cache_df = pl.read_parquet(cache_path).filter(pl.col("date") == as_of)
            # Only use rows with non-null market_cap_b
            valid = cache_df.filter(pl.col("market_cap_b").is_not_null())
            cached = dict(zip(valid["ticker"].to_list(), valid["market_cap_b"].to_list()))
        except Exception as exc:
            _LOG.warning("Failed to read market cap cache: %s", exc)

    missing = [t for t in tickers if t not in cached]
    if not missing:
        return {t: cached.get(t) for t in tickers}

    # Fetch missing tickers from yfinance
    fetched: dict[str, float | None] = {}
    for ticker in missing:
        try:
            info = yf.Ticker(ticker).fast_info
            market_cap = getattr(info, "market_cap", None)
            fetched[ticker] = round(market_cap / 1e9, 3) if market_cap else None
        except Exception as exc:
            _LOG.debug("Failed to fetch market cap for %s: %s", ticker, exc)
            fetched[ticker] = None
        time.sleep(0.1)

    # Append to cache
    if fetched:
        new_rows = pl.DataFrame({
            "ticker": list(fetched.keys()),
            "date": [as_of] * len(fetched),
            "market_cap_b": list(fetched.values()),
        })
        if cache_path.exists():
            existing = pl.read_parquet(cache_path).filter(pl.col("date") != as_of)
            pl.concat([existing, new_rows]).write_parquet(cache_path, compression="snappy")
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            new_rows.write_parquet(cache_path, compression="snappy")

    all_caps = {**cached, **fetched}
    return {t: all_caps.get(t) for t in tickers}
```

- [ ] **Step 2: Verify the cache function works manually**

```bash
python -c "
from processing.portfolio_metrics import _get_market_caps
from datetime import date
caps = _get_market_caps(['NVDA', 'TSM', 'MSFT'], date.today())
print(caps)
"
```

Expected: `{'NVDA': <float>, 'TSM': <float>, 'MSFT': <float>}` — three market caps in billions.

- [ ] **Step 3: Commit**

```bash
git add processing/portfolio_metrics.py
git commit -m "feat: portfolio_metrics — daily market cap cache via yfinance"
```

---

## Task 3: Peer Correlation

**Files:**
- Modify: `processing/portfolio_metrics.py` (add `_peer_correlation`)

- [ ] **Step 1: Add `_peer_correlation` to `processing/portfolio_metrics.py`**

Append after `_get_market_caps`:

```python
def _peer_correlation(
    df: pl.DataFrame,
    ohlcv_dir: Path,
    as_of: date,
    top_n: int = 10,
) -> pl.Series:
    """
    For each ticker, compute its average pairwise 90-day return correlation
    with the other top-(top_n-1) tickers by rank.

    Returns null for tickers ranked > top_n (they aren't basket members).
    High value (>0.65) means the pick is highly correlated with others already
    in the basket — flag for position sizing.
    """
    top_tickers = df.sort("rank").head(top_n)["ticker"].to_list()
    all_tickers = df["ticker"].to_list()
    window_start = as_of - timedelta(days=90)

    # Load 90-day daily returns for top-N tickers
    prices: dict[str, np.ndarray] = {}
    for ticker in top_tickers:
        ticker_parquets = list(ohlcv_dir.glob(f"{ticker}/*.parquet"))
        if not ticker_parquets:
            continue
        try:
            p = (
                pl.scan_parquet([str(x) for x in ticker_parquets])
                .filter(
                    (pl.col("date") >= window_start) & (pl.col("date") <= as_of)
                )
                .select(["date", "close_price"])
                .collect()
                .sort("date")
            )
            if len(p) > 5:
                returns = p["close_price"].pct_change().drop_nulls().to_numpy()
                if len(returns) > 5:
                    prices[ticker] = returns
        except Exception as exc:
            _LOG.debug("Failed to load OHLCV for %s: %s", ticker, exc)

    # Build pairwise correlation for each top-N ticker
    corr_map: dict[str, float | None] = {}
    for ticker in top_tickers:
        if ticker not in prices:
            corr_map[ticker] = None
            continue
        peers = [t for t in top_tickers if t != ticker and t in prices]
        if not peers:
            corr_map[ticker] = 0.0
            continue
        peer_corrs = []
        for peer in peers:
            min_len = min(len(prices[ticker]), len(prices[peer]))
            if min_len < 10:
                continue
            c = float(np.corrcoef(prices[ticker][-min_len:], prices[peer][-min_len:])[0, 1])
            if not np.isnan(c):
                peer_corrs.append(c)
        corr_map[ticker] = round(float(np.mean(peer_corrs)), 4) if peer_corrs else None

    # Tickers outside top-N get null
    return pl.Series(
        "peer_correlation_90d",
        [corr_map.get(t) for t in all_tickers],
        dtype=pl.Float64,
    )
```

- [ ] **Step 2: Run all portfolio metric tests**

```bash
pytest tests/test_portfolio_metrics.py -v
```

Expected: 5 PASSED

- [ ] **Step 3: Commit**

```bash
git add processing/portfolio_metrics.py
git commit -m "feat: portfolio_metrics — peer correlation for top-10 picks"
```

---

## Task 4: `enrich()` Entry Point + Wire Into Inference

**Files:**
- Modify: `processing/portfolio_metrics.py` (add `enrich()` + `__main__`)
- Modify: `models/inference.py` (call `enrich()` at end of `run_inference()`)

- [ ] **Step 1: Add `enrich()` and `__main__` to `processing/portfolio_metrics.py`**

Append to end of file:

```python
def enrich(date_str: str, predictions_dir: Path | None = None) -> pl.DataFrame:
    """
    Enrich predictions for a given date with portfolio metrics.

    Reads: data/predictions/date={date_str}/predictions.parquet
    Writes: data/predictions/date={date_str}/predictions_enriched.parquet
    Returns: enriched DataFrame
    """
    if predictions_dir is None:
        predictions_dir = Path("data/predictions")

    pred_path = predictions_dir / f"date={date_str}" / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions at {pred_path}. Run inference first.")

    df = pl.read_parquet(pred_path)
    as_of = date.fromisoformat(date_str)

    # 1. Market cap + liquidity
    caps = _get_market_caps(df["ticker"].to_list(), as_of)
    df = _apply_liquidity(df, caps)

    # 2. Model agreement
    df = df.with_columns(_model_agreement(df))

    # 3. Peer correlation (top-10 vs each other)
    ohlcv_dir = Path("data/raw/financials/ohlcv")
    df = df.with_columns(_peer_correlation(df, ohlcv_dir, as_of))

    # Write enriched output
    out_path = predictions_dir / f"date={date_str}" / "predictions_enriched.parquet"
    df.write_parquet(out_path, compression="snappy")

    # Print actionable summary
    liquid = df.filter(pl.col("is_liquid")).sort("rank")
    _LOG.info(
        "[PortfolioMetrics] %d liquid picks (≥$1B market cap) out of %d total",
        len(liquid), len(df),
    )
    print(f"\nLiquid picks for {date_str} (market cap ≥ $1B):\n")
    print(
        liquid.select([
            "rank", "ticker", "layer",
            "expected_annual_return", "market_cap_b",
            "model_agreement", "peer_correlation_90d",
        ])
    )
    illiquid = df.filter(~pl.col("is_liquid"))["ticker"].to_list()
    if illiquid:
        print(f"\nExcluded (< $1B): {', '.join(illiquid)}")

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    date_str = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    enrich(date_str)
```

- [ ] **Step 2: Wire into `models/inference.py`**

Find the end of `run_inference()` in `models/inference.py`. After the line that prints/logs inference results (the last `_LOG.info` or `print` statement before the return), add:

```python
    # Enrich predictions with portfolio metrics (liquidity, agreement, correlation)
    try:
        from processing.portfolio_metrics import enrich
        enrich(date_str)
    except Exception as exc:
        _LOG.warning("Portfolio metrics enrichment failed (non-fatal): %s", exc, exc_info=True)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_portfolio_metrics.py -v
```

Expected: 5 PASSED

- [ ] **Step 4: Run inference end-to-end to verify enrichment fires**

```bash
python -c "
from models.inference import run_inference
run_inference('2026-04-15')
" 2>&1 | grep -E "(PortfolioMetrics|Enriched|liquid|market_cap|ERROR)"
```

Expected: Output includes `[PortfolioMetrics] N liquid picks` line. Check that `data/predictions/date=2026-04-15/predictions_enriched.parquet` exists:

```bash
ls -lh data/predictions/date=2026-04-15/
```

Expected: Both `predictions.parquet` and `predictions_enriched.parquet` present.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -m "not integration" -q
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add processing/portfolio_metrics.py models/inference.py
git commit -m "feat: portfolio_metrics enrich() — liquidity filter, model agreement, peer correlation"
```

---

## Final Verification

```bash
# Full end-to-end: run inference, check enriched output
python -c "
from models.inference import run_inference
import polars as pl
run_inference('2026-04-15')
df = pl.read_parquet('data/predictions/date=2026-04-15/predictions_enriched.parquet')
print(df.sort('rank').select(['rank','ticker','market_cap_b','is_liquid','model_agreement','peer_correlation_90d']))
" 2>&1 | grep -v "UserWarning\|Sortedness\|join_asof"
```

Expected: Table with 80 rows, 4 new columns populated. Liquid picks clearly identifiable.
