# Options Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 ticker-specific options-derived features to all 141 tickers via a pluggable `OptionsSource` protocol, growing `FEATURE_COLS` from 55 → 61.

**Architecture:** Raw options chain contracts are ingested per-ticker via `yfinance` into `data/raw/options/date=YYYY-MM-DD/{ticker}.parquet`. A processing module reads all raw contracts, computes 6 features per `(ticker, date)`, and `join_options_features` left-joins them into the training/inference DataFrames with zero-fill on miss. The `OptionsSource` protocol allows drop-in replacement of the free yfinance source with paid sources (Tradier, CBOE) without changing the ingestor or processing code.

**Tech Stack:** Python 3.11+, Polars, yfinance, pytest, Parquet/snappy.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ingestion/options_ingestion.py` | Create | `OptionsSource` protocol, `YFinanceOptionsSource`, `ingest_options()`, `__main__` |
| `processing/options_features.py` | Create | `OPTIONS_FEATURE_COLS`, helpers, `build_options_features()`, `join_options_features()` |
| `models/train.py` | Modify | Import + append to `FEATURE_COLS`, add to `TIER_FEATURE_COLS["short"]`, add join call |
| `models/inference.py` | Modify | Import + add join call in `_build_feature_df` |
| `tests/test_options_ingestion.py` | Create | Unit tests for ingestion schema, empty chain, path correctness |
| `tests/test_options_features.py` | Create | Unit tests for all 6 features, zero-fill, join behavior |

---

### Task 1: Options Ingestion

**Files:**
- Create: `ingestion/options_ingestion.py`
- Create: `tests/test_options_ingestion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_options_ingestion.py
"""Tests for ingestion/options_ingestion.py — yfinance calls are mocked."""
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

_EXPECTED_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}


def _check_schema(df: pl.DataFrame) -> None:
    assert set(df.columns) == set(_EXPECTED_SCHEMA), f"Unexpected columns: {df.columns}"
    for col, dtype in _EXPECTED_SCHEMA.items():
        assert df[col].dtype == dtype, f"Column {col}: expected {dtype}, got {df[col].dtype}"


def _make_options_df(strikes: list[float], ivs: list[float]) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": ivs,
        "openInterest": [100] * len(strikes),
        "volume": [50] * len(strikes),
    })


def _mock_ticker(expiry_dates: list[str], strikes: list[float], ivs: list[float]):
    """Build a mock yf.Ticker that returns a known options chain."""
    chain = MagicMock()
    chain.calls = _make_options_df(strikes, ivs)
    chain.puts = _make_options_df(strikes, ivs)

    mock_t = MagicMock()
    mock_t.options = expiry_dates
    mock_t.option_chain.return_value = chain
    return mock_t


def test_yfinance_source_returns_correct_schema():
    """YFinanceOptionsSource.fetch returns all 8 schema columns with correct dtypes."""
    mock_t = _mock_ticker(
        expiry_dates=["2024-02-16"],
        strikes=[100.0, 105.0, 110.0],
        ivs=[0.30, 0.28, 0.32],
    )
    with patch("yfinance.Ticker", return_value=mock_t), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("NVDA", "2024-01-15")

    _check_schema(df)
    assert len(df) > 0
    assert set(df["option_type"].unique().to_list()) == {"call", "put"}
    assert df["ticker"][0] == "NVDA"
    assert df["date"][0] == datetime.date(2024, 1, 15)
    assert df["expiry"][0] == datetime.date(2024, 2, 16)


def test_yfinance_source_captures_all_strikes():
    """All strikes from the options chain are stored (ATM/OTM selection happens downstream)."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    mock_t = _mock_ticker(expiry_dates=["2024-02-16"], strikes=strikes, ivs=[0.3] * 5)
    with patch("yfinance.Ticker", return_value=mock_t), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("NVDA", "2024-01-15")

    stored_strikes = sorted(df["strike"].unique().to_list())
    assert stored_strikes == strikes, f"Expected all strikes stored, got: {stored_strikes}"


def test_yfinance_source_empty_chain_returns_empty():
    """Empty options chain returns empty DataFrame with correct schema — no crash."""
    mock_t = MagicMock()
    mock_t.options = []
    with patch("yfinance.Ticker", return_value=mock_t), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("DARK.L", "2024-01-15")

    _check_schema(df)
    assert len(df) == 0


def test_yfinance_source_exception_returns_empty():
    """If yfinance raises, return empty DataFrame — no crash."""
    mock_t = MagicMock()
    mock_t.options = property(lambda self: (_ for _ in ()).throw(Exception("network error")))
    with patch("yfinance.Ticker", side_effect=Exception("network error")), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("NVDA", "2024-01-15")

    _check_schema(df)
    assert len(df) == 0


def test_ingest_options_writes_parquet_at_correct_path(tmp_path):
    """ingest_options writes ticker.parquet under date=YYYY-MM-DD/ partition."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = pl.DataFrame(
        [{
            "ticker": "NVDA",
            "date": datetime.date(2024, 1, 15),
            "expiry": datetime.date(2024, 2, 16),
            "option_type": "call",
            "strike": 100.0,
            "iv": 0.30,
            "oi": 100,
            "volume": 50,
        }],
        schema={
            "ticker": pl.Utf8, "date": pl.Date, "expiry": pl.Date,
            "option_type": pl.Utf8, "strike": pl.Float64,
            "iv": pl.Float64, "oi": pl.Int64, "volume": pl.Int64,
        },
    )
    from ingestion.options_ingestion import ingest_options
    ingest_options(["NVDA"], "2024-01-15", tmp_path, source=mock_source)

    expected_path = tmp_path / "date=2024-01-15" / "NVDA.parquet"
    assert expected_path.exists(), f"Expected parquet at {expected_path}"
    result = pl.read_parquet(expected_path)
    assert result["ticker"][0] == "NVDA"
    assert result["date"][0] == datetime.date(2024, 1, 15)


def test_ingest_options_skips_empty_chain(tmp_path):
    """ingest_options does not write a file when the source returns empty data."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = pl.DataFrame(schema={
        "ticker": pl.Utf8, "date": pl.Date, "expiry": pl.Date,
        "option_type": pl.Utf8, "strike": pl.Float64,
        "iv": pl.Float64, "oi": pl.Int64, "volume": pl.Int64,
    })
    from ingestion.options_ingestion import ingest_options
    ingest_options(["DARK.L"], "2024-01-15", tmp_path, source=mock_source)

    assert not (tmp_path / "date=2024-01-15").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_options_ingestion.py -v
```

Expected: `ERROR` or `ModuleNotFoundError` — `ingestion/options_ingestion.py` does not exist yet.

- [ ] **Step 3: Implement `ingestion/options_ingestion.py`**

```python
"""
Ingest raw options chain data for all registered tickers via yfinance (or pluggable paid source).

Raw storage: data/raw/options/date=YYYY-MM-DD/{ticker}.parquet
Schema: (ticker, date, expiry, option_type, strike, iv, oi, volume)

Adding a paid source (Tradier, CBOE): implement OptionsSource.fetch() in a new class
and pass it to ingest_options(source=...).

Usage:
    python ingestion/options_ingestion.py               # fetch today
    python ingestion/options_ingestion.py --date 2024-01-15
"""
from __future__ import annotations

import argparse
import datetime
import logging
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

from ingestion.ticker_registry import TICKERS

_LOG = logging.getLogger(__name__)

_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


@runtime_checkable
class OptionsSource(Protocol):
    def fetch(self, ticker: str, date: str) -> pl.DataFrame:
        """Return raw options contracts for ticker on date.

        Returns DataFrame with schema matching _SCHEMA.
        Returns empty DataFrame if no options data is available.
        """
        ...


class YFinanceOptionsSource:
    """Fetch the full options chain via yfinance (free, no API key required)."""

    def fetch(self, ticker: str, date: str) -> pl.DataFrame:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            expiry_dates = t.options
        except Exception as exc:
            _LOG.debug("No options data for %s: %s", ticker, exc)
            return _empty()

        if not expiry_dates:
            return _empty()

        fetch_date = datetime.date.fromisoformat(date)
        rows: list[dict] = []

        for expiry_str in expiry_dates:
            try:
                expiry_date = datetime.date.fromisoformat(expiry_str)
                chain = t.option_chain(expiry_str)
                for opt_type, opts_df in [("call", chain.calls), ("put", chain.puts)]:
                    for _, row in opts_df.iterrows():
                        rows.append({
                            "ticker": ticker,
                            "date": fetch_date,
                            "expiry": expiry_date,
                            "option_type": opt_type,
                            "strike": float(row.get("strike", 0.0) or 0.0),
                            "iv": float(row.get("impliedVolatility", 0.0) or 0.0),
                            "oi": int(row.get("openInterest", 0) or 0),
                            "volume": int(row.get("volume", 0) or 0),
                        })
            except Exception as exc:
                _LOG.debug("Error fetching expiry %s for %s: %s", expiry_str, ticker, exc)
                continue

        if not rows:
            return _empty()

        return pl.DataFrame(rows, schema=_SCHEMA)


def ingest_options(
    tickers: list[str],
    date_str: str,
    output_dir: Path,
    source: OptionsSource | None = None,
) -> None:
    """Fetch options chain for all tickers and write Hive-partitioned parquet.

    Tickers with no options data (e.g. DARK.L) are silently skipped — no file written.
    time.sleep(0.5) is called between every ticker to respect yfinance rate limits.
    """
    if source is None:
        source = YFinanceOptionsSource()

    for ticker in tickers:
        df = source.fetch(ticker, date_str)
        if not df.is_empty():
            out_dir = output_dir / f"date={date_str}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_dir / f"{ticker}.parquet", compression="snappy")
            _LOG.info("Wrote %d contracts for %s on %s", len(df), ticker, date_str)
        else:
            _LOG.debug("No options data for %s on %s — skipping", ticker, date_str)
        time.sleep(0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest raw options chain data")
    parser.add_argument(
        "--date",
        default=str(datetime.date.today()),
        help="Date to fetch (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw/options")
    _LOG.info("Fetching options for %d tickers on %s", len(TICKERS), args.date)
    ingest_options(TICKERS, args.date, output_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_options_ingestion.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add ingestion/options_ingestion.py tests/test_options_ingestion.py
git commit -m "feat: add OptionsSource protocol and YFinanceOptionsSource ingestion"
```

---

### Task 2: Options Features Processing

**Files:**
- Create: `processing/options_features.py`
- Create: `tests/test_options_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_options_features.py
"""Tests for processing/options_features.py."""
import datetime
import math
from pathlib import Path

import polars as pl
import pytest

_RAW_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}


def _write_options(options_dir: Path, rows: list[dict]) -> None:
    """Write raw options rows to Hive-partitioned parquet."""
    df = pl.DataFrame(rows, schema=_RAW_SCHEMA)
    for date_val, date_group in df.group_by("date"):
        date_str = str(date_val[0]) if isinstance(date_val, tuple) else str(date_val)
        for ticker_val, ticker_group in date_group.group_by("ticker"):
            ticker = ticker_val[0] if isinstance(ticker_val, tuple) else ticker_val
            out = options_dir / f"date={date_str}" / f"{ticker}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            ticker_group.write_parquet(out, compression="snappy")


def _write_ohlcv(ohlcv_dir: Path, ticker: str, rows: list[dict]) -> None:
    """Write OHLCV rows to {ohlcv_dir}/{ticker}/all.parquet."""
    df = pl.DataFrame(
        rows,
        schema={"ticker": pl.Utf8, "date": pl.Date, "close_price": pl.Float64},
    )
    out = ohlcv_dir / ticker / "all.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="snappy")


def _near_term_expiry(as_of: datetime.date) -> datetime.date:
    """Return a date ~30 days from as_of."""
    return as_of + datetime.timedelta(days=30)


def _mid_term_expiry(as_of: datetime.date) -> datetime.date:
    """Return a date ~90 days from as_of."""
    return as_of + datetime.timedelta(days=90)


# ── OPTIONS_FEATURE_COLS ──────────────────────────────────────────────────────

def test_options_feature_cols_has_six_elements():
    """OPTIONS_FEATURE_COLS must have exactly 6 elements (spec requirement)."""
    from processing.options_features import OPTIONS_FEATURE_COLS
    assert len(OPTIONS_FEATURE_COLS) == 6
    expected = {
        "iv_rank_30d", "iv_hv_spread", "put_call_oi_ratio",
        "put_call_vol_ratio", "skew_otm", "iv_term_slope",
    }
    assert set(OPTIONS_FEATURE_COLS) == expected


# ── iv_rank_30d ───────────────────────────────────────────────────────────────

def test_iv_rank_in_range(tmp_path):
    """iv_rank_30d is clamped to [0, 100]."""
    as_of = datetime.date(2024, 1, 15)
    expiry = _near_term_expiry(as_of)

    # Write 35 days of history so we have enough for iv_rank
    rows = []
    for i in range(35):
        d = as_of - datetime.timedelta(days=34 - i)
        rows.extend([
            {"ticker": "NVDA", "date": d, "expiry": expiry,
             "option_type": "call", "strike": 100.0, "iv": 0.20 + i * 0.01, "oi": 100, "volume": 50},
            {"ticker": "NVDA", "date": d, "expiry": expiry,
             "option_type": "put", "strike": 100.0, "iv": 0.21 + i * 0.01, "oi": 100, "volume": 50},
        ])
    # Update near-term expiry for each date to be ~30 days out
    for row in rows:
        row["expiry"] = row["date"] + datetime.timedelta(days=30)

    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"
    _write_options(options_dir, rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    assert "iv_rank_30d" in result.columns
    iv_ranks = result["iv_rank_30d"].to_list()
    for rank in iv_ranks:
        assert 0.0 <= rank <= 100.0, f"iv_rank_30d out of range: {rank}"


def test_iv_rank_fallback_to_50_when_fewer_than_30_days(tmp_path):
    """iv_rank_30d falls back to 50.0 when fewer than 30 days of IV history exist."""
    as_of = datetime.date(2024, 1, 15)
    expiry = _near_term_expiry(as_of)

    # Only 5 days of history (< 30 required)
    rows = []
    for i in range(5):
        d = as_of - datetime.timedelta(days=4 - i)
        rows.extend([
            {"ticker": "NVDA", "date": d, "expiry": d + datetime.timedelta(days=30),
             "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 100, "volume": 50},
        ])

    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"
    _write_options(options_dir, rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    latest = result.sort("date").tail(1)
    assert latest["iv_rank_30d"][0] == pytest.approx(50.0), (
        f"Expected 50.0 fallback, got {latest['iv_rank_30d'][0]}"
    )


# ── iv_hv_spread ──────────────────────────────────────────────────────────────

def test_iv_hv_spread_positive_when_iv_exceeds_realized_vol(tmp_path):
    """iv_hv_spread > 0 when ATM IV > 30-day realized HV (vol premium scenario)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    # ATM IV = 0.40 (40%) — well above the realized vol we'll set via flat prices
    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.40, "oi": 500, "volume": 200},
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "put", "strike": 100.0, "iv": 0.41, "oi": 400, "volume": 150},
    ])

    # Flat OHLCV → log returns ≈ 0 → HV30 ≈ 0
    ohlcv_rows = [
        {"ticker": "NVDA", "date": as_of - datetime.timedelta(days=i), "close_price": 100.0}
        for i in range(35)
    ]
    _write_ohlcv(ohlcv_dir, "NVDA", ohlcv_rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    spread = result["iv_hv_spread"][0]
    assert spread > 0.0, f"Expected positive iv_hv_spread (IV > HV), got {spread}"


def test_iv_hv_spread_negative_when_hv_exceeds_iv(tmp_path):
    """iv_hv_spread < 0 when ATM IV < 30-day realized HV (vol discount scenario)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    # Very low ATM IV = 0.05 (5%)
    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.05, "oi": 500, "volume": 200},
    ])

    # Highly volatile OHLCV → HV30 well above 0.05
    import math as _math
    rng_prices = [100.0]
    for _ in range(34):
        rng_prices.append(rng_prices[-1] * _math.exp(0.05))  # 5% daily move → HV >> 0.05 annualized

    ohlcv_rows = [
        {"ticker": "NVDA", "date": as_of - datetime.timedelta(days=34 - i), "close_price": p}
        for i, p in enumerate(rng_prices)
    ]
    _write_ohlcv(ohlcv_dir, "NVDA", ohlcv_rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    spread = result["iv_hv_spread"][0]
    assert spread < 0.0, f"Expected negative iv_hv_spread (HV > IV), got {spread}"


# ── put_call ratios ───────────────────────────────────────────────────────────

def test_put_call_oi_ratio_correct(tmp_path):
    """put_call_oi_ratio = put_OI / call_OI for near-term expiry."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    _write_options(options_dir, [
        # call OI = 200, put OI = 400 → ratio = 2.0
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 100},
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "put", "strike": 100.0, "iv": 0.32, "oi": 400, "volume": 150},
    ])

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    assert result["put_call_oi_ratio"][0] == pytest.approx(2.0)


# ── skew_otm ─────────────────────────────────────────────────────────────────

def test_skew_otm_positive_when_put_iv_exceeds_call_iv(tmp_path):
    """skew_otm > 0 when OTM put IV > OTM call IV (bearish skew scenario)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"
    expiry = _near_term_expiry(as_of)
    spot = 100.0

    _write_options(options_dir, [
        # ATM
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "call", "strike": spot, "iv": 0.30, "oi": 300, "volume": 100},
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "put", "strike": spot, "iv": 0.31, "oi": 280, "volume": 90},
        # OTM put ~92% of spot
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "put", "strike": 92.0, "iv": 0.45, "oi": 200, "volume": 80},
        # OTM call ~108% of spot
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "call", "strike": 108.0, "iv": 0.25, "oi": 150, "volume": 60},
    ])

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    skew = result["skew_otm"][0]
    assert skew > 0.0, f"Expected positive skew (put IV 0.45 > call IV 0.25), got {skew}"


# ── iv_term_slope ─────────────────────────────────────────────────────────────

def test_iv_term_slope_positive_when_near_term_iv_exceeds_mid_term(tmp_path):
    """iv_term_slope > 0 when near-term IV > mid-term IV (inverted term structure)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    _write_options(options_dir, [
        # Near-term ATM IV = 0.50 (fear spike)
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.50, "oi": 300, "volume": 100},
        # Mid-term ATM IV = 0.30 (lower, normal)
        {"ticker": "NVDA", "date": as_of, "expiry": _mid_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 80},
    ])

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    slope = result["iv_term_slope"][0]
    assert slope > 0.0, f"Expected positive iv_term_slope (0.50 - 0.30 = 0.20), got {slope}"


# ── join_options_features ─────────────────────────────────────────────────────

def test_join_options_features_adds_all_six_columns(tmp_path):
    """join_options_features adds all 6 OPTIONS_FEATURE_COLS; row count unchanged."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 100},
    ])

    input_df = pl.DataFrame([
        {"ticker": "NVDA", "date": as_of, "return_1d": 0.01},
    ], schema={"ticker": pl.Utf8, "date": pl.Date, "return_1d": pl.Float64})

    from processing.options_features import join_options_features, OPTIONS_FEATURE_COLS
    result = join_options_features(input_df, options_dir, ohlcv_dir)

    assert len(result) == len(input_df), "Row count must not change"
    assert "return_1d" in result.columns, "Original columns must be preserved"
    for col in OPTIONS_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_join_options_features_zero_fills_missing_options_dir(tmp_path):
    """join_options_features zero-fills (not null) when options_dir does not exist."""
    input_df = pl.DataFrame([
        {"ticker": "NVDA", "date": datetime.date(2024, 1, 15), "return_1d": 0.01},
    ], schema={"ticker": pl.Utf8, "date": pl.Date, "return_1d": pl.Float64})

    nonexistent_dir = tmp_path / "options_missing"
    ohlcv_dir = tmp_path / "ohlcv"

    from processing.options_features import join_options_features, OPTIONS_FEATURE_COLS
    result = join_options_features(input_df, nonexistent_dir, ohlcv_dir)

    for col in OPTIONS_FEATURE_COLS:
        assert col in result.columns
        assert result[col][0] == pytest.approx(0.0), f"{col} should be 0.0 (zero-filled), got {result[col][0]}"
        assert result[col][0] is not None, f"{col} should be 0.0 not null"


def test_join_options_features_zero_fills_ticker_absent_from_options(tmp_path):
    """Ticker present in df but absent from options data → zero-filled (not null)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    # Write options for NVDA only
    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 100},
    ])

    # df includes MSFT which has no options data
    input_df = pl.DataFrame([
        {"ticker": "NVDA", "date": as_of, "return_1d": 0.01},
        {"ticker": "MSFT", "date": as_of, "return_1d": 0.02},  # no options file
    ], schema={"ticker": pl.Utf8, "date": pl.Date, "return_1d": pl.Float64})

    from processing.options_features import join_options_features, OPTIONS_FEATURE_COLS
    result = join_options_features(input_df, options_dir, ohlcv_dir)

    assert len(result) == 2
    msft_row = result.filter(pl.col("ticker") == "MSFT")
    for col in OPTIONS_FEATURE_COLS:
        val = msft_row[col][0]
        assert val == pytest.approx(0.0), f"MSFT {col} should be zero-filled, got {val}"
        assert val is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_options_features.py -v
```

Expected: `ERROR` or `ModuleNotFoundError` — `processing/options_features.py` does not exist yet.

- [ ] **Step 3: Implement `processing/options_features.py`**

```python
"""
Compute 6 ticker-specific options-derived features from raw options chain data.

OPTIONS_FEATURE_COLS:
    iv_rank_30d         — ATM IV percentile vs. 52-week high/low, scaled 0–100
    iv_hv_spread        — Near-term ATM IV minus 30-day realized historical vol (HV30)
    put_call_oi_ratio   — Put OI / Call OI for near-term expiry (≤45 DTE, closest to 30)
    put_call_vol_ratio  — Put volume / Call volume for near-term expiry
    skew_otm            — OTM put IV minus OTM call IV (~5–10% moneyness)
    iv_term_slope       — 30d ATM IV minus 90d ATM IV (positive = inverted = fear)
"""
from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

OPTIONS_FEATURE_COLS: list[str] = [
    "iv_rank_30d",
    "iv_hv_spread",
    "put_call_oi_ratio",
    "put_call_vol_ratio",
    "skew_otm",
    "iv_term_slope",
]

_RAW_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}

_FEATURE_SCHEMA = {"ticker": pl.Utf8, "date": pl.Date, **{c: pl.Float64 for c in OPTIONS_FEATURE_COLS}}


def _empty_features() -> pl.DataFrame:
    return pl.DataFrame(schema=_FEATURE_SCHEMA)


def _select_expiry(
    contracts: pl.DataFrame,
    as_of_date: datetime.date,
    target_dte: int,
    max_dte: int,
) -> pl.DataFrame:
    """Return contracts for the expiry closest to target_dte (DTE in (0, max_dte])."""
    expiries = sorted(contracts["expiry"].unique().to_list())
    candidates = [e for e in expiries if 0 < (e - as_of_date).days <= max_dte]
    if not candidates:
        return pl.DataFrame(schema=_RAW_SCHEMA)
    best = min(candidates, key=lambda e: abs((e - as_of_date).days - target_dte))
    return contracts.filter(pl.col("expiry") == best)


def _atm_strike(contracts: pl.DataFrame, spot: float) -> float | None:
    """Return the strike closest to spot price."""
    strikes = contracts["strike"].unique().to_list()
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s - spot))


def _otm_put_strike(contracts: pl.DataFrame, spot: float) -> float | None:
    """Return the OTM put strike closest to 92.5% of spot (searching 88–97% band)."""
    target = spot * 0.925
    strikes = contracts.filter(pl.col("option_type") == "put")["strike"].unique().to_list()
    candidates = [s for s in strikes if 0.88 * spot <= s <= 0.97 * spot]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s - target))


def _otm_call_strike(contracts: pl.DataFrame, spot: float) -> float | None:
    """Return the OTM call strike closest to 107.5% of spot (searching 103–112% band)."""
    target = spot * 1.075
    strikes = contracts.filter(pl.col("option_type") == "call")["strike"].unique().to_list()
    candidates = [s for s in strikes if 1.03 * spot <= s <= 1.12 * spot]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s - target))


def _get_atm_iv(contracts: pl.DataFrame, atm: float) -> float:
    """Return ATM IV (prefer call; fall back to any contract at that strike)."""
    atm_rows = contracts.filter(pl.col("strike") == atm)
    call_rows = atm_rows.filter(pl.col("option_type") == "call")
    iv_series = call_rows["iv"] if not call_rows.is_empty() else atm_rows["iv"]
    vals = [v for v in iv_series.to_list() if v and v > 0]
    return vals[0] if vals else 0.0


def _compute_hv30(ohlcv_dir: Path, ticker: str, as_of_date: datetime.date) -> float:
    """Compute 30-day realized historical volatility (annualized) from close_price."""
    ticker_dir = ohlcv_dir / ticker
    if not ticker_dir.exists():
        return float("nan")
    files = sorted(ticker_dir.glob("*.parquet"))
    if not files:
        return float("nan")

    df = (
        pl.concat([pl.read_parquet(f) for f in files])
        .filter(pl.col("date") <= pl.lit(as_of_date))
        .sort("date")
        .tail(35)
        .select(["date", "close_price"])
    )
    if len(df) < 2:
        return float("nan")

    closes = df["close_price"].to_list()
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] and closes[i - 1] > 0
    ]
    if len(log_returns) < 2:
        return float("nan")

    log_returns = log_returns[-30:]
    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    return math.sqrt(variance) * math.sqrt(252)


def _compute_row(
    ticker: str,
    as_of_date: datetime.date,
    contracts: pl.DataFrame,
    atm_iv_history: list[float],
    ohlcv_dir: Path,
) -> dict:
    """Compute all 6 options features for a single (ticker, date).

    atm_iv_history: list of ATM IV values for this ticker for all dates ≤ as_of_date,
                    sorted oldest → newest. Precomputed by build_options_features.
    """
    zero_row: dict = {"ticker": ticker, "date": as_of_date, **{c: 0.0 for c in OPTIONS_FEATURE_COLS}}

    near_term = _select_expiry(contracts, as_of_date, target_dte=30, max_dte=45)
    if near_term.is_empty():
        return zero_row

    # Spot price proxy: median of near-term strikes
    spot_val = near_term["strike"].median()
    if spot_val is None:
        return zero_row
    spot = float(spot_val)
    if spot <= 0:
        return zero_row

    # put_call_oi_ratio
    put_oi = int(near_term.filter(pl.col("option_type") == "put")["oi"].sum() or 0)
    call_oi = int(near_term.filter(pl.col("option_type") == "call")["oi"].sum() or 0)
    put_call_oi = float(put_oi / call_oi) if call_oi > 0 else 0.0

    # put_call_vol_ratio
    put_vol = int(near_term.filter(pl.col("option_type") == "put")["volume"].sum() or 0)
    call_vol = int(near_term.filter(pl.col("option_type") == "call")["volume"].sum() or 0)
    put_call_vol = float(put_vol / call_vol) if call_vol > 0 else 0.0

    # ATM IV (near-term, call-preferred)
    atm = _atm_strike(near_term, spot)
    atm_iv_near = _get_atm_iv(near_term, atm) if atm is not None else 0.0

    # skew_otm: OTM put IV minus OTM call IV
    skew = 0.0
    otm_put = _otm_put_strike(near_term, spot)
    otm_call = _otm_call_strike(near_term, spot)
    if otm_put is not None and otm_call is not None:
        put_iv_vals = [
            v for v in near_term.filter(
                (pl.col("option_type") == "put") & (pl.col("strike") == otm_put)
            )["iv"].to_list() if v and v > 0
        ]
        call_iv_vals = [
            v for v in near_term.filter(
                (pl.col("option_type") == "call") & (pl.col("strike") == otm_call)
            )["iv"].to_list() if v and v > 0
        ]
        put_otm_iv = put_iv_vals[0] if put_iv_vals else 0.0
        call_otm_iv = call_iv_vals[0] if call_iv_vals else 0.0
        skew = put_otm_iv - call_otm_iv

    # iv_term_slope: 30d ATM IV minus 90d ATM IV
    iv_term = 0.0
    mid_term = _select_expiry(contracts, as_of_date, target_dte=90, max_dte=180)
    if not mid_term.is_empty():
        mid_atm = _atm_strike(mid_term, spot)
        if mid_atm is not None:
            atm_iv_mid = _get_atm_iv(mid_term, mid_atm)
            iv_term = atm_iv_near - atm_iv_mid

    # iv_hv_spread: ATM IV minus 30-day realized HV
    hv30 = _compute_hv30(ohlcv_dir, ticker, as_of_date)
    iv_hv = (atm_iv_near - hv30) if not math.isnan(hv30) else 0.0

    # iv_rank_30d: ATM IV percentile over rolling 52-week window
    if len(atm_iv_history) < 30:
        iv_rank = 50.0  # neutral fallback — insufficient history
    else:
        window = atm_iv_history[-252:]
        min_iv = min(window)
        max_iv = max(window)
        if max_iv <= min_iv:
            iv_rank = 50.0
        else:
            iv_rank = max(0.0, min(100.0, (atm_iv_near - min_iv) / (max_iv - min_iv) * 100.0))

    return {
        "ticker": ticker,
        "date": as_of_date,
        "iv_rank_30d": iv_rank,
        "iv_hv_spread": iv_hv,
        "put_call_oi_ratio": put_call_oi,
        "put_call_vol_ratio": put_call_vol,
        "skew_otm": skew,
        "iv_term_slope": iv_term,
    }


def build_options_features(options_dir: Path, ohlcv_dir: Path) -> pl.DataFrame:
    """Aggregate raw options contracts into (ticker, date) feature rows.

    Reads all options parquets once, then processes per (ticker, date).
    iv_rank_30d requires historical ATM IV; pre-built per ticker before the main loop.
    """
    if not options_dir.exists():
        return _empty_features()

    all_files = sorted(options_dir.glob("date=*/*.parquet"))
    if not all_files:
        return _empty_features()

    # Pre-build ATM IV history per ticker (sorted by date, oldest → newest)
    # This avoids O(n²) re-reads inside _compute_row.
    atm_iv_by_ticker: dict[str, list[float]] = {}  # ticker → [iv_oldest, ..., iv_newest]
    atm_iv_dates: dict[str, list[datetime.date]] = {}  # ticker → parallel dates list

    for date_dir in sorted(options_dir.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        try:
            dir_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        for ticker_file in sorted(date_dir.glob("*.parquet")):
            ticker = ticker_file.stem
            contracts = pl.read_parquet(ticker_file)
            if contracts.is_empty():
                continue
            near_term = _select_expiry(contracts, dir_date, target_dte=30, max_dte=45)
            if near_term.is_empty():
                continue
            spot_val = near_term["strike"].median()
            if spot_val is None:
                continue
            spot = float(spot_val)
            if spot <= 0:
                continue
            atm = _atm_strike(near_term, spot)
            if atm is None:
                continue
            iv = _get_atm_iv(near_term, atm)
            if iv > 0:
                atm_iv_by_ticker.setdefault(ticker, []).append(iv)
                atm_iv_dates.setdefault(ticker, []).append(dir_date)

    # Main loop: compute all 6 features per (ticker, date)
    rows: list[dict] = []
    for date_dir in sorted(options_dir.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        try:
            as_of_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        for ticker_file in sorted(date_dir.glob("*.parquet")):
            ticker = ticker_file.stem
            contracts = pl.read_parquet(ticker_file)
            if contracts.is_empty():
                continue
            # Slice ATM IV history up to (and including) as_of_date
            all_dates = atm_iv_dates.get(ticker, [])
            all_ivs = atm_iv_by_ticker.get(ticker, [])
            history = [iv for d, iv in zip(all_dates, all_ivs) if d <= as_of_date]
            rows.append(_compute_row(ticker, as_of_date, contracts, history, ohlcv_dir))

    if not rows:
        return _empty_features()

    return pl.DataFrame(rows, schema=_FEATURE_SCHEMA)


def join_options_features(df: pl.DataFrame, options_dir: Path, ohlcv_dir: Path) -> pl.DataFrame:
    """Left-join options features to df by (ticker, date). Missing rows zero-fill (not null)."""
    options_df = build_options_features(options_dir, ohlcv_dir)
    result = df.join(options_df, on=["ticker", "date"], how="left")
    fill_exprs = [pl.col(col).fill_null(0.0) for col in OPTIONS_FEATURE_COLS]
    return result.with_columns(fill_exprs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_options_features.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add processing/options_features.py tests/test_options_features.py
git commit -m "feat: add options_features processing module with 6 OPTIONS_FEATURE_COLS"
```

---

### Task 3: Model Integration

**Files:**
- Modify: `models/train.py` (imports at line 41, FEATURE_COLS at line 106, TIER_FEATURE_COLS at line 131, build_training_dataset at line 320)
- Modify: `models/inference.py` (imports at line 44, `_build_feature_df` at line 120)
- Test: `tests/test_train.py` (add assertions — do NOT rewrite; append to existing file)

- [ ] **Step 1: Write the failing tests**

Add the following tests to the **end** of `tests/test_train.py` (do not modify existing tests):

```python
# ── Options signals integration ────────────────────────────────────────────────

def test_feature_cols_has_61_elements():
    """FEATURE_COLS must have exactly 61 elements after adding OPTIONS_FEATURE_COLS."""
    from models.train import FEATURE_COLS
    assert len(FEATURE_COLS) == 61, f"Expected 61 features, got {len(FEATURE_COLS)}"


def test_options_feature_cols_in_feature_cols():
    """All 6 OPTIONS_FEATURE_COLS must appear in FEATURE_COLS."""
    from models.train import FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    for col in OPTIONS_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_options_feature_cols_in_short_tier():
    """All 6 OPTIONS_FEATURE_COLS must be in TIER_FEATURE_COLS['short']."""
    from models.train import TIER_FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    short_cols = TIER_FEATURE_COLS["short"]
    for col in OPTIONS_FEATURE_COLS:
        assert col in short_cols, f"{col} missing from TIER_FEATURE_COLS['short']"


def test_options_feature_cols_in_medium_tier():
    """All 6 OPTIONS_FEATURE_COLS must be in TIER_FEATURE_COLS['medium'] (inherits FEATURE_COLS)."""
    from models.train import TIER_FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    medium_cols = TIER_FEATURE_COLS["medium"]
    for col in OPTIONS_FEATURE_COLS:
        assert col in medium_cols, f"{col} missing from TIER_FEATURE_COLS['medium']"


def test_options_feature_cols_not_in_long_tier():
    """OPTIONS_FEATURE_COLS must NOT appear in TIER_FEATURE_COLS['long'] (noise at year+ horizons)."""
    from models.train import TIER_FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    long_cols = set(TIER_FEATURE_COLS["long"])
    for col in OPTIONS_FEATURE_COLS:
        assert col not in long_cols, f"{col} must not be in TIER_FEATURE_COLS['long']"


def test_medium_tier_is_copy_of_feature_cols():
    """TIER_FEATURE_COLS['medium'] must be a separate list object from FEATURE_COLS."""
    from models.train import FEATURE_COLS, TIER_FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] is not FEATURE_COLS, (
        "medium tier must be list(FEATURE_COLS), not FEATURE_COLS itself"
    )
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_train.py::test_feature_cols_has_61_elements \
       tests/test_train.py::test_options_feature_cols_in_feature_cols \
       tests/test_train.py::test_options_feature_cols_in_short_tier \
       tests/test_train.py::test_options_feature_cols_not_in_long_tier -v
```

Expected: FAILED — FEATURE_COLS has 55 elements, OPTIONS_FEATURE_COLS not imported.

- [ ] **Step 3: Update `models/train.py`**

**3a. Add import** — add after the existing cyber threat import (line 41):

```python
from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS, join_cyber_threat_features
from processing.options_features import OPTIONS_FEATURE_COLS, join_options_features
```

**3b. Update `FEATURE_COLS`** — change the existing block (lines 106–113):

```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS + ENERGY_FEATURE_COLS
    + SUPPLY_CHAIN_FEATURE_COLS + FX_FEATURE_COLS
    + CYBER_THREAT_FEATURE_COLS  # 48 → 55 features total
    + OPTIONS_FEATURE_COLS       # 55 → 61 features total
)
```

**3c. Update `TIER_FEATURE_COLS["short"]`** — add `+ OPTIONS_FEATURE_COLS` at the end of the short tier (currently ends at `_CYBER_THREAT_SHORT_COLS`):

```python
TIER_FEATURE_COLS: dict[str, list[str]] = {
    "short": (
        PRICE_FEATURE_COLS
        + SENTIMENT_FEATURE_COLS
        + INSIDER_FEATURE_COLS
        + SHORT_INTEREST_FEATURE_COLS
        + _CYBER_THREAT_SHORT_COLS   # 5 features: *_7d only
        + OPTIONS_FEATURE_COLS       # all 6 options features
    ),
    "medium": list(FEATURE_COLS),    # all 61 features (copy to avoid shared mutable reference)
    "long": (
        PRICE_FEATURE_COLS
        + FUND_FEATURE_COLS
        + EARNINGS_FEATURE_COLS
        + GRAPH_FEATURE_COLS
        + OWNERSHIP_FEATURE_COLS
        + ENERGY_FEATURE_COLS
        + SUPPLY_CHAIN_FEATURE_COLS
        + FX_FEATURE_COLS
        # cyber threat and options features excluded — noise at year+ horizons
    ),
}
```

**3d. Add join call in `build_training_dataset`** — add after the cyber threat join (line 322):

```python
    # Join cyber threat regime features (date-keyed market-wide signals)
    cyber_threat_dir = fundamentals_dir.parent.parent / "cyber_threat"
    df = join_cyber_threat_features(df, cyber_threat_dir)

    # Join options-derived features (ticker-specific, joined by (ticker, date))
    options_dir = fundamentals_dir.parent.parent / "options"
    df = join_options_features(df, options_dir, ohlcv_dir)
```

- [ ] **Step 4: Update `models/inference.py`**

**4a. Add import** — add after the existing cyber threat import (line 44):

```python
from processing.cyber_threat_features import join_cyber_threat_features
from processing.options_features import join_options_features
```

**4b. Add join call in `_build_feature_df`** — add after the cyber threat join (line 121), and update the docstring:

```python
def _build_feature_df(
    date_str: str,
    data_dir: Path,
) -> pl.DataFrame:
    """Build the 61-feature DataFrame for all tickers on date_str."""
    ohlcv_dir        = data_dir / "financials" / "ohlcv"
    ...
    cyber_threat_dir = data_dir / "cyber_threat"
    df = join_cyber_threat_features(df, cyber_threat_dir)

    options_dir = data_dir / "options"
    df = join_options_features(df, options_dir, ohlcv_dir)

    return df
```

- [ ] **Step 5: Run all new tests to verify they pass**

```bash
pytest tests/test_train.py::test_feature_cols_has_61_elements \
       tests/test_train.py::test_options_feature_cols_in_feature_cols \
       tests/test_train.py::test_options_feature_cols_in_short_tier \
       tests/test_train.py::test_options_feature_cols_in_medium_tier \
       tests/test_train.py::test_options_feature_cols_not_in_long_tier \
       tests/test_train.py::test_medium_tier_is_copy_of_feature_cols -v
```

Expected: 6 PASSED.

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all tests pass. Note: `test_train.py` integration tests that build a full dataset will still pass because `join_options_features` zero-fills gracefully when `options_dir` does not exist.

- [ ] **Step 7: Commit**

```bash
git add models/train.py models/inference.py tests/test_train.py
git commit -m "feat: wire options signals into FEATURE_COLS and tier routing (55 → 61 features)"
```

---

## Success Criteria Verification

After all 3 tasks are complete, verify against the spec:

```bash
python - <<'EOF'
from models.train import FEATURE_COLS, TIER_FEATURE_COLS
from processing.options_features import OPTIONS_FEATURE_COLS

assert len(OPTIONS_FEATURE_COLS) == 6, f"Expected 6, got {len(OPTIONS_FEATURE_COLS)}"
assert len(FEATURE_COLS) == 61, f"Expected 61, got {len(FEATURE_COLS)}"
assert all(0 <= v <= 100 for v in [0.0, 50.0, 100.0]), "iv_rank_30d range ok"
assert not any(c in TIER_FEATURE_COLS["long"] for c in OPTIONS_FEATURE_COLS), "long tier clean"
assert TIER_FEATURE_COLS["medium"] is not FEATURE_COLS, "medium tier is copy"
print("All spec success criteria verified.")
EOF
```
