# Data Pipeline Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two silent data bugs (sentiment 0 non-null, 13F filer ranking) and add operational tooling (health check script, refresh script) to keep the pipeline healthy.

**Architecture:** Four targeted changes — two `join_asof` tolerance fixes in existing processing modules, one ranking function rewrite in the ingestion module, and two new tool scripts. No new dependencies.

**Tech Stack:** Python 3.11, Polars, zsh

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `processing/sentiment_features.py` | Modify | Add `tolerance=timedelta(days=30)` to `join_asof` |
| `processing/short_interest_features.py` | Modify | Add `tolerance=timedelta(days=7)` to `join_asof` |
| `ingestion/sec_13f_ingestion.py` | Modify | Replace CIK-age ranking with prior-quarter AUM ranking |
| `tools/pipeline_health.py` | Create | Prints freshness table for all data sources |
| `tools/run_refresh.sh` | Create | Ordered shell script to refresh the full pipeline |
| `tests/test_sentiment_join_tolerance.py` | Create | 3 tests for the tolerance fix |
| `tests/test_13f_ranking.py` | Create | 3 tests for AUM-based ranking |

---

## Task 1: Fix Sentiment Join Tolerance

**Files:**
- Modify: `processing/sentiment_features.py:278-284`
- Create: `tests/test_sentiment_join_tolerance.py`

The bug: `join_asof` with `strategy="backward"` and no `tolerance` matches arbitrarily far back. Sentiment features only exist from ~1 year ago, so all training rows older than that match the oldest available sentiment date — wrong. Fix: add `tolerance=timedelta(days=30)` so rows more than 30 days past the last sentiment observation get null.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sentiment_join_tolerance.py
"""Tests that join_sentiment_features respects the 30-day tolerance window."""
from __future__ import annotations
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from processing.sentiment_features import join_sentiment_features


def _make_sentiment_dir(tmp_path: Path, ticker: str, feature_date: date) -> Path:
    """Write a minimal sentiment features parquet for one ticker at one date."""
    features_dir = tmp_path / "sentiment_features"
    ticker_dir = features_dir / ticker
    ticker_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "ticker": [ticker],
        "date": [feature_date],
        "sentiment_mean_7d": [0.5],
        "sentiment_std_7d": [0.1],
        "article_count_7d": [3],
        "sentiment_momentum_14d": [0.2],
        "ticker_vs_market_7d": [0.1],
    })
    df.write_parquet(ticker_dir / "daily.parquet")
    return features_dir


def _make_spine(ticker: str, spine_date: date) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [ticker],
        "date": [spine_date],
    })


def test_sentiment_join_propagates_within_30d(tmp_path):
    """A spine row 15 days after sentiment data should get the sentiment value."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=15)  # within 30-day window

    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)

    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] == pytest.approx(0.5), \
        "Should propagate sentiment within 30-day tolerance"


def test_sentiment_join_null_beyond_30d(tmp_path):
    """A spine row 31 days after sentiment data should get null."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=31)  # beyond 30-day window

    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)

    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] is None, \
        "Should return null when spine row is >30 days past last sentiment"


def test_sentiment_join_null_before_data_exists(tmp_path):
    """A spine row from 3 years before sentiment data should get null."""
    feature_date = date(2025, 1, 1)
    spine_date = date(2022, 1, 1)  # 3 years before sentiment data

    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)

    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] is None, \
        "Should return null for rows far before sentiment data exists"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /path/to/project
pytest tests/test_sentiment_join_tolerance.py -v
```

Expected: 2 of 3 FAIL — `test_sentiment_join_null_beyond_30d` and `test_sentiment_join_null_before_data_exists` will fail because the current code propagates sentiment arbitrarily far back.

- [ ] **Step 3: Add the tolerance import and fix the join**

In `processing/sentiment_features.py`, the `join_sentiment_features` function is at line 248. Make two changes:

First, add `timedelta` to the import at the top of the file. The file already imports `from datetime import date` — change it to:

```python
from datetime import date, timedelta
```

Second, in `join_sentiment_features()`, replace the `join_asof` call (currently lines 278–284):

```python
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="feature_date",
        by="ticker",
        strategy="backward",
    )
```

with:

```python
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="feature_date",
        by="ticker",
        strategy="backward",
        tolerance=timedelta(days=30),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sentiment_join_tolerance.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add processing/sentiment_features.py tests/test_sentiment_join_tolerance.py
git commit -m "fix: add 30-day tolerance to sentiment join_asof — prevents stale propagation"
```

---

## Task 2: Fix Short Interest Join Tolerance

**Files:**
- Modify: `processing/short_interest_features.py:214-220`

Short interest data is daily but if data is missing for a ticker on some dates, the backward join would propagate a stale value indefinitely. Limit to 7 days (short interest changes rapidly; a week-old value is still meaningful, older is not).

- [ ] **Step 1: Add the tolerance fix**

In `processing/short_interest_features.py`, find the `join_asof` call (around line 214). The file already imports `from datetime import date` — change it to `from datetime import date, timedelta`.

Replace the `join_asof` call:

```python
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="si_date",
        by="ticker",
        strategy="backward",
    )
```

with:

```python
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="si_date",
        by="ticker",
        strategy="backward",
        tolerance=timedelta(days=7),
    )
```

- [ ] **Step 2: Run full test suite to confirm no regressions**

```bash
pytest tests/ -m "not integration" -q
```

Expected: All tests pass (170+).

- [ ] **Step 3: Commit**

```bash
git add processing/short_interest_features.py
git commit -m "fix: add 7-day tolerance to short interest join_asof"
```

---

## Task 3: Fix 13F Filer Ranking (AUM-Based)

**Files:**
- Modify: `ingestion/sec_13f_ingestion.py:161-176` (replace `rank_filers_by_position_count`)
- Modify: `ingestion/sec_13f_ingestion.py:333` (update call site in `ingest_quarter`)
- Create: `tests/test_13f_ranking.py`

The current ranking sorts by CIK integer (lower = older registration). Replace with: use prior-quarter's downloaded parquets to rank by total portfolio value. Falls back to CIK-age for the first quarter where no prior data exists.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_13f_ranking.py
"""Tests for AUM-based 13F filer ranking."""
from __future__ import annotations
import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ingestion.sec_13f_ingestion import rank_filers_by_position_count, _prior_quarter


def _write_prior_quarter_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write mock prior-quarter holdings parquet files."""
    quarter_dir = tmp_path / "2024Q4"
    quarter_dir.mkdir(parents=True)
    for row in rows:
        cik = row["cik"]
        df = pl.DataFrame({
            "cik": [cik],
            "quarter": ["2024Q4"],
            "period_end": [None],
            "cusip": ["123456789"],
            "ticker": ["NVDA"],
            "shares_held": [row["shares"]],
            "value_usd_thousands": [row["value"]],
        })
        df.write_parquet(quarter_dir / f"{cik}.parquet")
    return tmp_path


def _make_index_df(ciks: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "cik": ciks,
        "date_filed": ["2025-02-14"] * len(ciks),
        "filename": [f"edgar/data/{c}/0001.txt" for c in ciks],
    })


def test_rank_uses_prior_quarter_aum(tmp_path):
    """Filers ranked by prior-quarter total position value descending."""
    # CIK 0000000003 has highest value — should rank first despite highest CIK int
    prior_rows = [
        {"cik": "0000000001", "shares": 100, "value": 1000},
        {"cik": "0000000002", "shares": 500, "value": 5000},
        {"cik": "0000000003", "shares": 900, "value": 9000},
    ]
    prior_dir = _write_prior_quarter_parquet(tmp_path, prior_rows)
    index_df = _make_index_df(["0000000001", "0000000002", "0000000003"])

    result = rank_filers_by_position_count(index_df, top_n=3, prior_quarter_dir=prior_dir / "2024Q4")

    assert result[0] == "0000000003", "Highest-value CIK should rank first"
    assert result[1] == "0000000002"
    assert result[2] == "0000000001"


def test_rank_falls_back_to_cik_age(tmp_path):
    """When prior_quarter_dir is None, falls back to CIK-integer sort (ascending)."""
    index_df = _make_index_df(["0000000300", "0000000100", "0000000200"])
    result = rank_filers_by_position_count(index_df, top_n=3, prior_quarter_dir=None)

    assert result == ["0000000100", "0000000200", "0000000300"], \
        "Fallback should sort CIK ascending (lower = older = larger institution)"


def test_rank_appends_new_ciks_not_in_prior(tmp_path):
    """CIKs in index but absent from prior quarter are appended at end."""
    prior_rows = [
        {"cik": "0000000001", "shares": 100, "value": 1000},
    ]
    prior_dir = _write_prior_quarter_parquet(tmp_path, prior_rows)
    # index has CIK 001 (in prior) + CIK 002 (new, not in prior)
    index_df = _make_index_df(["0000000001", "0000000002"])

    result = rank_filers_by_position_count(index_df, top_n=2, prior_quarter_dir=prior_dir / "2024Q4")

    assert result[0] == "0000000001", "Known filer should rank first"
    assert result[1] == "0000000002", "New filer appended at end"


def test_prior_quarter_helper():
    """_prior_quarter returns correct (year, quarter) tuple."""
    assert _prior_quarter(2025, 1) == (2024, 4)
    assert _prior_quarter(2025, 2) == (2025, 1)
    assert _prior_quarter(2025, 4) == (2025, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_13f_ranking.py -v
```

Expected: All 4 FAIL (`rank_filers_by_position_count` doesn't accept `prior_quarter_dir`, `_prior_quarter` doesn't exist).

- [ ] **Step 3: Add `_prior_quarter` helper and rewrite `rank_filers_by_position_count`**

In `ingestion/sec_13f_ingestion.py`, replace the existing `rank_filers_by_position_count` function (lines 161–176) with:

```python
def _prior_quarter(year: int, quarter: int) -> tuple[int, int]:
    """Return (year, quarter) for the quarter immediately before the given one."""
    if quarter == 1:
        return (year - 1, 4)
    return (year, quarter - 1)


def rank_filers_by_position_count(
    index_df: pl.DataFrame,
    top_n: int = 500,
    prior_quarter_dir: Path | None = None,
) -> list[str]:
    """
    Return the top_n filer CIKs, ranked by total prior-quarter portfolio value.

    If prior_quarter_dir is provided and contains per-filer parquets, ranks by
    sum(value_usd_thousands) per CIK descending. CIKs absent from prior quarter
    (new filers) are appended at the end sorted by CIK integer.

    Falls back to CIK-integer ascending sort (older registration = proxy for
    larger institution) when no prior quarter data is available.
    """
    index_ciks: set[str] = set(index_df["cik"].to_list())

    if prior_quarter_dir is not None and prior_quarter_dir.exists():
        parquets = list(prior_quarter_dir.glob("*.parquet"))
        if parquets:
            prior = pl.concat([pl.read_parquet(p) for p in parquets])
            aum_rank = (
                prior.group_by("cik")
                .agg(pl.col("value_usd_thousands").sum().alias("total_value"))
                .sort("total_value", descending=True)
            )
            ranked_ciks = [c for c in aum_rank["cik"].to_list() if c in index_ciks]
            seen = set(ranked_ciks)
            # Append new filers not in prior quarter, sorted by CIK age
            remaining = sorted(
                [c for c in index_ciks if c not in seen],
                key=lambda x: int(x),
            )
            return (ranked_ciks + remaining)[:top_n]

    # Fallback: CIK-age sort (lower CIK = older EDGAR registrant = typically larger)
    sorted_df = index_df.with_columns(
        pl.col("cik").cast(pl.UInt64).alias("cik_int")
    ).sort("cik_int")
    return sorted_df["cik"].head(top_n).to_list()
```

- [ ] **Step 4: Update the call site in `ingest_quarter`**

Find the line in `ingest_quarter()` (around line 333):

```python
    top_ciks = rank_filers_by_position_count(index_df, top_n=top_n)
```

Replace with:

```python
    prior_year, prior_qtr = _prior_quarter(year, quarter)
    prior_quarter_dir = output_dir / f"{prior_year}Q{prior_qtr}"
    top_ciks = rank_filers_by_position_count(
        index_df, top_n=top_n, prior_quarter_dir=prior_quarter_dir
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_13f_ranking.py -v
```

Expected: 4 PASSED

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -m "not integration" -q
```

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add ingestion/sec_13f_ingestion.py tests/test_13f_ranking.py
git commit -m "feat: rank 13F filers by prior-quarter AUM instead of CIK age"
```

---

## Task 4: Pipeline Health Check Tool

**Files:**
- Create: `tools/pipeline_health.py`

- [ ] **Step 1: Create the tools directory and write the script**

```bash
mkdir -p tools
```

```python
# tools/pipeline_health.py
"""
Pipeline health check — prints data freshness for every source.

Usage:
    python tools/pipeline_health.py

Exit code 0 = all healthy, 1 = one or more sources stale or missing.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

_PROJECT_ROOT = Path(__file__).parent.parent
_TODAY = date.today()

# Each entry: (display_name, data_path_or_glob, date_column, cadence)
# cadence: "daily" → warn if >3 days, "weekly" → warn if >10 days, "quarterly" → never warn
_SOURCES: list[tuple[str, str, str, str]] = [
    (
        "OHLCV",
        str(_PROJECT_ROOT / "data/raw/financials/ohlcv/**/*.parquet"),
        "date",
        "daily",
    ),
    (
        "Short interest",
        str(_PROJECT_ROOT / "data/raw/financials/short_interest/short_interest_daily.parquet"),
        "date",
        "daily",
    ),
    (
        "Earnings",
        str(_PROJECT_ROOT / "data/raw/financials/earnings/earnings_surprises.parquet"),
        "quarter_end",
        "weekly",
    ),
    (
        "Sentiment (scored)",
        str(_PROJECT_ROOT / "data/raw/news/scored/**/*.parquet"),
        "article_date",
        "daily",
    ),
    (
        "Sentiment (features)",
        str(_PROJECT_ROOT / "data/raw/news/sentiment_features/*/daily.parquet"),
        "date",
        "daily",
    ),
    (
        "Graph features",
        str(_PROJECT_ROOT / "data/raw/graph/features/*/graph_daily.parquet"),
        "date",
        "weekly",
    ),
    (
        "13F holdings (raw)",
        str(_PROJECT_ROOT / "data/raw/financials/13f_holdings/raw/**/*.parquet"),
        "period_end",
        "quarterly",
    ),
    (
        "13F features",
        str(_PROJECT_ROOT / "data/raw/financials/13f_holdings/features/*/quarterly.parquet"),
        "period_end",
        "quarterly",
    ),
]

_STALE_DAYS = {"daily": 3, "weekly": 10, "quarterly": 999}


def _latest_date(glob_or_path: str, date_col: str) -> date | None:
    try:
        with duckdb.connect() as con:
            result = con.execute(
                f"SELECT MAX({date_col}) FROM read_parquet(?)", [glob_or_path]
            ).fetchone()
        val = result[0] if result else None
        if val is None:
            return None
        return val if isinstance(val, date) else date.fromisoformat(str(val))
    except Exception:
        return None


def main() -> int:
    print(f"\nAI Infra Predictor — Pipeline Health Check ({_TODAY})")
    print("═" * 60)
    print(f"{'Source':<28} {'Latest':>12}  {'Age':>8}  Status")
    print("─" * 60)

    any_problem = False
    for name, glob_path, date_col, cadence in _SOURCES:
        latest = _latest_date(glob_path, date_col)
        if latest is None:
            print(f"{name:<28} {'MISSING':>12}  {'—':>8}  ✗ MISSING")
            any_problem = True
            continue

        age_days = (_TODAY - latest).days
        threshold = _STALE_DAYS[cadence]
        status = "✓ OK" if age_days <= threshold else f"✗ STALE ({cadence} threshold: {threshold}d)"
        if age_days > threshold:
            any_problem = True

        print(f"{name:<28} {str(latest):>12}  {age_days:>6}d  {status}")

    print("═" * 60)
    if any_problem:
        print("Issues found. Run: bash tools/run_refresh.sh\n")
        return 1
    print("All sources healthy.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
python tools/pipeline_health.py
```

Expected: Prints a table. Any missing or stale sources shown with ✗.

- [ ] **Step 3: Commit**

```bash
git add tools/pipeline_health.py
git commit -m "feat: add pipeline_health.py — data freshness check for all sources"
```

---

## Task 5: Pipeline Refresh Script

**Files:**
- Create: `tools/run_refresh.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/9  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/9  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 3/9  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 4/9  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 5/9  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 6/9  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 7/9  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 8/9  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 9/9  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
```

- [ ] **Step 2: Make executable and test**

```bash
chmod +x tools/run_refresh.sh
# Dry-run syntax check only (don't actually run all ingestion):
zsh -n tools/run_refresh.sh
echo "Syntax OK"
```

Expected: "Syntax OK"

- [ ] **Step 3: Commit**

```bash
git add tools/run_refresh.sh
git commit -m "feat: add run_refresh.sh — ordered pipeline refresh script"
```

---

## Final Check

```bash
pytest tests/ -m "not integration" -q
python tools/pipeline_health.py
```

Expected: All tests pass. Health check runs without errors.
