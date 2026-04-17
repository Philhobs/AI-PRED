# Phase C-A — Data Pipeline Health & Quality Fixes

**Goal:** Fix two silent data bugs (sentiment 0 non-null, 13F filer ranking heuristic) and add operational tooling to keep the pipeline healthy.

**Architecture:** Two bug fixes in existing modules + two new tools (`pipeline_health.py`, `run_refresh.sh`). No new deps.

**Tech Stack:** Python 3.11, Polars, yfinance (already used), zsh

---

## 1. Bug Fix — Sentiment 0 Non-Null in Training

**Root cause:** Sentiment features exist only for recent dates (~1 year), but the training spine goes back 10 years. The backward-asof join finds no match for the vast majority of rows because there is no historical sentiment data older than ~1 year, leaving `sentiment_mean_7d` etc. null everywhere.

**Fix location:** `processing/sentiment_features.py` → `join_sentiment_features()`

**Current behaviour:** Backward-asof join on full training spine → 0 non-null for pre-sentiment dates.

**Fix:** Cap the look-back to 30 days in the asof join. Any training row more than 30 days past the most recent sentiment date gets null — this is correct and honest. But rows within 30 days of a sentiment observation should get that value.

Concretely: add `tolerance=timedelta(days=30)` to the `join_asof()` call. This is the Polars `join_asof` `tolerance` parameter — it sets the maximum allowed gap between the left key and the matched right key. Without it, asof joins match arbitrarily far back.

```python
# in join_sentiment_features(), change:
result = df.sort(["ticker", "date"]).join_asof(
    features_renamed,
    left_on="date",
    right_on="sentiment_date",
    by="ticker",
    strategy="backward",
)
# to:
from datetime import timedelta
result = df.sort(["ticker", "date"]).join_asof(
    features_renamed,
    left_on="date",
    right_on="sentiment_date",
    by="ticker",
    strategy="backward",
    tolerance=timedelta(days=30),
)
```

This means: for training rows before sentiment data existed, features are null (correct — we had no data). For rows after sentiment data exists, features propagate up to 30 days forward (correct — sentiment is valid for a short window).

**Same fix applies to:** Check `join_short_interest_features()` in `processing/short_interest_features.py` — if short interest data is also sparse in early history, apply the same `tolerance=timedelta(days=7)` (short interest changes more frequently, 7 days is appropriate).

---

## 2. Bug Fix — 13F Filer Ranking

**Root cause:** `rank_filers_by_position_count()` in `ingestion/sec_13f_ingestion.py` ranks by CIK integer value (lower = older registration = assumed larger). This is a poor proxy: many large quant funds registered recently; many old CIKs belong to tiny inactive filers.

**Fix:** Use prior-quarter's already-downloaded holdings data to rank by total portfolio value (`sum(value_usd_thousands)` per CIK). Falls back to CIK-age sort if no prior-quarter data exists (first bootstrap quarter).

**Implementation:**

```python
def rank_filers_by_position_count(
    index_df: pl.DataFrame,
    top_n: int = 500,
    prior_quarter_dir: Path | None = None,
) -> list[str]:
    """
    Rank filers by total portfolio value from prior quarter if available,
    otherwise by CIK integer (age proxy).
    """
    if prior_quarter_dir is not None and prior_quarter_dir.exists():
        parquets = list(prior_quarter_dir.glob("*.parquet"))
        if parquets:
            prior = pl.concat([pl.read_parquet(p) for p in parquets])
            aum_rank = (
                prior.group_by("cik")
                .agg(pl.col("value_usd_thousands").sum().alias("total_value"))
                .sort("total_value", descending=True)
            )
            ranked_ciks = aum_rank["cik"].to_list()
            # Keep only CIKs that appear in this quarter's index
            index_ciks = set(index_df["cik"].to_list())
            ranked_ciks = [c for c in ranked_ciks if c in index_ciks]
            # Append any index CIKs not seen in prior quarter (sorted by CIK age)
            seen = set(ranked_ciks)
            remaining = sorted(
                [c for c in index_ciks if c not in seen],
                key=lambda x: int(x)
            )
            return (ranked_ciks + remaining)[:top_n]

    # Fallback: CIK-age sort (lower CIK = older registration)
    return (
        index_df.sort("cik", descending=False)["cik"]
        .unique(maintain_order=True)
        .to_list()[:top_n]
    )
```

**Wire up in `ingest_quarter()`:** pass the prior quarter's directory:

```python
prior_qtr = _prior_quarter(year, quarter)
prior_dir = output_dir / f"{prior_qtr[0]}Q{prior_qtr[1]}"
top_ciks = rank_filers_by_position_count(index_df, top_n=top_n, prior_quarter_dir=prior_dir)
```

Add helper:
```python
def _prior_quarter(year: int, quarter: int) -> tuple[int, int]:
    if quarter == 1:
        return (year - 1, 4)
    return (year, quarter - 1)
```

---

## 3. New Tool — `tools/pipeline_health.py`

Prints a freshness table for every data source. Run anytime to see what's stale.

```
$ python tools/pipeline_health.py

AI Infra Predictor — Pipeline Health Check (2026-04-17)
═══════════════════════════════════════════════════════
Source                    Latest Date     Age      Status
─────────────────────────────────────────────────────────
OHLCV                     2026-04-15      2 days   ✓ OK
Short interest            2026-04-15      2 days   ✓ OK
Earnings                  2026-03-31      17 days  ✓ OK
Sentiment (scored)        2026-04-15      2 days   ✓ OK
Sentiment (features)      2026-04-15      2 days   ✓ OK
Graph features            2026-04-10      7 days   ✓ OK
13F holdings              2026-03-31      17 days  ✓ OK (quarterly)
13F features              2026-03-31      17 days  ✓ OK (quarterly)
═══════════════════════════════════════════════════════
All sources healthy.
```

**Schema per source:** dict of `{name, glob_or_path, date_col, cadence}` where cadence is "daily" (warn if >3 days stale), "weekly" (warn if >10 days), or "quarterly" (never stale-warn).

**File:** `tools/pipeline_health.py` (new, ~100 lines)

---

## 4. New Tool — `tools/run_refresh.sh`

Ordered shell script for a full pipeline refresh. Documents the correct execution order.

```bash
#!/bin/zsh
# Full pipeline refresh — run from project root
set -e
cd "$(dirname "$0")/.."

echo "=== 1/9 OHLCV ==="
python ingestion/ohlcv_ingestion.py

echo "=== 2/9 Short interest ==="
python ingestion/short_interest_ingestion.py

echo "=== 3/9 Earnings ==="
python ingestion/earnings_ingestion.py

echo "=== 4/9 News ingestion ==="
python ingestion/news_ingestion.py

echo "=== 5/9 NLP pipeline (FinBERT) ==="
python processing/nlp_pipeline.py

echo "=== 6/9 Sentiment features ==="
python processing/sentiment_features.py

echo "=== 7/9 Graph features ==="
python processing/graph_features.py

echo "=== 8/9 13F ingestion (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo "=== 9/9 Ownership features ==="
python processing/ownership_features.py

echo "=== Refresh complete. Run: python models/train.py ==="
```

**File:** `tools/run_refresh.sh` (new, ~30 lines, chmod +x)

---

## 5. Storage

No new storage. Existing paths unchanged.

---

## 6. Testing

**`tests/test_sentiment_join_tolerance.py`** (new):
- `test_sentiment_join_no_stale_propagation` — sentiment feature at date T should not propagate to date T+31 (31 days later gets null)
- `test_sentiment_join_propagates_within_30d` — sentiment at T propagates to T+15 (within window)
- `test_sentiment_join_null_before_data_exists` — rows from 3 years ago get null (no data)

**`tests/test_13f_ranking.py`** (new):
- `test_rank_uses_prior_quarter_aum` — given a mock prior-quarter dir with 3 CIKs of known values, verify ranking order matches value descending
- `test_rank_falls_back_to_cik_age` — when prior_quarter_dir is None, falls back to CIK-integer sort
- `test_rank_appends_new_ciks` — CIKs present in index but absent from prior quarter are appended at the end

---

## 7. Files Created / Modified

**New:**
```
tools/pipeline_health.py
tools/run_refresh.sh
tests/test_sentiment_join_tolerance.py
tests/test_13f_ranking.py
```

**Modified:**
```
processing/sentiment_features.py        ← add tolerance=timedelta(days=30)
processing/short_interest_features.py  ← add tolerance=timedelta(days=7)
ingestion/sec_13f_ingestion.py          ← AUM-based ranking + _prior_quarter helper
```
