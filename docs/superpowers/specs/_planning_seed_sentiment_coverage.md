# Planning Seed: Sentiment Coverage Gap

**Trigger phrase:** "fix sentiment coverage" or "score GDELT + EDGAR news"
**Effort:** 3-5h
**Created:** 2026-04-27 (during sentiment "0 non-null" investigation)

## Problem

`sentiment_mean_7d` was zero-non-null across the 165-ticker universe. Investigation found three independent issues:

1. **Spine date drift.** OHLCV spine ended 2026-04-24; only news scored was from 2026-04-25/26 (40 articles each day). The 7-day join window contained no scored news for any spine date.
2. **GDELT producing zero articles** (FIXED 2026-04-27). Root cause: query `"semiconductor export control chip data center nuclear power"` was treated as a single phrase by GDELT. Switched to OR-grouped query → returns 250/day.
3. **EDGAR full-text 403** (FIXED 2026-04-27). Root cause: missing `User-Agent` header. Added one matching the convention in `deal_ingestion.py`/`sec_13f_ingestion.py`/etc. → 0 → 100 hits.

The RSS-only path was working (~40 articles/day) and is what populated the two scored days that exist.

## Remaining gaps (not fixed)

### 1. GDELT articles never get scored
`processing/nlp_pipeline.py:127` only reads `data/raw/news/rss/date=*/data.parquet`. GDELT articles land in `data/raw/news/gdelt/date=*/data.parquet` and are silently ignored. Should be merged into the input glob.

**Why:** GDELT now produces 6× more articles than RSS — leaving them unscored throws away the bulk of the signal.

**How:** Either glob both directories or add a second scoring pass. Schema is already identical (both use `SCHEMA` from `news_ingestion.py`).

### 2. EDGAR hits never persisted
`ingestion/news_ingestion.py:340` calls `search_edgar_fulltext(...)` and only prints the count. The `hits` list is discarded.

**Why:** 100 8-K hits/day around topics like "power purchase agreement AND data center" is exactly the kind of high-signal text we want feeding sentiment.

**How:** Normalize the EDGAR `_source.{file_date,entity_name,form_type}` into the same SCHEMA used by RSS/GDELT and write to `data/raw/news/edgar/date=YYYY-MM-DD/data.parquet`. The mentioned_tickers tag would come from matching `entity_name` to `TICKERS_INFO[*].name`.

### 3. No backfill mechanism
Even with #1+#2 fixed, the existing scored archive only has Apr 25-26. To populate sentiment for the trailing-90-day spine we'd need a one-time backfill that fetches RSS/GDELT/EDGAR for each historical date.

**Constraint:** RSS feeds typically only expose the last ~50 items. GDELT goes back years. EDGAR full-text goes back to 2001. Backfill effectiveness varies wildly by source.

**How:** Probably easiest to add a `--backfill-from YYYY-MM-DD` flag that loops day-by-day, calling each fetcher with `days_back=1` and writing under the historical date partition (not "today"). Then re-run `nlp_pipeline.py` over the full archive.

### 4. Spine alignment is brittle
The 7-day window assumes article dates ≤ spine date + 7d. If spine is stale (e.g., weekend, holiday, ingestion failure), the window opens but no news lines up. Worth investigating whether the join should be "last 7 calendar days before max(article_date)" rather than "centered on spine date".

## Success criteria

- `sentiment_mean_7d` non-null for ≥ 50% of (ticker × date) rows in the most recent 30 days
- `article_count_7d` median ≥ 5 across the universe
- GDELT + EDGAR + RSS all contribute to the scored archive (verifiable via `source` column)

## What was already shipped (2026-04-27)

- `ingestion/news_ingestion.py`: GDELT query OR-grouped, EDGAR User-Agent added, `--days N` argparse flag (default 7) for cold-start sizing.
