"""Backfill GDELT news for the last N days, one calendar day per request.

Why this exists:
  ingestion/news_ingestion.py's __main__ runs once per cron tick and writes only
  today's articles. For sentiment_features.py to populate `sentiment_mean_7d`
  on training rows older than today, we need scored news whose article_date
  matches each spine date in a 7-day window. This script fetches each historical
  day independently and writes per-date partitions that the existing scorer
  + sentiment join consume unchanged.

Run: python -m tools.backfill_news --days 90
Output: data/raw/news/gdelt/date=YYYY-MM-DD/data.parquet for each missing day.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from ingestion.news_ingestion import (
    GDELT_DOC_API,
    SCHEMA,
    _tag_tickers,
)

_OUTPUT_DIR = Path("data/raw/news/gdelt")

# Same OR-grouped query as live ingestion. Keep in sync.
# GDELT requires every quoted phrase to be ≥5 chars or the API returns 200
# with body "The specified phrase is too short." and 0 articles — silent fail.
_QUERY = (
    '("data center" OR "semiconductor" OR "nuclear power" OR '
    '"export control" OR "AI chip" OR "hyperscaler" OR '
    '"humanoid robot" OR "industrial automation" OR "AI agent" OR '
    '"agentic AI" OR "ransomware" OR "zero trust")'
)


def _fetch_day(day: date) -> list[dict]:
    """Fetch GDELT articles published on a single UTC calendar day."""
    start_dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)
    params = {
        "query": _QUERY,
        "mode": "artlist",
        "maxrecords": 250,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
        "format": "json",
    }
    resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.text
    # GDELT returns 200 + plain-text error like "The specified phrase is too short."
    # for malformed queries — guard against feeding that to .json()
    if not body.lstrip().startswith("{"):
        raise requests.RequestException(f"GDELT non-JSON response: {body[:80]!r}")
    data = resp.json()

    # Use midnight-of-day as the timestamp so downstream
    # `CAST(timestamp AS DATE)` lands on the correct article_date for the
    # sentiment join's BETWEEN window.
    ts = start_dt
    return [
        {
            "timestamp": ts,
            "source": "gdelt",
            "url": art.get("url", ""),
            "title": art.get("title", ""),
            "content_snippet": (art.get("title", "") + " " + art.get("excerpt", ""))[:500].strip(),
            "theme_tags": [],
            "goldstein_score": 0.0,
            "tone_score": float(str(art.get("tone", "0") or "0").split(",")[0]) if art.get("tone") else 0.0,
            "num_articles": 1,
            "actors": [],
            "countries": [],
            "mentioned_tickers": _tag_tickers(art.get("title", ""), art.get("excerpt", "")),
        }
        for art in data.get("articles", [])
    ]


def _save_day(records: list[dict], day: date) -> Path:
    out_path = _OUTPUT_DIR / f"date={day.isoformat()}" / "data.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def backfill(days: int, sleep_s: float = 2.0, force: bool = False) -> None:
    today = date.today()
    # Newest-first so the dates that matter most for the current spine
    # (within the 7-day sentiment window) are written before the older ones.
    targets = [today - timedelta(days=i) for i in range(1, days + 1)]
    targets.sort(reverse=True)

    fetched = skipped = empty = errored = 0
    for d in targets:
        out_path = _OUTPUT_DIR / f"date={d.isoformat()}" / "data.parquet"
        if out_path.exists() and not force:
            print(f"[backfill] {d}: exists — skip", flush=True)
            skipped += 1
            continue
        try:
            records = _fetch_day(d)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            print(f"[backfill] {d}: HTTP {status} — skip", flush=True)
            errored += 1
            time.sleep(sleep_s)
            continue
        except requests.RequestException as exc:
            print(f"[backfill] {d}: {exc} — skip", flush=True)
            errored += 1
            time.sleep(sleep_s)
            continue

        if not records:
            print(f"[backfill] {d}: 0 articles — skip write", flush=True)
            empty += 1
        else:
            path = _save_day(records, d)
            print(f"[backfill] {d}: {len(records)} articles → {path}", flush=True)
            fetched += 1
        time.sleep(sleep_s)

    print(
        f"\n[backfill] done. fetched={fetched} skipped(existed)={skipped} "
        f"empty={empty} errored={errored} (of {len(targets)} target days)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days (default 90).")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between requests (default 2.0, GDELT is rate-limit prone).")
    parser.add_argument("--force", action="store_true", help="Re-fetch days that already have a parquet on disk.")
    args = parser.parse_args()

    backfill(days=args.days, sleep_s=args.sleep, force=args.force)
