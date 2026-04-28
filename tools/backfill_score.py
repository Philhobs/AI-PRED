"""Score news parquets per-date partition (for sentiment backfill).

Why this exists:
  processing/nlp_pipeline.py's __main__ scores all input articles and writes a
  SINGLE output file at scored/date={mode_date}/data.parquet — fine for daily
  ingest, broken for backfill where input partitions span 60-90 calendar days.
  This tool scores each input date partition independently, writing to the
  matching scored/date={D}/data.parquet so the existing sentiment join works
  unchanged.

Idempotent: skips date partitions whose scored output already exists.

Run: python -m tools.backfill_score
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

from processing.nlp_pipeline import FinBERTScorer

_RAW_DIRS = [
    Path("data/raw/news/rss"),
    Path("data/raw/news/gdelt"),
]
_SCORED_DIR = Path("data/raw/news/scored")
_DATE_RE = re.compile(r"date=(\d{4}-\d{2}-\d{2})$")


def _enumerate_date_partitions() -> list[str]:
    dates: set[str] = set()
    for raw_dir in _RAW_DIRS:
        if not raw_dir.exists():
            continue
        for child in raw_dir.iterdir():
            if not child.is_dir():
                continue
            m = _DATE_RE.match(child.name)
            if not m:
                continue
            if (child / "data.parquet").exists():
                dates.add(m.group(1))
    return sorted(dates)


def _input_paths_for(date_str: str) -> list[str]:
    paths = []
    for raw_dir in _RAW_DIRS:
        candidate = raw_dir / f"date={date_str}" / "data.parquet"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def score_date_partition(scorer: FinBERTScorer, date_str: str, force: bool = False) -> tuple[int, str]:
    """Score one date's RSS+GDELT partitions; return (n_articles, status)."""
    out_path = _SCORED_DIR / f"date={date_str}" / "data.parquet"
    if out_path.exists() and not force:
        return 0, "skip"

    paths = _input_paths_for(date_str)
    if not paths:
        return 0, "noinput"

    df = duckdb.read_parquet(paths).fetchdf()
    if df.empty:
        return 0, "empty"

    texts = df["content_snippet"].fillna("").tolist()
    scores = scorer.score_batch(texts)

    scored_df = df.copy()
    for field in ["positive", "negative", "neutral", "net_sentiment", "label"]:
        scored_df[field] = [s[field] for s in scores]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(str(out_path), compression="snappy")
    return len(scored_df), "scored"


def backfill_all(force: bool = False) -> None:
    dates = _enumerate_date_partitions()
    if not dates:
        print("[backfill_score] No date partitions found in data/raw/news/{rss,gdelt}/")
        return

    print(f"[backfill_score] Loading FinBERT (first run downloads ~440MB)...")
    scorer = FinBERTScorer()

    totals = {"scored": 0, "skip": 0, "noinput": 0, "empty": 0}
    rows_written = 0
    for d in dates:
        n, status = score_date_partition(scorer, d, force=force)
        totals[status] += 1
        rows_written += n
        if status == "scored":
            print(f"[backfill_score] {d}: scored {n} articles")
        elif status == "skip":
            print(f"[backfill_score] {d}: scored output exists — skip")

    print(
        f"\n[backfill_score] done. scored={totals['scored']} skip={totals['skip']} "
        f"empty={totals['empty']} noinput={totals['noinput']} | total rows written: {rows_written}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="Re-score even if scored output exists.")
    args = parser.parse_args()

    backfill_all(force=args.force)
