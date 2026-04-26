"""USAJOBS federal AI/ML job postings ingestion.

Fetches federal job postings across 5 AI/ML keyword terms.
Output: data/raw/usajobs/date=YYYY-MM-DD/postings.parquet

Staleness guard: skips re-download if existing snapshot is from the same ISO week.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://data.usajobs.gov/api/search"
_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "GPU computing",
    "semiconductor",
]
_RESULTS_PER_PAGE = 500
_DATE_POSTED = 60  # 60-day lookback covers both the 30d and prior-30d windows
_SLEEP_BETWEEN_KEYWORDS = 1.0
_MAX_PAGES = 20  # safety cap: 20 × 500 = 10,000 results

_SCHEMA = {
    "date": pl.Date,
    "posting_id": pl.Utf8,
    "title": pl.Utf8,
    "posted_date": pl.Date,
    "keyword": pl.Utf8,
}


def _same_iso_week(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent parquet in existing_dir is from the same ISO week as today."""
    files = sorted(existing_dir.glob("date=*/postings.parquet"))
    if not files:
        return False
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        last_iso = last_date.isocalendar()
        today_iso = today.isocalendar()
        return last_iso.week == today_iso.week and last_iso.year == today_iso.year
    except ValueError:
        return False


def _fetch_keyword(keyword: str, run_date: datetime.date, user_agent: str) -> list[dict]:
    """Fetch all USAJOBS postings for one keyword term across all pages."""
    headers = {"Host": "data.usajobs.gov", "User-Agent": user_agent}
    records: list[dict] = []
    fetched = 0
    page = 1

    while page <= _MAX_PAGES:
        params = {
            "Keyword": keyword,
            "DatePosted": str(_DATE_POSTED),
            "ResultsPerPage": str(_RESULTS_PER_PAGE),
            "Page": str(page),
        }
        resp = requests.get(_BASE_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("SearchResult", {}).get("SearchResultItems", [])
        if not items:
            break

        for item in items:
            desc = item.get("MatchedObjectDescriptor", {})
            position_id = desc.get("PositionID", "")
            if not position_id:
                continue
            pub_start = desc.get("PublicationStartDate", "")
            try:
                posted = datetime.date.fromisoformat(pub_start[:10])
            except (ValueError, TypeError):
                continue
            records.append({
                "date": run_date,
                "posting_id": position_id,
                "title": desc.get("PositionTitle", ""),
                "posted_date": posted,
                "keyword": keyword,
            })

        total = data.get("SearchResult", {}).get("SearchResultCountAll")
        if total is None:
            _LOG.warning("USAJOBS: SearchResultCountAll missing — cannot paginate, stopping after page %d", page)
            break
        fetched += len(items)
        if fetched >= total or len(items) < _RESULTS_PER_PAGE:
            break
        page += 1

    if page > _MAX_PAGES:
        _LOG.warning("USAJOBS: reached _MAX_PAGES=%d limit for keyword %r — result may be truncated", _MAX_PAGES, keyword)
    return records


def fetch_postings(date_str: str) -> pl.DataFrame:
    """Fetch all federal AI/ML job postings, deduplicated on posting_id.

    Returns DataFrame with _SCHEMA. Empty DataFrame if no results.
    """
    user_agent = os.environ.get("USAJOBS_USER_AGENT", "ai-pred-research@example.com")
    run_date = datetime.date.fromisoformat(date_str)

    all_records: list[dict] = []
    for i, keyword in enumerate(_KEYWORDS):
        all_records.extend(_fetch_keyword(keyword, run_date, user_agent))
        if i < len(_KEYWORDS) - 1:
            time.sleep(_SLEEP_BETWEEN_KEYWORDS)

    if not all_records:
        return pl.DataFrame(schema=_SCHEMA)

    return (
        pl.DataFrame(all_records, schema=_SCHEMA)
        .unique(subset=["posting_id"], keep="first")
    )


def ingest_usajobs(date_str: str, output_dir: Path) -> None:
    """Fetch and persist USAJOBS postings for date_str.

    Skips download if same-ISO-week snapshot exists. Writes nothing when results are empty.
    """
    if _same_iso_week(output_dir, date_str):
        _LOG.info("USAJOBS: same ISO week snapshot exists — skipping for %s", date_str)
        return

    _LOG.info("USAJOBS: fetching AI/ML federal job postings for %s", date_str)
    df = fetch_postings(date_str)
    if not df.is_empty():
        out = output_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "postings.parquet", compression="snappy")
        _LOG.info("USAJOBS: wrote %d posting rows to %s", len(df), out)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ingest_usajobs(
        datetime.date.today().isoformat(),
        Path("data/raw/usajobs"),
    )
