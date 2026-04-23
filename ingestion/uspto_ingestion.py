"""USPTO patent ingestion from PatentsView v2 API.

Fetches published patent applications and granted patents for AI/semiconductor CPC codes.
Output parquet files (snappy):
  data/raw/patents/applications/date=YYYY-MM-DD/apps.parquet
  data/raw/patents/grants/date=YYYY-MM-DD/grants.parquet

Staleness guard: skips re-download if existing snapshot is from the same ISO week
(PatentsView updates weekly).
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_APPS_URL = "https://api.patentsview.org/applications/query"
_GRANTS_URL = "https://api.patentsview.org/patents/query"
_CPC_CODES = ["G06N", "H01L", "G06F", "G11C"]
_PER_PAGE = 100
_SLEEP_BETWEEN_PAGES = 1.5
_MAX_PAGES = 200  # safety circuit breaker: ~20,000 records maximum

_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}


def _lookback_start(date_str: str) -> str:
    """Return the date 365 days before date_str as YYYY-MM-DD."""
    d = datetime.date.fromisoformat(date_str)
    return (d - datetime.timedelta(days=365)).isoformat()


def _same_iso_week(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent existing parquet in existing_dir is from the same ISO week as today."""
    files = sorted(existing_dir.glob("date=*/apps.parquet")) + sorted(existing_dir.glob("date=*/grants.parquet"))
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


def fetch_applications(date_str: str) -> pl.DataFrame:
    """Fetch all AI/semiconductor patent applications in the 365-day window ending date_str.

    Returns DataFrame with _APP_SCHEMA. Empty DataFrame if no results.
    """
    start = _lookback_start(date_str)
    records: list[dict] = []
    page = 1

    while True:
        payload = {
            "q": {"_and": [
                {"_gte": {"app_date": start}},
                {"_lte": {"app_date": date_str}},
                {"_or": [{"cpc_group_id": c} for c in _CPC_CODES]},
            ]},
            "f": ["app_id", "assignee_organization", "cpc_group_id", "app_date"],
            "o": {"per_page": _PER_PAGE, "page": page},
        }
        resp = requests.post(_APPS_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("applications", [])
        records.extend(batch)

        total = data.get("total_app_count", 0)
        if len(records) >= total or not batch:
            break
        if page >= _MAX_PAGES:
            _LOG.warning("USPTO: reached _MAX_PAGES=%d limit — result may be truncated", _MAX_PAGES)
            break
        time.sleep(_SLEEP_BETWEEN_PAGES)
        page += 1

    if not records:
        return pl.DataFrame(schema=_APP_SCHEMA)

    run_date = datetime.date.fromisoformat(date_str)
    rows = []
    for r in records:
        try:
            filing = datetime.date.fromisoformat(r["app_date"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "date": run_date,
            "assignee_name": r.get("assignee_organization") or "",
            "app_id": r.get("app_id") or "",
            "cpc_group": r.get("cpc_group_id") or "",
            "filing_date": filing,
        })

    return pl.DataFrame(rows, schema=_APP_SCHEMA)


def fetch_grants(date_str: str) -> pl.DataFrame:
    """Fetch all AI/semiconductor granted patents in the 365-day window ending date_str.

    Returns DataFrame with _GRANT_SCHEMA. Empty DataFrame if no results.
    """
    start = _lookback_start(date_str)
    records: list[dict] = []
    page = 1

    while True:
        payload = {
            "q": {"_and": [
                {"_gte": {"patent_date": start}},
                {"_lte": {"patent_date": date_str}},
                {"_or": [{"cpc_group_id": c} for c in _CPC_CODES]},
            ]},
            "f": ["patent_id", "assignee_organization", "cpc_group_id", "patent_date", "cited_by_count"],
            "o": {"per_page": _PER_PAGE, "page": page},
        }
        resp = requests.post(_GRANTS_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("patents", [])
        records.extend(batch)

        total = data.get("total_patent_count", 0)
        if len(records) >= total or not batch:
            break
        if page >= _MAX_PAGES:
            _LOG.warning("USPTO: reached _MAX_PAGES=%d limit — result may be truncated", _MAX_PAGES)
            break
        time.sleep(_SLEEP_BETWEEN_PAGES)
        page += 1

    if not records:
        return pl.DataFrame(schema=_GRANT_SCHEMA)

    run_date = datetime.date.fromisoformat(date_str)
    rows = []
    for r in records:
        try:
            grant_d = datetime.date.fromisoformat(r["patent_date"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            citation_count = int(r.get("cited_by_count") or 0)
        except (TypeError, ValueError):
            citation_count = 0
        rows.append({
            "date": run_date,
            "assignee_name": r.get("assignee_organization") or "",
            "patent_id": r.get("patent_id") or "",
            "cpc_group": r.get("cpc_group_id") or "",
            "grant_date": grant_d,
            "forward_citation_count": citation_count,
        })

    return pl.DataFrame(rows, schema=_GRANT_SCHEMA)


def ingest_uspto(date_str: str, apps_dir: Path, grants_dir: Path) -> None:
    """Fetch and persist patent applications and grants for date_str.

    Skips download if both directories already have a same-ISO-week snapshot.
    Writes nothing when results are empty.
    """
    if _same_iso_week(apps_dir, date_str) and _same_iso_week(grants_dir, date_str):
        _LOG.info("USPTO: same ISO week snapshot exists — skipping download for %s", date_str)
        return

    _LOG.info("USPTO: fetching applications for %s (365d lookback)", date_str)
    apps_df = fetch_applications(date_str)
    if not apps_df.is_empty():
        out = apps_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        apps_df.write_parquet(out / "apps.parquet", compression="snappy")
        _LOG.info("USPTO: wrote %d application rows to %s", len(apps_df), out)

    _LOG.info("USPTO: fetching grants for %s (365d lookback)", date_str)
    grants_df = fetch_grants(date_str)
    if not grants_df.is_empty():
        out = grants_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        grants_df.write_parquet(out / "grants.parquet", compression="snappy")
        _LOG.info("USPTO: wrote %d grant rows to %s", len(grants_df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    date_str = datetime.date.today().isoformat()
    base = Path("data/raw/patents")
    ingest_uspto(date_str, base / "applications", base / "grants")
