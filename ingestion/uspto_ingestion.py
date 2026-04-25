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

# Physical-AI mode: map bucket name → CPC class prefixes that count toward it.
# A patent's cpc_group_id must START WITH any of the prefixes to be counted in that bucket.
_PHYSICAL_AI_BUCKETS: dict[str, tuple[str, ...]] = {
    "B25J":   ("B25J",),
    "B64":    ("B64C", "B64U"),
    "B60W":   ("B60W",),
    "G05D1":  ("G05D1",),
    "G05B19": ("G05B19",),
    "G06V":   ("G06V",),
}

_PHYSICAL_AI_AGG_SCHEMA = {
    "quarter_end": pl.Date,
    "cpc_class":   pl.Utf8,
    "filing_count": pl.Int64,
}
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


def _quarter_end(d: "datetime.date") -> "datetime.date":
    """Return the last calendar day of d's quarter (Mar 31 / Jun 30 / Sep 30 / Dec 31)."""
    quarter = (d.month - 1) // 3 + 1
    last_month = quarter * 3
    if last_month == 3:
        return datetime.date(d.year, 3, 31)
    if last_month == 6:
        return datetime.date(d.year, 6, 30)
    if last_month == 9:
        return datetime.date(d.year, 9, 30)
    return datetime.date(d.year, 12, 31)


def _bucket_for_cpc(cpc_group: str) -> str | None:
    """Return the bucket name for a cpc_group_id, or None if no match.

    CPC classes look like 'G05D1/02' (main group + subgroup) or 'B25J9/02' (subclass + main group + subgroup).
    Strip the subgroup, then check if the head:
      - Exactly matches a bucket prefix (main-group prefix like 'G05D1' matches only 'G05D1' / 'G05D1/02').
      - Or starts with a subclass prefix (last char is a letter) followed by a digit
        (so 'B25J' bucket matches all main groups B25J1, B25J9, B25J11, etc.).
    """
    head = cpc_group.split("/", 1)[0]   # 'G05D1/02' -> 'G05D1', 'B25J9' -> 'B25J9'
    for bucket, prefixes in _PHYSICAL_AI_BUCKETS.items():
        for p in prefixes:
            if head == p:
                return bucket
            if (
                not p[-1].isdigit()           # prefix is a subclass (e.g. B25J, B64C)
                and head.startswith(p)
                and len(head) > len(p)
                and head[len(p)].isdigit()    # next char is a digit (a main group within the subclass)
            ):
                return bucket
    return None


def _aggregate_physical_ai(raw: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw filings (filing_date, cpc_group) → (quarter_end, cpc_class, filing_count)."""
    if raw.is_empty():
        return pl.DataFrame(schema=_PHYSICAL_AI_AGG_SCHEMA)

    df = raw.with_columns([
        pl.col("filing_date").map_elements(_quarter_end, return_dtype=pl.Date).alias("quarter_end"),
        pl.col("cpc_group").map_elements(_bucket_for_cpc, return_dtype=pl.Utf8).alias("cpc_class"),
    ])
    df = df.filter(pl.col("cpc_class").is_not_null())
    if df.is_empty():
        return pl.DataFrame(schema=_PHYSICAL_AI_AGG_SCHEMA)
    return (
        df.group_by(["quarter_end", "cpc_class"])
          .agg(pl.len().alias("filing_count").cast(pl.Int64))
          .sort(["quarter_end", "cpc_class"])
    )


def fetch_physical_ai_filings(date_str: str) -> pl.DataFrame:
    """Fetch all filings in the 365-day window ending date_str whose cpc_group matches
    any physical-AI bucket prefix. Returns aggregated (quarter_end, cpc_class, filing_count)."""
    start = _lookback_start(date_str)
    all_prefixes = sorted({p for prefixes in _PHYSICAL_AI_BUCKETS.values() for p in prefixes})

    records: list[dict] = []
    for prefix in all_prefixes:
        page = 1
        while page <= _MAX_PAGES:
            payload = {
                "q": {"_and": [
                    {"_gte": {"app_date": start}},
                    {"_lte": {"app_date": date_str}},
                    {"_begins": {"cpc_group_id": prefix}},
                ]},
                "f": ["app_id", "cpc_group_id", "app_date"],
                "o": {"per_page": _PER_PAGE, "page": page},
            }
            try:
                resp = requests.post(_APPS_URL, json=payload, timeout=30)
                resp.raise_for_status()
                page_data = resp.json().get("applications", [])
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("[USPTO] physical_ai prefix=%s page=%d: %s", prefix, page, exc)
                break
            if not page_data:
                break
            for entry in page_data:
                try:
                    fd = datetime.date.fromisoformat(entry["app_date"])
                except (KeyError, ValueError):
                    continue
                cpc = entry.get("cpc_group_id", "")
                if not cpc:
                    continue
                records.append({"filing_date": fd, "cpc_group": cpc})
            page += 1
            time.sleep(_SLEEP_BETWEEN_PAGES)

    raw = pl.DataFrame(records, schema={"filing_date": pl.Date, "cpc_group": pl.Utf8})
    return _aggregate_physical_ai(raw)


def save_physical_ai_filings(out_dir: Path, agg: pl.DataFrame) -> None:
    """Write one parquet per cpc_class bucket. Schema: (quarter_end, cpc_class, filing_count)."""
    if agg.is_empty():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for bucket in _PHYSICAL_AI_BUCKETS:
        sub = agg.filter(pl.col("cpc_class") == bucket)
        if sub.is_empty():
            continue
        bucket_dir = out_dir / f"cpc_class={bucket}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(bucket_dir / "filings.parquet", compression="snappy")


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
        try:
            resp = requests.post(_APPS_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — fail-soft per project convention
            _LOG.warning("[USPTO] applications page=%d fetch failed (%s) — stopping", page, exc)
            break
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
        try:
            resp = requests.post(_GRANTS_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — fail-soft per project convention
            _LOG.warning("[USPTO] grants page=%d fetch failed (%s) — stopping", page, exc)
            break
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
    _ROOT = Path(__file__).parent.parent
    date_str = datetime.date.today().isoformat()
    base = Path("data/raw/patents")
    ingest_uspto(date_str, base / "applications", base / "grants")

    today = datetime.date.today().isoformat()
    physical_ai_dir = _ROOT / "data" / "raw" / "uspto" / "physical_ai"
    _LOG.info("Fetching physical-AI patent filings (6 CPC buckets)...")
    agg = fetch_physical_ai_filings(today)
    save_physical_ai_filings(physical_ai_dir, agg)
    _LOG.info("[USPTO] physical_ai: %d (quarter, bucket) rows written", len(agg))
