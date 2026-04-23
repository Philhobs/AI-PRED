"""BLS JOLTS tech sector job openings ingestion.

Fetches Computer & Electronic Products (NAICS 334) job openings from BLS API v2.
Series: JTS510000000000000JOL (job openings level, thousands, seasonally adjusted)
Output: data/raw/bls_jolts/date=YYYY-MM-DD/openings.parquet

Staleness guard: skips re-download if existing snapshot is from the same calendar month
(BLS JOLTS publishes monthly data with ~6-week publication lag).
"""
from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_SERIES_ID = "JTS510000000000000JOL"

_SCHEMA = {
    "date": pl.Date,
    "series_id": pl.Utf8,
    "year": pl.Int32,
    "period": pl.Utf8,
    "value": pl.Float64,
}


def _same_month(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent parquet snapshot was taken in the same calendar month as today."""
    files = sorted(existing_dir.glob("date=*/openings.parquet"))
    if not files:
        return False
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        return last_date.year == today.year and last_date.month == today.month
    except ValueError:
        return False


def fetch_jolts(date_str: str) -> pl.DataFrame:
    """Fetch BLS JOLTS tech sector openings for 12-month window ending date_str.

    Returns DataFrame with _SCHEMA. Empty DataFrame if no results.
    """
    today = datetime.date.fromisoformat(date_str)
    payload: dict = {
        "seriesid": [_SERIES_ID],
        "startyear": str(today.year - 1),
        "endyear": str(today.year),
    }
    api_key = os.environ.get("BLS_API_KEY", "")
    if api_key:
        payload["registrationkey"] = api_key

    resp = requests.post(_BLS_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    series_list = data.get("Results", {}).get("series", [])
    if not series_list:
        return pl.DataFrame(schema=_SCHEMA)

    rows = []
    for entry in series_list[0].get("data", []):
        period = entry.get("period", "")
        if not (period.startswith("M") and period != "M13"):
            continue
        try:
            year = int(entry["year"])
            value = float(entry["value"])
        except (KeyError, ValueError, TypeError):
            continue
        rows.append({
            "date": today,
            "series_id": _SERIES_ID,
            "year": year,
            "period": period,
            "value": value,
        })

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)

    return pl.DataFrame(rows, schema=_SCHEMA)


def ingest_bls_jolts(date_str: str, output_dir: Path) -> None:
    """Fetch and persist BLS JOLTS openings for date_str.

    Skips download if same-month snapshot exists. Writes nothing when results are empty.
    """
    if _same_month(output_dir, date_str):
        _LOG.info("BLS JOLTS: same month snapshot exists — skipping for %s", date_str)
        return

    _LOG.info("BLS JOLTS: fetching tech sector job openings for %s", date_str)
    df = fetch_jolts(date_str)
    if not df.is_empty():
        out = output_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "openings.parquet", compression="snappy")
        _LOG.info("BLS JOLTS: wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ingest_bls_jolts(
        datetime.date.today().isoformat(),
        Path("data/raw/bls_jolts"),
    )
