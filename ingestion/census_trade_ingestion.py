"""Census international trade data ingestion.

Fetches US semiconductor and data center equipment import/export values
from the Census International Trade API (timeseries/intltrade).

Queries (10 per run):
  Imports, all partners: HS 8541, 8542, 8471, 8473
  Imports, Taiwan (CTY=5830): HS 8541, 8542
  Exports, all partners: HS 8541, 8542
  Exports, China (CTY=5700): HS 8541, 8542

Output: data/raw/census_trade/date=YYYY-MM-DD/trade.parquet

Staleness guard: skips re-download if existing snapshot is from the same calendar month.
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

_IMPORTS_URL = "https://api.census.gov/data/timeseries/intltrade/imports"
_EXPORTS_URL = "https://api.census.gov/data/timeseries/intltrade/exports"

_SCHEMA = {
    "date": pl.Date,
    "direction": pl.Utf8,
    "hs_code": pl.Utf8,
    "partner_code": pl.Utf8,
    "year": pl.Int32,
    "month": pl.Int32,
    "value_usd": pl.Float64,
}

_SLEEP_BETWEEN_QUERIES = 0.5

# (direction, hs_code, partner_code) — "ALL" means omit CTY_CODE parameter
_QUERIES: list[tuple[str, str, str]] = [
    ("import", "8541", "ALL"),
    ("import", "8542", "ALL"),
    ("import", "8471", "ALL"),
    ("import", "8473", "ALL"),
    ("import", "8541", "5830"),  # Taiwan semiconductor imports
    ("import", "8542", "5830"),  # Taiwan integrated circuit imports
    ("export", "8541", "ALL"),
    ("export", "8542", "ALL"),
    ("export", "8541", "5700"),  # China semiconductor exports
    ("export", "8542", "5700"),  # China integrated circuit exports
]


def _same_month(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent parquet snapshot was taken in the same calendar month as today."""
    files = sorted(existing_dir.glob("date=*/trade.parquet"))
    if not files:
        return False
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        return last_date.year == today.year and last_date.month == today.month
    except ValueError:
        return False


def _fetch_query(
    direction: str,
    hs_code: str,
    partner_code: str,
    run_date: datetime.date,
    api_key: str,
) -> list[dict]:
    """Fetch 12-month lookback for one (direction, hs_code, partner_code) combination."""
    url = _IMPORTS_URL if direction == "import" else _EXPORTS_URL
    value_field = "GEN_VAL_MO" if direction == "import" else "ALL_VAL_MO"

    from_str = f"{run_date.year - 1}-{run_date.month:02d}"
    to_str = f"{run_date.year}-{run_date.month:02d}"

    params: dict = {
        "get": f"{value_field},E_COMMODITY",
        "COMM_LVL": "HS4",
        "E_COMMODITY": hs_code,
        "time": f"from+{from_str}+to+{to_str}",
    }
    if partner_code != "ALL":
        params["CTY_CODE"] = partner_code
    if api_key:
        params["key"] = api_key

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data or len(data) < 2:
        return []

    headers = data[0]
    try:
        val_idx = headers.index(value_field)
        time_idx = headers.index("time")
    except ValueError:
        _LOG.warning(
            "Census: unexpected headers %s for %s %s %s", headers, direction, hs_code, partner_code
        )
        return []

    rows = []
    for row in data[1:]:
        try:
            time_str = row[time_idx]   # "YYYY-MM"
            year = int(time_str[:4])
            month = int(time_str[5:7])
            value = float(row[val_idx])
        except (ValueError, TypeError, IndexError):
            continue
        rows.append({
            "date": run_date,
            "direction": direction,
            "hs_code": hs_code,
            "partner_code": partner_code,
            "year": year,
            "month": month,
            "value_usd": value,
        })

    return rows


def fetch_trade(date_str: str) -> pl.DataFrame:
    """Fetch all Census trade queries for date_str.

    Returns DataFrame with _SCHEMA. Empty DataFrame if no results.
    """
    run_date = datetime.date.fromisoformat(date_str)
    api_key = os.environ.get("CENSUS_API_KEY", "")

    all_rows: list[dict] = []
    for i, (direction, hs_code, partner_code) in enumerate(_QUERIES):
        all_rows.extend(_fetch_query(direction, hs_code, partner_code, run_date, api_key))
        if i < len(_QUERIES) - 1:
            time.sleep(_SLEEP_BETWEEN_QUERIES)

    if not all_rows:
        return pl.DataFrame(schema=_SCHEMA)

    return pl.DataFrame(all_rows, schema=_SCHEMA)


def ingest_census_trade(date_str: str, output_dir: Path) -> None:
    """Fetch and persist Census trade data for date_str.

    Skips download if same-month snapshot exists. Writes nothing when results are empty.
    """
    if _same_month(output_dir, date_str):
        _LOG.info("Census trade: same month snapshot exists — skipping for %s", date_str)
        return

    _LOG.info("Census trade: fetching semiconductor + DC equipment trade data for %s", date_str)
    df = fetch_trade(date_str)
    if df.is_empty():
        _LOG.info("Census trade: API returned no data rows for %s — nothing written", date_str)
    else:
        out = output_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "trade.parquet", compression="snappy")
        _LOG.info("Census trade: wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ingest_census_trade(
        datetime.date.today().isoformat(),
        Path("data/raw/census_trade"),
    )
