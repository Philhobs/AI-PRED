"""
EIA capacity + PJM interconnection queue ingestion.

Sources:
  - EIA API v2: monthly US electricity generation capacity by fuel type
  - PJM: Virginia corridor interconnection queue backlog

Usage:
    python ingestion/eia_ingestion.py

Requires EIA_API_KEY in .env (register free at https://www.eia.gov/opendata/)
"""
from __future__ import annotations

import io
import logging
import os
import time
from datetime import date
from pathlib import Path

import openpyxl
import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_EIA_URL = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
_PJM_URL = (
    "https://www.pjm.com/-/media/planning/rtep/rtep-documents/"
    "active-interconnection-requests-report.ashx"
)
_VIRGINIA_ZONES = {"MAAC", "AECO", "SWVA"}
_FUEL_TYPE_MAP = {
    "NUC": "nuclear",
    "NG": "natural_gas",
    "SUN": "solar",
    "WND": "wind",
}


def fetch_eia_capacity(api_key: str, months: int = 36) -> pl.DataFrame:
    """
    Fetch monthly US installed capacity by fuel type from EIA API v2.

    Returns DataFrame with columns: date (date), fuel_type (str), capacity_gw (float64).
    """
    _EMPTY = pl.DataFrame(schema={"date": pl.Date, "fuel_type": pl.Utf8, "capacity_gw": pl.Float64})
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[]": "capacity",
        "facets[fueltypeid][]": list(_FUEL_TYPE_MAP.keys()),
        "facets[sectorid][]": "99",  # all sectors
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": str(months * len(_FUEL_TYPE_MAP) + 10),
    }

    try:
        resp = requests.get(_EIA_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        _LOG.warning("[EIA] Fetch failed — %s. Skipping.", exc)
        return _EMPTY
    records = resp.json().get("response", {}).get("data", [])

    rows = []
    for r in records:
        fuel_id = r.get("fueltypeid", "")
        if fuel_id not in _FUEL_TYPE_MAP:
            continue
        period = r.get("period", "")
        cap = r.get("capacity")
        if not period or cap is None:
            continue
        # EIA returns capacity in gigawatts
        rows.append({
            "date": date.fromisoformat(period + "-01"),
            "fuel_type": _FUEL_TYPE_MAP[fuel_id],
            "capacity_gw": float(cap),
        })

    if not rows:
        _LOG.warning("[EIA] No capacity records returned — check API key and params")
        return pl.DataFrame(schema={"date": pl.Date, "fuel_type": pl.Utf8, "capacity_gw": pl.Float64})

    return pl.DataFrame(rows).sort("date", descending=True)


def fetch_pjm_queue() -> pl.DataFrame:
    """
    Fetch PJM interconnection queue, filter to Virginia corridor zones.

    Returns DataFrame with columns:
      date (date), zone (str), queue_backlog_gw (float64), project_count (int32)
    """
    _EMPTY = pl.DataFrame(
        schema={"date": pl.Date, "zone": pl.Utf8, "queue_backlog_gw": pl.Float64, "project_count": pl.Int32}
    )

    try:
        resp = requests.get(_PJM_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        _LOG.warning("[PJM] Fetch failed — %s. Skipping.", exc)
        return _EMPTY

    wb = openpyxl.load_workbook(io.BytesIO(resp.content), read_only=True, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c else "" for c in next(rows_iter)]

    def _find_col(candidates: list[str]) -> int | None:
        for name in candidates:
            for i, h in enumerate(header):
                if name.lower() in h.lower():
                    return i
        return None

    zone_col   = _find_col(["Zone", "LDA", "Zone Name"])
    mw_col     = _find_col(["MW", "Capacity MW", "Interconnection Request MW", "MW Requested"])
    status_col = _find_col(["Status", "Queue Status", "Interconnection Status"])

    if any(c is None for c in [zone_col, mw_col, status_col]):
        _LOG.warning(
            "[PJM] Could not find expected columns in queue report. Header: %s", header
        )
        return pl.DataFrame(
            schema={"date": pl.Date, "zone": pl.Utf8, "queue_backlog_gw": pl.Float64, "project_count": pl.Int32}
        )

    virginia_mw = 0.0
    project_count = 0
    for row in rows_iter:
        zone   = str(row[zone_col]).strip() if row[zone_col] else ""
        status = str(row[status_col]).strip() if row[status_col] else ""
        mw_val = row[mw_col]

        if zone not in _VIRGINIA_ZONES:
            continue
        if "withdrawn" in status.lower():
            continue
        if mw_val is None:
            continue

        try:
            virginia_mw += float(mw_val)
            project_count += 1
        except (ValueError, TypeError):
            continue

    wb.close()

    today = date.today()
    return pl.DataFrame([{
        "date": today,
        "zone": "ALL_VIRGINIA",
        "queue_backlog_gw": round(virginia_mw / 1000.0, 3),
        "project_count": project_count,
    }]).with_columns(pl.col("project_count").cast(pl.Int32))


def save_eia_capacity(df: pl.DataFrame, output_dir: Path) -> None:
    """Append-and-deduplicate EIA capacity records."""
    out_path = output_dir / "eia_capacity.parquet"
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        df = pl.concat([existing, df]).unique(subset=["date", "fuel_type"], keep="last")
    output_dir.mkdir(parents=True, exist_ok=True)  # always ensure dir exists
    df.sort("date", descending=True).write_parquet(out_path, compression="snappy")
    _LOG.info("[EIA] Saved %d capacity records to %s", len(df), out_path)


def save_pjm_queue(df: pl.DataFrame, output_dir: Path) -> None:
    """Append-and-deduplicate PJM queue records."""
    out_path = output_dir / "pjm_queue.parquet"
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        df = pl.concat([existing, df]).unique(subset=["date", "zone"], keep="last")
    output_dir.mkdir(parents=True, exist_ok=True)  # always ensure dir exists
    df.sort("date", descending=True).write_parquet(out_path, compression="snappy")
    _LOG.info("[PJM] Saved %d queue records to %s", len(df), out_path)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        _LOG.error("EIA_API_KEY not set in .env — skipping EIA capacity fetch")
        sys.exit(1)

    output_dir = Path(__file__).parent.parent / "data" / "raw" / "energy"

    _LOG.info("Fetching EIA capacity data...")
    eia_df = fetch_eia_capacity(api_key)
    save_eia_capacity(eia_df, output_dir)
    time.sleep(1)

    _LOG.info("Fetching PJM queue data...")
    pjm_df = fetch_pjm_queue()
    save_pjm_queue(pjm_df, output_dir)

    _LOG.info("Done. Run python processing/energy_geo_features.py to compute features.")
