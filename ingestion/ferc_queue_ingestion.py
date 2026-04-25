"""
Ingest FERC interconnection queue data from Lawrence Berkeley National Lab.

Source: LBL "Queued Up" semi-annual Excel dataset
URL: https://emp.lbl.gov/sites/default/files/queued_up.xlsx
     Override via FERC_QUEUE_URL environment variable.

Raw storage: data/raw/ferc_queue/date=YYYY-MM-DD/queue.parquet
Schema: (snapshot_date, queue_date, project_name, mw, state, fuel, status, iso)
Filtered to DC power states only: VA, TX, OH, AZ, NV, OR, GA, WA

Staleness: LBL updates in January and July. Re-download is skipped when an existing
parquet has a snapshot_date in the same half-year (Jan–Jun or Jul–Dec).

Usage:
    python ingestion/ferc_queue_ingestion.py               # snapshot today
    python ingestion/ferc_queue_ingestion.py --date 2024-01-15
"""
from __future__ import annotations

import argparse
import datetime
import io
import logging
import os
from pathlib import Path

import pandas as pd
import requests
import polars as pl

_LOG = logging.getLogger(__name__)

_FERC_QUEUE_DEFAULT_URL = "https://emp.lbl.gov/sites/default/files/queued_up.xlsx"

_DC_STATES = {"VA", "TX", "OH", "AZ", "NV", "OR", "GA", "WA"}

_FERC_SCHEMA = {
    "snapshot_date": pl.Date,
    "queue_date": pl.Date,
    "project_name": pl.Utf8,
    "mw": pl.Float64,
    "state": pl.Utf8,
    "fuel": pl.Utf8,
    "status": pl.Utf8,
    "iso": pl.Utf8,
}

# Flexible column mapping — LBL uses slightly different names across versions
_COLUMN_MAP: dict[str, str] = {
    "Queue Date": "queue_date",
    "Date Entered Queue": "queue_date",
    "Project Name": "project_name",
    "Project": "project_name",
    "MW": "mw",
    "MW (DC)": "mw",
    "Capacity (MW)": "mw",
    "State": "state",
    "State/Province/Territory": "state",
    "Fuel": "fuel",
    "Fuel Type": "fuel",
    "Status": "status",
    "Interconnection Queue Status": "status",
    "ISO": "iso",
    "ISO/RTO": "iso",
    "Balancing Authority": "iso",
}


def _is_same_half_year(d1: datetime.date, d2: datetime.date) -> bool:
    """Return True if both dates fall in the same half-year."""
    def _half(d: datetime.date) -> tuple[int, int]:
        return (d.year, 1 if d.month <= 6 else 2)
    return _half(d1) == _half(d2)


def _parse_excel(content: bytes, snapshot_date: datetime.date) -> pl.DataFrame:
    """Parse LBL Excel bytes into a DC-state-filtered DataFrame."""
    try:
        raw = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse FERC queue Excel: {exc}") from exc

    if raw.empty:
        return pl.DataFrame(schema=_FERC_SCHEMA)

    # Apply column mapping — first match wins for each target column
    rename_map: dict[str, str] = {}
    target_used: set[str] = set()
    for src_col in raw.columns:
        target = _COLUMN_MAP.get(str(src_col))
        if target and target not in target_used:
            rename_map[src_col] = target
            target_used.add(target)

    raw = raw.rename(columns=rename_map)

    for col in ["queue_date", "project_name", "mw", "state", "fuel", "status", "iso"]:
        if col not in raw.columns:
            raw[col] = "" if col != "mw" else 0.0

    # Filter to DC power states only
    raw["state"] = raw["state"].astype(str).str.strip().str.upper()
    raw = raw[raw["state"].isin(_DC_STATES)].copy()
    if raw.empty:
        return pl.DataFrame(schema=_FERC_SCHEMA)

    raw["snapshot_date"] = snapshot_date
    raw["queue_date"] = pd.to_datetime(raw["queue_date"], errors="coerce").dt.date
    raw["mw"] = pd.to_numeric(raw["mw"], errors="coerce").fillna(0.0)
    for col in ["project_name", "state", "fuel", "status", "iso"]:
        raw[col] = raw[col].fillna("").astype(str)

    df = pl.from_pandas(
        raw[["snapshot_date", "queue_date", "project_name",
             "mw", "state", "fuel", "status", "iso"]]
    )
    return df.cast(_FERC_SCHEMA)


def ingest_ferc_queue(
    date_str: str,
    output_dir: Path,
    ferc_url: str | None = None,
) -> None:
    """Download LBL FERC queue Excel and write Hive-partitioned parquet.

    Skips download if existing data is from the same half-year (Jan/Jul update cycle).
    """
    url = ferc_url or os.environ.get("FERC_QUEUE_URL", _FERC_QUEUE_DEFAULT_URL)
    snapshot_date = datetime.date.fromisoformat(date_str)

    # Staleness check
    existing_files = sorted(output_dir.glob("date=*/queue.parquet"))
    if existing_files:
        try:
            existing = pl.read_parquet(existing_files[-1])
            if not existing.is_empty() and "snapshot_date" in existing.columns:
                existing_snap = existing["snapshot_date"][0]
                if _is_same_half_year(existing_snap, snapshot_date):
                    _LOG.info("FERC queue is current for %s half-year — skipping", date_str)
                    return
        except Exception:
            pass  # Re-download on any read error

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — fail-soft per project convention
        _LOG.warning("[FERC] download failed (%s) — skipping this snapshot", exc)
        return

    df = _parse_excel(resp.content, snapshot_date)

    if df.is_empty():
        _LOG.debug("No DC-state FERC queue entries for %s — skipping", date_str)
        return

    out_dir = output_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "queue.parquet", compression="snappy")
    _LOG.info("Wrote %d FERC queue entries for %s", len(df), date_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest FERC interconnection queue from LBL")
    parser.add_argument(
        "--date",
        default=str(datetime.date.today()),
        help="Snapshot date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw/ferc_queue")
    _LOG.info("Fetching FERC queue snapshot for %s", args.date)
    ingest_ferc_queue(args.date, output_dir)
