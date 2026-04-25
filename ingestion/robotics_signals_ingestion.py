"""FRED-based robotics macro signals ingestion.

Fetches 4 FRED series tracking the macro demand backdrop for industrial
automation and physical-AI growth:
  NEWORDER  — Manufacturers' New Orders: Nondefense Capital Goods Ex Aircraft
  NAPM      — ISM Manufacturing PMI (headline)
  IPG3331S  — Industrial Production: Industrial Machinery
  WPU114    — PPI: Industrial Machinery

Output: data/raw/robotics_signals/{series_id}.parquet
Schema: date (Date), value (Float64).

REQUIRES FRED_API_KEY in .env — register free at
https://fred.stlouisfed.org/docs/api/api_key.html. Without a key, every
request returns HTTP 400 Bad Request. Module fail-softs (writes empty parquet,
logs warning) when the key is missing or any other error occurs.
FRED returns '.' for missing observations — converted to null.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_SERIES: tuple[str, ...] = ("NEWORDER", "NAPM", "IPG3331S", "WPU114")
_NAPM_FALLBACK = "USAPMI"  # if NAPM is unreachable, try this alternative

_SCHEMA = {"date": pl.Date, "value": pl.Float64}
_EMPTY_DF = pl.DataFrame(schema=_SCHEMA)


def fetch_fred_series(series_id: str, observation_start: str = "2010-01-01") -> pl.DataFrame:
    """Fetch a single FRED series. Returns empty DataFrame on failure."""
    api_key = os.getenv("FRED_API_KEY", "")
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": observation_start,
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(_FRED_BASE, params=params, timeout=30)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
    except Exception as exc:  # noqa: BLE001 — fail soft per spec
        _LOG.warning("[FRED] %s: fetch failed (%s); returning empty", series_id, exc)
        return _EMPTY_DF.clone()

    if not observations:
        return _EMPTY_DF.clone()

    rows = []
    for obs in observations:
        try:
            d = date.fromisoformat(obs["date"])
        except (KeyError, ValueError):
            continue
        raw_val = obs.get("value", ".")
        val: float | None
        if raw_val == "." or raw_val is None:
            val = None
        else:
            try:
                val = float(raw_val)
            except ValueError:
                val = None
        rows.append({"date": d, "value": val})

    if not rows:
        return _EMPTY_DF.clone()
    return pl.DataFrame(rows, schema=_SCHEMA)


def save_robotics_signals(out_dir: Path, series_dfs: dict[str, pl.DataFrame]) -> None:
    """Write one snappy-compressed parquet per non-empty series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for series_id, df in series_dfs.items():
        if df.is_empty():
            _LOG.warning("[FRED] %s: empty DataFrame — skipping write", series_id)
            continue
        df.write_parquet(out_dir / f"{series_id}.parquet", compression="snappy")


def fetch_all() -> dict[str, pl.DataFrame]:
    """Fetch all 4 FRED series (with NAPM fallback to USAPMI). Sleep 1s between calls."""
    out: dict[str, pl.DataFrame] = {}
    for series_id in _FRED_SERIES:
        df = fetch_fred_series(series_id)
        if df.is_empty() and series_id == "NAPM":
            _LOG.warning("[FRED] NAPM unreachable — trying fallback %s", _NAPM_FALLBACK)
            df = fetch_fred_series(_NAPM_FALLBACK)
        out[series_id] = df
        time.sleep(1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ROOT = Path(__file__).parent.parent
    out_dir = _ROOT / "data" / "raw" / "robotics_signals"
    _LOG.info("Fetching %d FRED series...", len(_FRED_SERIES))
    rates = fetch_all()
    save_robotics_signals(out_dir, rates)
    for sid, df in rates.items():
        _LOG.info("  %s: %d rows", sid, len(df))
