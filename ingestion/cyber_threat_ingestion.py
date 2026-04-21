"""Cyber threat data ingestion from free public sources.

Fetches CVEs (NVD), known exploited vulnerabilities (CISA), and threat pulses
(AlienVault OTX) for a date range and writes Hive-partitioned parquet files:

    data/raw/cyber_threat/date=YYYY-MM-DD/threats.parquet

Schema: date (Date), source (Utf8), metric (Utf8), value (Float64)

Metrics produced:
    cve_critical  — CVSS >= 9.0 vulnerability published (NVD)
    cve_high      — 7.0 <= CVSS < 9.0 vulnerability published (NVD)
    cisa_kev      — CISA known exploited vulnerability added (CISA)
    otx_pulse     — AlienVault OTX threat pulse created (OTX)

Adding a paid source: implement CyberThreatSource.fetch() in a new class
and pass it to ingest_cyber_threats(sources=[...]).

Usage:
    python ingestion/cyber_threat_ingestion.py             # last 30 days
    python ingestion/cyber_threat_ingestion.py --start 2024-01-01 --end 2024-03-31
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path
from typing import Protocol

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_NVD_URL  = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_CISA_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_OTX_URL  = "https://otx.alienvault.com/api/v1/pulses/subscribed"

_SCHEMA = {"date": pl.Date, "source": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def _get_cvss_score(cve: dict) -> float | None:
    """Extract the highest available CVSS base score from a CVE object."""
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            return entries[0].get("cvssData", {}).get("baseScore")
    return None


class CyberThreatSource(Protocol):
    """Protocol for pluggable threat data sources."""

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Return rows with schema (date Date, source Utf8, metric Utf8, value Float64)."""
        ...


class NVDSource:
    """NIST NVD CVE API v2 — no API key required (rate limited to 5 req/30s)."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("NVD_API_KEY")

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        rows: list[dict] = []
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)

        # NVD accepts at most 120-day windows; chunk to 30 days for safety
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + datetime.timedelta(days=29), end)
            s_str = f"{chunk_start.isoformat()}T00:00:00.000"
            e_str = f"{chunk_end.isoformat()}T23:59:59.999"

            start_idx = 0
            while True:
                params: dict = {
                    "pubStartDate": s_str,
                    "pubEndDate": e_str,
                    "resultsPerPage": 2000,
                    "startIndex": start_idx,
                }
                if self._api_key:
                    params["apiKey"] = self._api_key

                try:
                    resp = requests.get(_NVD_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    _LOG.warning("[NVD] Fetch failed: %s", exc)
                    break

                vulns = data.get("vulnerabilities", [])
                total = data.get("totalResults", 0)

                for v in vulns:
                    cve = v.get("cve", {})
                    published = (cve.get("published") or "")[:10]
                    if not published:
                        continue
                    score = _get_cvss_score(cve)
                    if score is None or score < 7.0:
                        continue
                    metric = "cve_critical" if score >= 9.0 else "cve_high"
                    rows.append({
                        "date": datetime.date.fromisoformat(published),
                        "source": "nvd",
                        "metric": metric,
                        "value": 1.0,
                    })

                start_idx += len(vulns)
                # Respect rate limit: 5 req/30s without key, 50 req/30s with key
                time.sleep(1 if self._api_key else 6)
                if start_idx >= total or not vulns:
                    break

            chunk_start = chunk_end + datetime.timedelta(days=1)

        return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


class CISASource:
    """CISA Known Exploited Vulnerabilities feed — no key required."""

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)

        try:
            resp = requests.get(_CISA_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            _LOG.warning("[CISA] Fetch failed: %s", exc)
            return _empty()

        rows: list[dict] = []
        for entry in data.get("vulnerabilities", []):
            date_str = entry.get("dateAdded", "")
            if not date_str:
                continue
            try:
                d = datetime.date.fromisoformat(date_str)
            except ValueError:
                continue
            if not (start <= d <= end):
                continue
            rows.append({
                "date": d,
                "source": "cisa",
                "metric": "cisa_kev",
                "value": 1.0,
            })

        return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


class OTXSource:
    """AlienVault Open Threat Exchange — free API key required (OTX_API_KEY in .env).

    Degrades gracefully to empty DataFrame when key is absent.
    Register free at https://otx.alienvault.com
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("OTX_API_KEY")

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        if not self._api_key:
            _LOG.debug("[OTX] OTX_API_KEY not set — skipping OTX source")
            return _empty()

        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)
        headers = {"X-OTX-API-KEY": self._api_key}
        rows: list[dict] = []
        url: str | None = _OTX_URL

        while url:
            params = {"modified_since": f"{start_date}T00:00:00", "limit": 50}
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                _LOG.warning("[OTX] Fetch failed: %s", exc)
                break

            for pulse in data.get("results", []):
                created_str = (pulse.get("created") or "")[:10]
                if not created_str:
                    continue
                try:
                    d = datetime.date.fromisoformat(created_str)
                except ValueError:
                    continue
                if not (start <= d <= end):
                    continue
                rows.append({
                    "date": d,
                    "source": "otx",
                    "metric": "otx_pulse",
                    "value": 1.0,
                })

            url = data.get("next")
            time.sleep(1)

        return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


def ingest_cyber_threats(
    start_date: str,
    end_date: str,
    output_dir: Path = Path("data/raw/cyber_threat"),
    sources: list | None = None,
) -> None:
    """Fetch threat events from all sources and write per-date parquet files.

    output_dir: root of Hive partition tree, e.g. data/raw/cyber_threat/
    sources: override list of CyberThreatSource instances (default: NVD + CISA + OTX)
    """
    if sources is None:
        sources = [NVDSource(), CISASource(), OTXSource()]

    frames: list[pl.DataFrame] = []
    for source in sources:
        df = source.fetch(start_date, end_date)
        if not df.is_empty():
            frames.append(df)

    if not frames:
        _LOG.info("[CyberThreat] No threat data fetched for %s–%s", start_date, end_date)
        return

    combined = pl.concat(frames)

    # Write one parquet file per date partition
    for date_val, group in combined.group_by("date"):
        date_str = str(date_val[0]) if isinstance(date_val, tuple) else str(date_val)
        out_path = output_dir / f"date={date_str}" / "threats.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        group.write_parquet(out_path, compression="snappy")
        _LOG.info("[CyberThreat] Wrote %d rows → %s", len(group), out_path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest cyber threat data.")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    today = datetime.date.today()
    end   = datetime.date.fromisoformat(args.end)   if args.end   else today
    start = datetime.date.fromisoformat(args.start) if args.start else today - datetime.timedelta(days=30)

    ingest_cyber_threats(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        output_dir=Path("data/raw/cyber_threat"),
    )
