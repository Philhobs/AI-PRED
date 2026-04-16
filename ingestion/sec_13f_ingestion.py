# ingestion/sec_13f_ingestion.py
"""
SEC EDGAR 13F-HR Institutional Ownership Ingestion

Downloads quarterly 13F-HR filings for the top 500 institutional filers,
filters holdings to our 80-ticker CUSIP watchlist, saves as Parquet.

Storage layout:
  data/raw/financials/13f_holdings/raw/<YYYYQQ>/<CIK>.parquet  ← per-filer per-quarter
  data/raw/financials/cusip_map.json                            ← ticker→CUSIP lookup

Run:
  python ingestion/sec_13f_ingestion.py              # current + prior quarter
  python ingestion/sec_13f_ingestion.py --bootstrap  # full history from 2013-Q1
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests

_LOG = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ai-infra-predictor research@example.com"}
_SLEEP = 0.11  # stay well under SEC 10 req/s fair-use limit

_RAW_SCHEMA = pa.schema([
    pa.field("cik",                  pa.string()),
    pa.field("quarter",              pa.string()),
    pa.field("period_end",           pa.date32()),
    pa.field("cusip",                pa.string()),
    pa.field("ticker",               pa.string()),
    pa.field("shares_held",          pa.int64()),
    pa.field("value_usd_thousands",  pa.int64()),
])


# ── XML helpers ───────────────────────────────────────────────────────────────

def _strip_ns(xml_str: str) -> str:
    """Remove XML namespace declarations so ElementTree finds tags by local name."""
    return re.sub(r'\s+xmlns(?::[^=]+)?="[^"]*"', "", xml_str)


def parse_holdings_xml(xml_str: str, cusip_map: dict[str, str]) -> list[dict]:
    """
    Parse a 13F-HR information table XML string.

    Returns list of dicts with keys: cusip, ticker, shares_held, value_usd_thousands.
    Filters to:
    - CUSIPs present in cusip_map (our watchlist)
    - sshPrnamtType == "SH" (shares only, not bonds/principal)
    """
    cusip_to_ticker = {v: k for k, v in cusip_map.items()}

    try:
        root = ET.fromstring(_strip_ns(xml_str))
    except ET.ParseError:
        return []

    records = []
    for info_table in root.iter("infoTable"):
        cusip_el  = info_table.find("cusip")
        type_el   = info_table.find(".//sshPrnamtType")
        shares_el = info_table.find(".//sshPrnamt")
        value_el  = info_table.find("value")

        if any(el is None for el in (cusip_el, type_el, shares_el, value_el)):
            continue
        if (type_el.text or "").strip() != "SH":
            continue

        cusip = (cusip_el.text or "").strip()
        if cusip not in cusip_to_ticker:
            continue

        try:
            shares = int((shares_el.text or "0").strip().replace(",", ""))
            value  = int((value_el.text or "0").strip().replace(",", ""))
        except ValueError:
            continue

        records.append({
            "cusip":               cusip,
            "ticker":              cusip_to_ticker[cusip],
            "shares_held":         shares,
            "value_usd_thousands": value,
        })

    return records


# ── Quarter index ─────────────────────────────────────────────────────────────

def _parse_index_content(content: str) -> pl.DataFrame:
    """
    Parse EDGAR full-index company.gz content (after decompression).

    Format: pipe-separated lines — Company Name|Form Type|CIK|Date Filed|Filename
    Returns DataFrame with [cik, date_filed, filename] for 13F-HR rows only.
    CIK is zero-padded to 10 digits.
    """
    records = []
    for line in content.splitlines():
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        form_type = parts[1].strip()
        if form_type != "13F-HR":
            continue
        records.append({
            "cik":        parts[2].strip().zfill(10),
            "date_filed": parts[3].strip(),
            "filename":   parts[4].strip(),
        })

    if not records:
        return pl.DataFrame({"cik": [], "date_filed": [], "filename": []},
                            schema={"cik": pl.Utf8, "date_filed": pl.Utf8, "filename": pl.Utf8})
    return pl.DataFrame(records)


def fetch_quarter_index(year: int, quarter: int) -> pl.DataFrame:
    """
    Download EDGAR full-index for one quarter, return 13F-HR filings DataFrame.

    URL: https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.gz
    Returns DataFrame with columns: cik, date_filed, filename.
    Raises requests.HTTPError on non-200 response.
    """
    url = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"{year}/QTR{quarter}/company.gz"
    )
    _LOG.info("Fetching quarter index: %s Q%s", year, quarter)
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    time.sleep(_SLEEP)
    resp.raise_for_status()

    content = gzip.decompress(resp.content).decode("latin-1")
    return _parse_index_content(content)


def rank_filers_by_position_count(index_df: pl.DataFrame, top_n: int = 500) -> list[str]:
    """
    Return the top_n filer CIKs from a quarter index DataFrame.

    Proxy for AUM: lower CIK integer value = older EDGAR registrant = typically
    a larger, more established institution (Vanguard=102909, Fidelity=315066).
    Returns list of CIK strings (zero-padded to 10 digits), sorted ascending by CIK int.
    """
    sorted_df = index_df.with_columns(
        pl.col("cik").cast(pl.UInt64).alias("cik_int")
    ).sort("cik_int")
    return sorted_df["cik"].head(top_n).to_list()
