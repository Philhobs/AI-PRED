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
    except ET.ParseError as exc:
        _LOG.debug("Failed to parse holdings XML: %s", exc)
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

    Note: The EDGAR full-index does not include AUM or position counts per filer.
    CIK age is used as a best-available heuristic. Vanguard=102909, Fidelity=315066.
    Large institutions registered after ~2010 may be ranked lower than their AUM warrants.
    """
    sorted_df = index_df.with_columns(
        pl.col("cik").cast(pl.UInt64).alias("cik_int")
    ).sort("cik_int")
    return sorted_df["cik"].head(top_n).to_list()


# ── Filing XML fetcher ────────────────────────────────────────────────────────

def _quarter_end_date(year: int, quarter: int) -> str:
    """Return ISO quarter-end date string for a given year/quarter."""
    if not 1 <= quarter <= 4:
        raise ValueError(f"quarter must be 1-4, got {quarter!r}")
    ends = {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}
    return ends[quarter]


def _fetch_filing_index_html(filename: str) -> str | None:
    """Fetch the HTML filing index page for a 13F-HR submission."""
    url = f"https://www.sec.gov/{filename}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        time.sleep(_SLEEP)
        if resp.status_code != 200:
            _LOG.debug("[13F] Filing index returned %s for %s", resp.status_code, url)
            return None
        return resp.text
    except requests.RequestException:
        return None


def _find_xml_url(index_html: str, cik: str, filename: str) -> str | None:
    """
    Extract the information table XML URL from a 13F-HR filing index HTML.

    Looks for .xml links in the HTML. Returns the first XML URL found.
    """
    cik_num = str(int(cik))  # strip leading zeros for URL path
    accession_nodash = filename.split("/")[-1].replace("-index.htm", "").replace("-", "")
    base = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession_nodash}"
    )

    xml_paths = re.findall(r'href="([^"]*\.xml)"', index_html, re.IGNORECASE)
    for path in xml_paths:
        if path.startswith("http"):
            url = path
        elif path.startswith("/"):
            url = f"https://www.sec.gov{path}"
        else:
            url = f"{base}/{path}"
        return url  # return first XML link found

    return None


def fetch_filing_xml(cik: str, filename: str) -> str | None:
    """
    Fetch the 13F-HR information table XML for one filer filing.

    Returns XML string if successful and contains infoTable elements, None otherwise.
    """
    html = _fetch_filing_index_html(filename)
    if html is None:
        return None

    xml_url = _find_xml_url(html, cik, filename)
    if xml_url is None:
        return None

    try:
        resp = requests.get(xml_url, headers=_HEADERS, timeout=30)
        time.sleep(_SLEEP)
        if resp.status_code != 200:
            return None
        xml = resp.text
        if "infoTable" not in xml and "informationTable" not in xml:
            return None
        return xml
    except requests.RequestException:
        return None


# ── Quarter orchestration ─────────────────────────────────────────────────────

def ingest_quarter(
    year: int,
    quarter: int,
    cusip_map: dict[str, str],
    output_dir: Path,
    top_n: int = 500,
) -> int:
    """
    Download and parse 13F-HR filings for one quarter, save per-filer Parquets.

    Returns total number of rows written across all filers.
    Output: output_dir/<YYYYQQ>/<CIK>.parquet (only rows matching cusip_map).
    Skips filers where the output file already exists (idempotent).
    """
    import datetime as dt

    quarter_str = f"{year}Q{quarter}"
    period_end  = _quarter_end_date(year, quarter)
    out_dir     = output_dir / quarter_str
    out_dir.mkdir(parents=True, exist_ok=True)

    index_df = fetch_quarter_index(year, quarter)
    if index_df.is_empty():
        _LOG.warning("[13F] No 13F-HR filers found for %s", quarter_str)
        return 0

    top_ciks = rank_filers_by_position_count(index_df, top_n=top_n)
    filer_map = {
        row["cik"]: row["filename"]
        for row in index_df.iter_rows(named=True)
        if row["cik"] in top_ciks
    }

    total_rows = 0
    for i, cik in enumerate(top_ciks):
        filename = filer_map.get(cik)
        if filename is None:
            continue

        out_path = out_dir / f"{cik}.parquet"
        if out_path.exists():
            _LOG.debug("[13F] %s/%s already exists — skip", quarter_str, cik)
            continue

        xml = fetch_filing_xml(cik, filename)
        if xml is None:
            _LOG.debug("[13F] %s/%s: no XML found", quarter_str, cik)
            continue

        rows = parse_holdings_xml(xml, cusip_map)
        if not rows:
            continue

        for row in rows:
            row["cik"]        = cik
            row["quarter"]    = quarter_str
            row["period_end"] = dt.date.fromisoformat(period_end)

        table = pa.Table.from_pylist(rows, schema=_RAW_SCHEMA)
        pq.write_table(table, str(out_path), compression="snappy")
        total_rows += len(rows)
        _LOG.info("[13F] %s/%s: %d watchlist rows", quarter_str, cik, len(rows))

        if i % 50 == 0 and i > 0:
            _LOG.info("[13F] %s: %d/%d filers processed, %d rows", quarter_str, i, len(top_ciks), total_rows)

    _LOG.info("[13F] %s complete: %d total rows", quarter_str, total_rows)
    return total_rows


# ── History bootstrap ─────────────────────────────────────────────────────────

def build_13f_history(
    cusip_map_path: Path,
    output_dir: Path,
    start_year: int = 2013,
    end_year: int | None = None,
    top_n: int = 500,
) -> None:
    """
    Bootstrap full 13F history from start_year-Q1 to present.

    Idempotent: skips quarters where *.parquet files already exist in the output dir.
    output_dir: path to the 13f_holdings/raw directory.
    """
    import datetime as dt

    cusip_map: dict[str, str] = json.loads(cusip_map_path.read_text())
    today = dt.date.today()
    if end_year is None:
        end_year = today.year

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            # Skip future quarters
            quarter_end_month = quarter * 3
            if year == today.year and quarter_end_month > today.month:
                break

            quarter_dir = output_dir / f"{year}Q{quarter}"
            existing = list(quarter_dir.glob("*.parquet")) if quarter_dir.exists() else []
            if existing:
                _LOG.info("[13F] %sQ%s: %d files exist — skip", year, quarter, len(existing))
                continue

            _LOG.info("[13F] Ingesting %sQ%s ...", year, quarter)
            try:
                ingest_quarter(year, quarter, cusip_map, output_dir, top_n=top_n)
            except Exception as exc:
                _LOG.warning("[13F] %sQ%s failed: %s — continuing", year, quarter, exc)


if __name__ == "__main__":
    import argparse
    import datetime as dt
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download SEC EDGAR 13F-HR institutional holdings.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Download full history from 2013-Q1. Omit for incremental (current + prior quarter).",
    )
    parser.add_argument("--top-n", type=int, default=500, help="Top N filers per quarter (default 500).")
    args = parser.parse_args()

    project_root   = Path(__file__).parent.parent
    cusip_map_path = project_root / "data" / "raw" / "financials" / "cusip_map.json"
    output_dir     = project_root / "data" / "raw" / "financials" / "13f_holdings" / "raw"

    if not cusip_map_path.exists():
        raise FileNotFoundError(
            f"CUSIP map not found at {cusip_map_path}. Run: python ingestion/build_cusip_map.py"
        )

    if args.bootstrap:
        build_13f_history(cusip_map_path, output_dir, start_year=2013, top_n=args.top_n)
    else:
        # Incremental: current quarter and prior quarter
        cusip_map: dict[str, str] = json.loads(cusip_map_path.read_text())
        today = dt.date.today()
        current_q = (today.month - 1) // 3 + 1
        pairs = [(today.year, current_q)]
        if current_q == 1:
            pairs.append((today.year - 1, 4))
        else:
            pairs.append((today.year, current_q - 1))
        for year, q in pairs:
            _LOG.info("[13F] Incremental: %sQ%s", year, q)
            ingest_quarter(year, q, cusip_map, output_dir, top_n=args.top_n)

    _LOG.info("[13F] Done.")
