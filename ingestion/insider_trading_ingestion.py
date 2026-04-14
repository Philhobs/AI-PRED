"""
Insider trading ingestion: SEC EDGAR Form 4 + congressional trades.

Corporate insiders: open-market purchases (P) and sales (S) only.
Congressional trades: House Stock Watcher bulk JSON + Senate EFTS paginated API.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl
import requests

from ingestion.edgar_fundamentals_ingestion import CIK_MAP

_LOG = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ai-infra-predictor research@example.com"}
_OPEN_MARKET_CODES = {"P", "S"}


def _parse_form4_xml(xml_text: str, ticker: str, filed_date: str) -> list[dict]:
    """Parse Form 4 XML. Returns list of dicts for P/S transactions only."""
    # Strip XML namespaces so XPath expressions work on all EDGAR variants
    xml_text = re.sub(r' xmlns(?::\w+)?="[^"]*"', '', xml_text)

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        _LOG.warning("Failed to parse Form 4 XML for %s filed %s", ticker, filed_date)
        return []

    # Extract reporting owner info (first owner)
    insider_name = ""
    insider_title = ""
    owner = root.find(".//reportingOwner")
    if owner is not None:
        name_el = owner.find(".//rptOwnerName")
        if name_el is not None and name_el.text:
            insider_name = name_el.text.strip()
        title_el = owner.find(".//officerTitle")
        if title_el is not None and title_el.text:
            insider_title = title_el.text.strip()
        if not insider_title:
            for tag in ("isDirector", "isOfficer", "isTenPercentOwner"):
                el = owner.find(f".//{tag}")
                if el is not None and el.text and el.text.strip() == "1":
                    insider_title = tag.replace("is", "")
                    break

    rows = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el = txn.find(".//transactionCode")
        if code_el is None or code_el.text is None:
            continue
        code = code_el.text.strip()
        if code not in _OPEN_MARKET_CODES:
            continue

        date_el = txn.find(".//transactionDate/value")
        shares_el = txn.find(".//transactionShares/value")
        price_el = txn.find(".//transactionPricePerShare/value")

        if date_el is None or shares_el is None or price_el is None:
            continue

        try:
            shares = float(shares_el.text)
            price = float(price_el.text)
        except (ValueError, TypeError):
            _LOG.debug("Skipping transaction for %s: non-numeric shares=%r price=%r",
                       ticker, shares_el.text, price_el.text)
            continue

        rows.append({
            "ticker": ticker,
            "filed_date": filed_date,
            "transaction_date": (date_el.text or "").strip(),
            "insider_name": insider_name,
            "insider_title": insider_title,
            "transaction_code": code,
            "shares": shares,
            "price_per_share": price,
            "value": shares * price,
        })

    return rows


_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _fetch_form4_filings(cik: str, ticker: str) -> list[dict]:
    """Fetch Form 4 filing accession numbers from EDGAR submissions API.

    Returns list of dicts with keys: accession (no dashes), filed.
    """
    cik_padded = cik.lstrip("0").zfill(10)
    url = _EDGAR_SUBMISSIONS_URL.format(cik=cik_padded)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        _LOG.warning("Failed to fetch submissions for %s (%s): %s", ticker, cik, exc)
        return []

    data = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    filed_dates = filings.get("filingDate", [])

    result = []
    for form, accession, filed in zip(forms, accessions, filed_dates):
        if form == "4":
            result.append({
                "accession": accession.replace("-", ""),
                "filed": filed,
            })

    # Fetch older filings pages if present
    older = data.get("filings", {}).get("files", [])
    for file_entry in older:
        older_url = f"https://data.sec.gov/submissions/{file_entry['name']}"
        try:
            time.sleep(0.1)
            r2 = requests.get(older_url, headers=_HEADERS, timeout=30)
            r2.raise_for_status()
            older_data = r2.json()
            o_forms = older_data.get("form", [])
            o_accessions = older_data.get("accessionNumber", [])
            o_dates = older_data.get("filingDate", [])
            for form, accession, filed in zip(o_forms, o_accessions, o_dates):
                if form == "4":
                    result.append({
                        "accession": accession.replace("-", ""),
                        "filed": filed,
                    })
        except requests.RequestException as exc:
            _LOG.warning("Failed to fetch older submissions page for %s: %s", ticker, exc)

    _LOG.info("Found %d Form 4 filings for %s", len(result), ticker)
    return result


def _fetch_form4_xml(cik: str, accession: str, ticker: str) -> str:
    """Fetch raw Form 4 XML text for one filing. Returns empty string on failure.

    Accession is the 18-digit string with no dashes.
    Reconstructs dashed form for the filename: XXXXXXXXXX-YY-ZZZZZZ.
    """
    cik_num = cik.lstrip("0")
    accession_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession}/{accession_dashed}.txt"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        _LOG.warning("Failed to fetch Form 4 XML for %s accession %s: %s", ticker, accession, exc)
        return ""


def _empty_insider_df() -> pl.DataFrame:
    """Return an empty DataFrame with the full insider trades schema."""
    return pl.DataFrame(schema={
        "ticker": pl.Utf8,
        "filed_date": pl.Date,
        "transaction_date": pl.Date,
        "insider_name": pl.Utf8,
        "insider_title": pl.Utf8,
        "transaction_code": pl.Utf8,
        "shares": pl.Float64,
        "price_per_share": pl.Float64,
        "value": pl.Float64,
    })


def fetch_corporate_insider_trades(ticker: str) -> pl.DataFrame:
    """Fetch all Form 4 P/S transactions for one ticker from EDGAR.

    Returns DataFrame with schema:
    [ticker, filed_date, transaction_date, insider_name, insider_title,
     transaction_code, shares, price_per_share, value]
    """
    cik = CIK_MAP.get(ticker)
    if not cik:
        _LOG.warning("No CIK found for ticker %s", ticker)
        return _empty_insider_df()

    filings = _fetch_form4_filings(cik, ticker)
    all_rows: list[dict] = []

    for i, filing in enumerate(filings):
        time.sleep(0.1)  # SEC rate limit: 10 req/s
        xml_text = _fetch_form4_xml(cik, filing["accession"], ticker)
        if not xml_text:
            continue
        rows = _parse_form4_xml(xml_text, ticker=ticker, filed_date=filing["filed"])
        all_rows.extend(rows)
        if (i + 1) % 50 == 0:
            _LOG.info("%s: processed %d/%d filings, %d transactions so far",
                      ticker, i + 1, len(filings), len(all_rows))

    if not all_rows:
        return _empty_insider_df()

    df = pl.DataFrame(all_rows).with_columns([
        pl.col("filed_date").str.to_date("%Y-%m-%d"),
        pl.col("transaction_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("shares").cast(pl.Float64),
        pl.col("price_per_share").cast(pl.Float64),
        pl.col("value").cast(pl.Float64),
    ])
    return df


def save_corporate_insider_trades(df: pl.DataFrame, ticker: str, output_dir: Path) -> None:
    """Write to data/raw/financials/insider_trades/<TICKER>/transactions.parquet."""
    out_path = output_dir / ticker / "transactions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="snappy")
    _LOG.info("Saved %d insider trade rows to %s", len(df), out_path)
