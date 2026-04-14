"""
Insider trading ingestion: SEC EDGAR Form 4 + congressional trades.

Corporate insiders: open-market purchases (P) and sales (S) only.
Congressional trades: House Stock Watcher bulk JSON + Senate EFTS paginated API.
"""
from __future__ import annotations

import logging
import re as _re
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
    xml_text = _re.sub(r' xmlns(?::\w+)?="[^"]*"', '', xml_text)

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


# ─────────────────────────────────────────────────────────────────────────────
# Congressional trades
# ─────────────────────────────────────────────────────────────────────────────

_HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_EFTS_URL = (
    "https://efts.senate.gov/LATEST/search-index"
    "?q=&dateRange=custom&forms=PTR&limit=100&offset={offset}"
)


def _parse_amount_band(amount_str: str) -> float | None:
    """Parse amount band string like '$15,001 - $50,000' to midpoint.

    Returns None if the string cannot be parsed.
    """
    if not amount_str:
        return None
    s = amount_str.strip().lower().replace(",", "")
    if "over" in s:
        return 1_500_000.0  # Use 150% of the $1M lower bound as a midpoint heuristic
    nums = _re.findall(r"\$?([\d]+)", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return None


def _empty_congressional_df() -> pl.DataFrame:
    """Return empty congressional trades DataFrame with correct schema."""
    return pl.DataFrame(schema={
        "ticker": pl.Utf8,
        "trade_date": pl.Date,
        "politician_name": pl.Utf8,
        "chamber": pl.Utf8,
        "party": pl.Utf8,
        "transaction_type": pl.Utf8,
        "amount_low": pl.Float64,
        "amount_high": pl.Float64,
        "amount_mid": pl.Float64,
    })


def _parse_house_json_records(records: list[dict], watchlist: set[str]) -> pl.DataFrame:
    """Parse House Stock Watcher JSON records, filter to watchlist tickers.

    Returns DataFrame with congressional trades schema.
    """
    rows = []
    for rec in records:
        ticker = str(rec.get("ticker", "")).upper().strip()
        if ticker not in watchlist:
            continue
        amount_str = str(rec.get("amount", ""))
        amount_mid = _parse_amount_band(amount_str) or 0.0
        # Extract low/high from same string for consistency
        clean = amount_str.replace(",", "")
        if "over" in clean.lower():
            amount_low = 1_000_000.0
            amount_high = 1_000_000.0  # lower bound only; mid = 1.5M per _parse_amount_band
        else:
            nums = _re.findall(r"\$?([\d]+)", clean)
            amount_low = float(nums[0]) if len(nums) >= 1 else 0.0
            amount_high = float(nums[1]) if len(nums) >= 2 else amount_low

        trade_date = rec.get("transaction_date") or rec.get("disclosure_date", "")
        if not rec.get("transaction_date") and trade_date:
            _LOG.debug("Using disclosure_date as trade_date for %s %s", ticker, trade_date)
        txn_type = str(rec.get("type", "")).lower()
        if "purchase" in txn_type or "buy" in txn_type:
            txn_type = "purchase"
        elif "sale" in txn_type or "sell" in txn_type:
            txn_type = "sale"

        if txn_type not in ("purchase", "sale"):
            _LOG.debug("Unknown transaction type %r for %s — skipping", txn_type, ticker)
            continue

        rows.append({
            "ticker": ticker,
            "trade_date": trade_date,
            "politician_name": str(rec.get("representative", "")),
            "chamber": "house",
            "party": str(rec.get("party", "")).lower(),
            "transaction_type": txn_type,
            "amount_low": amount_low,
            "amount_high": amount_high,
            "amount_mid": amount_mid,
        })

    if not rows:
        return _empty_congressional_df()

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("trade_date").str.to_date("%Y-%m-%d", strict=False))
        .drop_nulls(subset=["trade_date"])
    )


def fetch_congressional_trades_house() -> pl.DataFrame:
    """Fetch all House Stock Watcher transactions, filter to watchlist tickers."""
    watchlist = set(CIK_MAP.keys())
    _LOG.info("Fetching House Stock Watcher data...")
    try:
        resp = requests.get(_HOUSE_URL, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        records = resp.json()
    except requests.RequestException as exc:
        _LOG.warning("Failed to fetch House Stock Watcher data: %s", exc)
        return _empty_congressional_df()

    df = _parse_house_json_records(records, watchlist=watchlist)
    _LOG.info("House trades: %d rows for watchlist tickers", len(df))
    return df


def fetch_congressional_trades_senate() -> pl.DataFrame:
    """Fetch Senate PTR filings, paginate until <100 records per page.

    Filters to watchlist tickers by extracting ticker from parentheses in asset_name,
    e.g. 'NVIDIA Corp (NVDA)' → 'NVDA'.
    """
    watchlist = set(CIK_MAP.keys())
    all_rows: list[dict] = []
    offset = 0

    while True:
        url = _SENATE_EFTS_URL.format(offset=offset)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            _LOG.warning("Failed to fetch Senate EFTS at offset %d: %s", offset, exc)
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            src = hit.get("_source", {})
            asset_name = str(src.get("asset_name", ""))
            ticker_match = _re.search(r"\(([A-Z]{2,5})\)", asset_name)
            if not ticker_match:
                continue
            ticker = ticker_match.group(1)
            if ticker not in watchlist:
                continue

            txn_type = str(src.get("type", "")).lower()
            if "purchase" in txn_type or "buy" in txn_type:
                txn_type = "purchase"
            elif "sale" in txn_type or "sell" in txn_type:
                txn_type = "sale"

            if txn_type not in ("purchase", "sale"):
                _LOG.debug("Unknown transaction type %r for %s — skipping", txn_type, ticker)
                continue

            amount_str = str(src.get("amount", ""))
            amount_mid = _parse_amount_band(amount_str) or 0.0
            # Extract low/high from same string for consistency
            clean = amount_str.replace(",", "")
            if "over" in clean.lower():
                amount_low = 1_000_000.0
                amount_high = 1_000_000.0  # lower bound only; mid = 1.5M per _parse_amount_band
            else:
                nums = _re.findall(r"\$?([\d]+)", clean)
                amount_low = float(nums[0]) if len(nums) >= 1 else 0.0
                amount_high = float(nums[1]) if len(nums) >= 2 else amount_low

            trade_date = src.get("transaction_date") or src.get("date", "")
            first_name = str(src.get("first_name", "")).strip()
            last_name = str(src.get("last_name", "")).strip()
            politician_name = f"{first_name} {last_name}".strip()

            all_rows.append({
                "ticker": ticker,
                "trade_date": trade_date,
                "politician_name": politician_name,
                "chamber": "senate",
                "party": str(src.get("senator_party", "")).lower(),
                "transaction_type": txn_type,
                "amount_low": amount_low,
                "amount_high": amount_high,
                "amount_mid": amount_mid,
            })

        _LOG.info("Senate EFTS: fetched %d records at offset %d", len(hits), offset)
        if len(hits) < 100:
            break
        offset += 100
        time.sleep(1.0)  # Rate limit: 1s between Senate EFTS pages

    if not all_rows:
        return _empty_congressional_df()

    return (
        pl.DataFrame(all_rows)
        .with_columns(pl.col("trade_date").str.to_date("%Y-%m-%d", strict=False))
        .drop_nulls(subset=["trade_date"])
    )


def save_congressional_trades(df: pl.DataFrame, output_dir: Path) -> None:
    """Write to data/raw/financials/congressional_trades/all_transactions.parquet."""
    out_path = output_dir / "all_transactions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="snappy")
    _LOG.info("Saved %d congressional trade rows to %s", len(df), out_path)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — full ingestion run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    insider_dir = project_root / "data" / "raw" / "financials" / "insider_trades"
    congressional_dir = project_root / "data" / "raw" / "financials" / "congressional_trades"

    tickers = list(CIK_MAP.keys())
    _LOG.info("Fetching Form 4 filings for %d tickers...", len(tickers))

    for ticker in tickers:
        _LOG.info("--- %s ---", ticker)
        df = fetch_corporate_insider_trades(ticker)
        if len(df) > 0:
            save_corporate_insider_trades(df, ticker, insider_dir)
        else:
            _LOG.info("No Form 4 P/S transactions found for %s", ticker)

    _LOG.info("Fetching congressional trades...")
    house_df = fetch_congressional_trades_house()
    senate_df = fetch_congressional_trades_senate()

    dfs_to_concat = [d for d in [house_df, senate_df] if len(d) > 0]
    if dfs_to_concat:
        combined = pl.concat(dfs_to_concat)
        save_congressional_trades(combined, congressional_dir)
    else:
        _LOG.warning("No congressional trades found for watchlist tickers")

    _LOG.info("Insider trading ingestion complete.")
