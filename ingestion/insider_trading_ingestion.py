"""
Insider trading ingestion: SEC EDGAR Form 4 + congressional trades.

Corporate insiders: open-market purchases (P) and sales (S) only.
Congressional trades: House Stock Watcher bulk JSON + Senate EFTS paginated API.
"""
from __future__ import annotations

import io
import logging
import re as _re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, date
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

    Returns list of dicts with keys: accession (no dashes), filed, primary_doc.
    primary_doc is the path within the accession folder (e.g. 'wf-form4_xxx.xml'
    or 'xslF345X06/wf-form4_xxx.xml' — the xsl prefix is stripped at fetch time).
    Filings without an .xml primary_doc are skipped (legacy paper filings).
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

    def _collect(block: dict) -> list[dict]:
        forms = block.get("form", [])
        accessions = block.get("accessionNumber", [])
        filed_dates = block.get("filingDate", [])
        primary_docs = block.get("primaryDocument", [""] * len(forms))
        rows = []
        for form, accession, filed, primary_doc in zip(forms, accessions, filed_dates, primary_docs):
            if form != "4":
                continue
            if not primary_doc or not primary_doc.endswith(".xml"):
                continue   # legacy paper filing or unexpected primary_doc shape
            rows.append({
                "accession": accession.replace("-", ""),
                "filed": filed,
                "primary_doc": primary_doc,
            })
        return rows

    result: list[dict] = _collect(data.get("filings", {}).get("recent", {}))

    # Fetch older filings pages if present
    older = data.get("filings", {}).get("files", [])
    for file_entry in older:
        older_url = f"https://data.sec.gov/submissions/{file_entry['name']}"
        try:
            time.sleep(0.1)
            r2 = requests.get(older_url, headers=_HEADERS, timeout=30)
            r2.raise_for_status()
            result.extend(_collect(r2.json()))
        except requests.RequestException as exc:
            _LOG.warning("Failed to fetch older submissions page for %s: %s", ticker, exc)

    _LOG.info("Found %d Form 4 (XML) filings for %s", len(result), ticker)
    return result


def _raw_xml_path(primary_doc: str) -> str:
    """Strip a leading xsl* viewer prefix to get the raw XML filename.

    EDGAR's submissions JSON returns paths like 'xslF345X06/wf-form4_xxx.xml' which
    point to the HTML-rendered viewer. The raw XML lives at the same accession
    folder one level up: 'wf-form4_xxx.xml'.
    """
    if "/" in primary_doc and primary_doc.split("/", 1)[0].startswith("xsl"):
        return primary_doc.split("/", 1)[1]
    return primary_doc


def _fetch_form4_xml(cik: str, accession: str, primary_doc: str, ticker: str) -> str:
    """Fetch raw Form 4 XML text for one filing. Returns empty string on failure.

    Accession is the 18-digit string with no dashes. primary_doc is the path
    from EDGAR's submissions JSON; the xsl viewer prefix is stripped to get the
    raw XML URL.
    """
    cik_num = cik.lstrip("0")
    raw_name = _raw_xml_path(primary_doc)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession}/{raw_name}"
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
        xml_text = _fetch_form4_xml(cik, filing["accession"], filing["primary_doc"], ticker)
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

_HOUSE_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
_HOUSE_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"


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


def _fetch_house_ptr_index(year: int) -> list[dict]:
    """Download annual House disclosure ZIP and return list of PTR filing dicts."""
    url = _HOUSE_INDEX_URL.format(year=year)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        _LOG.warning("Failed to download House index for %d: %s", year, exc)
        return []

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(f"{year}FD.xml") as f:
            tree = ET.parse(f)

    return [
        {
            "last": m.findtext("Last", ""),
            "first": m.findtext("First", ""),
            "year": year,
            "filing_date": m.findtext("FilingDate", ""),
            "doc_id": m.findtext("DocID", ""),
        }
        for m in tree.findall("Member")
        if m.findtext("FilingType") == "P"
    ]


def _parse_ptr_pdf_text(text: str, politician_name: str, watchlist: set[str]) -> list[dict]:
    """Extract stock trades from PTR PDF text for watchlist tickers.

    Each transaction in the text has a ticker in parentheses, e.g. '(NVDA)',
    preceded (within 3 lines) by a transaction type 'P' or 'S' and a date.
    """
    lines = text.split("\n")
    rows: list[dict] = []
    seen: set[tuple] = set()  # deduplicate (ticker, date, type)

    for i, line in enumerate(lines):
        ticker_match = _re.search(r"\(([A-Z]{1,5})\)", line)
        if not ticker_match:
            continue
        ticker = ticker_match.group(1)
        if ticker not in watchlist:
            continue

        # Context for type/date: current + 3 preceding lines (transaction type always precedes ticker)
        pre_context = " ".join(lines[max(0, i - 3):i + 1])

        # Transaction type: prefer match on current line (type+date on same line as ticker);
        # fall back to pre_context for wrapped descriptions where ticker is on a continuation line.
        _TYPE_PAT = r"(?<!\w)([PS])\s*(?:\(partial\))?\s+(\d{2}/\d{2}/\d{4})"
        type_match = _re.search(_TYPE_PAT, line) or _re.search(_TYPE_PAT, pre_context)
        if not type_match:
            continue
        txn_type = "purchase" if type_match.group(1) == "P" else "sale"
        txn_date_str = type_match.group(2)
        try:
            txn_date = datetime.strptime(txn_date_str, "%m/%d/%Y").date()
        except ValueError:
            continue

        # Dedup: same (ticker, date, type) from different PDF lines
        key = (ticker, txn_date, txn_type)
        if key in seen:
            continue
        seen.add(key)

        # Amount range: only from pre_context (preceding + current line) to avoid
        # bleeding into the next transaction's amounts on the following line
        amounts = [float(a.replace(",", "")) for a in _re.findall(r"\$([\d,]+)", pre_context)]
        if len(amounts) >= 2:
            amount_low, amount_high = sorted(amounts[-2:])
        elif len(amounts) == 1:
            amount_low = amount_high = amounts[0]
        else:
            amount_low = amount_high = 0.0

        rows.append({
            "ticker": ticker,
            "trade_date": txn_date,
            "politician_name": politician_name,
            "chamber": "house",
            "party": "",
            "transaction_type": txn_type,
            "amount_low": amount_low,
            "amount_high": amount_high,
            "amount_mid": (amount_low + amount_high) / 2,
        })

    return rows


def fetch_congressional_trades_house(days_back: int = 365) -> pl.DataFrame:
    """Fetch House PTR filings from disclosures-clerk.house.gov.

    Downloads the annual filing index ZIP for the current and prior year,
    filters to PTRs filed within `days_back` days, then downloads and parses
    each PDF with pdfplumber to extract stock transactions for watchlist tickers.
    """
    try:
        import pdfplumber
    except ImportError:
        _LOG.warning("pdfplumber not installed — run: pip install pdfplumber")
        return _empty_congressional_df()

    watchlist = set(CIK_MAP.keys())
    cutoff_date = date.today().replace(year=date.today().year) - __import__("datetime").timedelta(days=days_back)
    current_year = date.today().year
    all_rows: list[dict] = []

    for year in [current_year, current_year - 1]:
        ptrs = _fetch_house_ptr_index(year)
        _LOG.info("House %d: %d PTR filings in index", year, len(ptrs))

        # Filter to recent filings only
        recent = []
        for ptr in ptrs:
            try:
                fd = datetime.strptime(ptr["filing_date"], "%m/%d/%Y").date()
                if fd >= cutoff_date:
                    recent.append(ptr)
            except ValueError:
                pass
        _LOG.info("House %d: %d PTRs filed within %d days", year, len(recent), days_back)

        for i, ptr in enumerate(recent):
            doc_id = ptr["doc_id"]
            politician_name = f"{ptr['first']} {ptr['last']}".strip()
            pdf_url = _HOUSE_PDF_URL.format(year=year, doc_id=doc_id)

            try:
                pdf_resp = requests.get(pdf_url, headers=_HEADERS, timeout=30)
                pdf_resp.raise_for_status()
            except requests.RequestException as exc:
                _LOG.debug("Failed to fetch PDF %s: %s", doc_id, exc)
                continue

            try:
                with pdfplumber.open(io.BytesIO(pdf_resp.content)) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as exc:
                _LOG.debug("Failed to parse PDF %s: %s", doc_id, exc)
                continue

            rows = _parse_ptr_pdf_text(text, politician_name, watchlist)
            all_rows.extend(rows)

            if (i + 1) % 25 == 0:
                _LOG.info(
                    "House %d: %d/%d PDFs processed, %d matching rows so far",
                    year, i + 1, len(recent), len(all_rows),
                )
            time.sleep(0.3)  # Be polite to the server

    if not all_rows:
        _LOG.warning("No House congressional trades found for watchlist tickers")
        return _empty_congressional_df()

    df = pl.DataFrame(all_rows)
    _LOG.info("House trades: %d rows for %d unique tickers", len(df), df["ticker"].n_unique())
    return df


def fetch_congressional_trades_senate() -> pl.DataFrame:
    """Fetch Senate PTR filings.

    Note: efts.senate.gov (the previous source) is no longer accessible (NXDOMAIN).
    Senate data currently unavailable without a paid API key (e.g. QuiverQuant).
    Returns an empty DataFrame — congressional features will rely on House data only.
    """
    _LOG.warning(
        "Senate EFTS (efts.senate.gov) is no longer accessible. "
        "Senate congressional trades unavailable. "
        "Features will use House data only."
    )
    return _empty_congressional_df()


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
