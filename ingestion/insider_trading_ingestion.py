"""
Insider trading ingestion: SEC EDGAR Form 4 + congressional trades.

Corporate insiders: open-market purchases (P) and sales (S) only.
Congressional trades: House Stock Watcher bulk JSON + Senate EFTS paginated API.
"""
from __future__ import annotations

import logging
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
            continue

        rows.append({
            "ticker": ticker,
            "filed_date": filed_date,
            "transaction_date": date_el.text.strip(),
            "insider_name": insider_name,
            "insider_title": insider_title,
            "transaction_code": code,
            "shares": shares,
            "price_per_share": price,
            "value": shares * price,
        })

    return rows
