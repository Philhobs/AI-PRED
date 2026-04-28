"""Discover CUSIPs for unmapped registry tickers by parsing a large 13F-HR filing.

Approach: pick a mega-cap holder (Vanguard Total Stock Market, BlackRock iShares,
State Street SPY) whose 13F lists virtually every US-listed equity. Parse the full
information table — NOT filtered through cusip_map — and build a name → cusip
table. Then match our missing tickers' canonical issuer names against it.

Run: python -m tools.discover_cusips
Outputs: prints the discovered (ticker, cusip) pairs and emits a copy-pasteable
block for `_STATIC_CUSIP_MAP` in ingestion/build_cusip_map.py.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import requests

from ingestion.sec_13f_ingestion import (
    _HEADERS,
    _strip_ns,
    fetch_filing_xml,
    fetch_quarter_index,
)
from ingestion.ticker_registry import us_listed_tickers

# Mega-cap holders whose 13Fs list nearly every US-listed equity. We try them
# in order — first one that yields all 26 missing tickers wins.
# CIKs verified via SEC EDGAR.
_PROBE_FILERS = [
    ("0000102909", "Vanguard Group Inc"),
    ("0000093751", "State Street Corp"),
    ("0001037389", "Renaissance Technologies"),  # holds many foreign ADRs
    ("0001067983", "Berkshire Hathaway"),
]

# Company names for the 26 missing tickers. These are what appears in the
# `nameOfIssuer` element of a 13F holding row. Names are matched case-
# insensitively after stripping punctuation. Include a few alternates per
# ticker because filers don't normalize: e.g. "Cisco Systems Inc" vs
# "CISCO SYS INC". Keep the patterns broad enough to survive minor variation.
_TICKER_NAMES: dict[str, list[str]] = {
    "ADI":  ["analog devices"],
    "AKAM": ["akamai"],
    "CGNX": ["cognex"],
    "CHKP": ["check point software"],
    "CRWD": ["crowdstrike"],
    "CYBR": ["cyberark"],
    "EMR":  ["emerson electric", "emerson elec"],
    "ERIC": ["ericsson", "telefonaktiebolaget", "lm erics"],
    "FTNT": ["fortinet"],
    "ISRG": ["intuitive surgical"],
    "JNPR": ["juniper networks"],
    "MCHP": ["microchip technology"],
    "OKTA": ["okta"],
    "PANW": ["palo alto networks"],
    "QLYS": ["qualys"],
    "ROK":  ["rockwell automation"],
    "RPD":  ["rapid7"],
    "S":    ["sentinelone"],
    "STM":  ["stmicroelectronics"],
    "SYM":  ["symbotic"],
    "TENB": ["tenable"],
    "TSLA": ["tesla"],
    "TXN":  ["texas instruments", "texas instrs"],
    "VRNS": ["varonis"],
    "ZBRA": ["zebra technologies"],
    "ZS":   ["zscaler"],
}


def _normalize(name: str) -> str:
    """Lowercase + strip non-alphanumeric for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _parse_holdings_unfiltered(xml_str: str) -> list[tuple[str, str]]:
    """Return (cusip, name_of_issuer) for every SH holding in the XML.

    Unlike `parse_holdings_xml` in the ingestion module this does NOT filter
    against an existing cusip_map — that's the whole point.
    """
    try:
        root = ET.fromstring(_strip_ns(xml_str))
    except ET.ParseError as exc:
        print(f"  parse error: {exc}", file=sys.stderr)
        return []

    out = []
    for info in root.iter("infoTable"):
        cusip_el = info.find("cusip")
        type_el  = info.find(".//sshPrnamtType")
        name_el  = info.find("nameOfIssuer")
        if any(el is None for el in (cusip_el, type_el, name_el)):
            continue
        if (type_el.text or "").strip() != "SH":
            continue
        cusip = (cusip_el.text or "").strip()
        name  = (name_el.text or "").strip()
        if cusip and name:
            out.append((cusip, name))
    return out


def _latest_13f_filename_for(cik: str) -> str | None:
    """Find the most recent 13F-HR filing for a given filer CIK by walking
    the prior 4 quarterly indices until we find one."""
    from datetime import date

    today = date.today()
    qtr = (today.month - 1) // 3 + 1
    year = today.year
    for _ in range(5):
        try:
            idx = fetch_quarter_index(year, qtr)
        except requests.HTTPError:
            idx = None
        if idx is not None and not idx.is_empty():
            hits = idx.filter(idx["cik"].cast(str).str.zfill(10) == cik)
            if hits.height:
                return hits.row(0, named=True)["filename"]
        qtr -= 1
        if qtr == 0:
            qtr = 4
            year -= 1
    return None


def _match_tickers(
    holdings: list[tuple[str, str]],
    targets: dict[str, list[str]],
) -> dict[str, str]:
    """Map each target ticker → first cusip whose name contains any pattern."""
    found: dict[str, str] = {}
    norm_holdings = [(c, _normalize(n)) for c, n in holdings]
    for ticker, patterns in targets.items():
        for cusip, n in norm_holdings:
            if any(_normalize(p) in n for p in patterns):
                found[ticker] = cusip
                break
    return found


def discover() -> dict[str, str]:
    missing = sorted(set(_TICKER_NAMES) & set(us_listed_tickers()))
    cusip_path = Path("data/raw/financials/cusip_map.json")
    existing = json.loads(cusip_path.read_text()) if cusip_path.exists() else {}
    todo = [t for t in missing if t not in existing]

    print(f"Searching for {len(todo)} unmapped tickers: {todo}")

    aggregated: dict[str, str] = {}
    for cik, label in _PROBE_FILERS:
        if len(aggregated) >= len(todo):
            break
        print(f"\nProbing {label} (CIK {cik})...")
        filename = _latest_13f_filename_for(cik)
        if filename is None:
            print(f"  no recent 13F-HR found")
            continue
        xml = fetch_filing_xml(cik, filename)
        if xml is None:
            print(f"  could not fetch XML for {filename}")
            continue
        holdings = _parse_holdings_unfiltered(xml)
        print(f"  parsed {len(holdings)} SH holdings")

        new = _match_tickers(holdings, {t: _TICKER_NAMES[t] for t in todo if t not in aggregated})
        for t, c in new.items():
            aggregated[t] = c
            print(f"    {t}: {c}")

    print(f"\nDiscovered {len(aggregated)} of {len(todo)} ({len(todo) - len(aggregated)} still missing).")
    if missing_after := [t for t in todo if t not in aggregated]:
        print(f"Still missing: {missing_after}")

    if aggregated:
        print("\nPaste into ingestion/build_cusip_map.py::_STATIC_CUSIP_MAP:")
        for t in sorted(aggregated):
            print(f'    "{t}":   "{aggregated[t]}",')
    return aggregated


if __name__ == "__main__":
    discover()
