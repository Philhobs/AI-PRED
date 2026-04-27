"""
Audit CIK_MAP entries against SEC's live registry.

Walks every (ticker, cik) in ingestion.edgar_fundamentals_ingestion.CIK_MAP and
checks two things:

  1. Does our CIK still respond 200 from the EDGAR submissions API?
     (catches CIKs that became 404 after a corporate reorganization)

  2. For tickers SEC's current company_tickers.json maps to a different CIK,
     does the SEC's CIK have materially fresher 8-K activity?
     (catches CIKs that point to a legacy / pre-restructure entity)

Reports are grouped by severity:
  ✗ BROKEN   — our CIK 404s; must switch
  ⚠ STALE    — SEC has fresher 8-Ks (>30d gap); should switch
  ⚙ MINOR    — SEC mismatch but our CIK is fresher or gap <30d (no action)
  ✓ OK       — our CIK matches SEC current

Usage:
    python tools/audit_ciks.py                   # full audit, ~3 min for 94 CIKs
    python tools/audit_ciks.py --threshold 60    # only flag gaps > 60 days

Exit code 0 = all CIKs healthy, 1 = one or more BROKEN or STALE CIKs.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date as _date
from pathlib import Path
from typing import NamedTuple

import requests

_HEADERS = {"User-Agent": "AI-PRED CIK audit plarenberg@gmail.com"}
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"


class TickerStatus(NamedTuple):
    """Per-ticker audit result."""
    ticker: str
    our_cik: str
    sec_cik: str | None             # what SEC's company_tickers.json currently lists
    our_latest_8k: str | None        # ISO date or None
    sec_latest_8k: str | None        # ISO date if sec_cik differs from ours
    severity: str                    # "broken", "stale", "minor", "ok"
    suggested_cik: str | None        # propose to switch if severity is broken/stale


def _latest_8k(cik: str) -> tuple[str | None, int]:
    """Return (most_recent_8k_iso_date, http_status). Date is None if no 8-Ks or HTTP failure."""
    cik_padded = cik.lstrip("0").zfill(10)
    try:
        r = requests.get(_SUBMISSIONS_URL.format(cik=cik_padded), headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return None, r.status_code
        recent = r.json().get("filings", {}).get("recent", {})
        for form, d in zip(recent.get("form", []), recent.get("filingDate", [])):
            if form == "8-K":
                return d, 200
        return None, 200   # 200 OK but no 8-Ks in the recent window
    except Exception:
        return None, 0


def _load_sec_ticker_map() -> dict[str, str]:
    """Fetch SEC's master ticker→CIK map. Returns {} on failure."""
    try:
        r = requests.get(_TICKERS_JSON_URL, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        return {e["ticker"]: str(e["cik_str"]).zfill(10) for e in r.json().values()}
    except Exception as exc:
        print(f"WARNING: failed to fetch SEC company_tickers.json ({exc}) — proceeding without it",
              file=sys.stderr)
        return {}


def audit(threshold_days: int = 30, sleep_s: float = 0.15) -> list[TickerStatus]:
    """Run the full audit. Returns one TickerStatus per ticker in CIK_MAP."""
    from ingestion.edgar_fundamentals_ingestion import CIK_MAP

    sec_map = _load_sec_ticker_map()
    results: list[TickerStatus] = []

    for ticker, our_cik in CIK_MAP.items():
        our_d, our_status = _latest_8k(our_cik)
        time.sleep(sleep_s)

        sec_cik = sec_map.get(ticker)
        sec_d: str | None = None
        if sec_cik and sec_cik != our_cik:
            sec_d, _ = _latest_8k(sec_cik)
            time.sleep(sleep_s)

        # Severity decision
        if our_status != 200:
            severity = "broken"
            suggested = sec_cik   # if SEC has it; else None
        elif our_d is None and sec_d is not None:
            severity = "stale"
            suggested = sec_cik
        elif our_d is not None and sec_d is not None and sec_d > our_d:
            gap = (_date.fromisoformat(sec_d) - _date.fromisoformat(our_d)).days
            if gap > threshold_days:
                severity = "stale"
                suggested = sec_cik
            else:
                severity = "minor"
                suggested = None
        elif sec_cik and sec_cik != our_cik:
            severity = "minor"   # mismatch but ours is fresher or equal
            suggested = None
        else:
            severity = "ok"
            suggested = None

        results.append(TickerStatus(
            ticker=ticker, our_cik=our_cik, sec_cik=sec_cik,
            our_latest_8k=our_d, sec_latest_8k=sec_d,
            severity=severity, suggested_cik=suggested,
        ))

    return results


def _print_report(results: list[TickerStatus], threshold_days: int) -> int:
    by_severity: dict[str, list[TickerStatus]] = {"broken": [], "stale": [], "minor": [], "ok": []}
    for r in results:
        by_severity[r.severity].append(r)

    print(f"\nCIK audit — {len(results)} tickers checked, threshold {threshold_days}d")
    print("=" * 70)

    if by_severity["broken"]:
        print(f"\n✗ BROKEN ({len(by_severity['broken'])}):  our CIK returns non-200")
        print("-" * 70)
        for r in by_severity["broken"]:
            sug = f"  →  switch to {r.suggested_cik}" if r.suggested_cik else "  →  no SEC suggestion"
            print(f"  {r.ticker:<6} CIK {r.our_cik}{sug}")

    if by_severity["stale"]:
        print(f"\n⚠ STALE ({len(by_severity['stale'])}):  SEC has materially fresher 8-Ks")
        print("-" * 70)
        for r in by_severity["stale"]:
            ours = r.our_latest_8k or "(no 8-Ks)"
            theirs = r.sec_latest_8k or "?"
            print(f"  {r.ticker:<6} ours last 8-K {ours} | SEC {theirs}  →  switch to {r.suggested_cik}")

    if by_severity["minor"]:
        print(f"\n⚙ MINOR ({len(by_severity['minor'])}):  SEC mismatch but ours is fresher or gap < {threshold_days}d")
        print("-" * 70)
        for r in by_severity["minor"][:10]:    # show first 10; truncate the rest
            ours = r.our_latest_8k or "(no 8-Ks)"
            theirs = r.sec_latest_8k or "?"
            print(f"  {r.ticker:<6} ours {ours} | SEC {theirs} (CIK {r.sec_cik})")
        if len(by_severity["minor"]) > 10:
            print(f"  ... and {len(by_severity['minor']) - 10} more")

    n_ok = len(by_severity["ok"])
    print(f"\n✓ OK: {n_ok}/{len(results)}")
    print("=" * 70)

    if by_severity["broken"] or by_severity["stale"]:
        print("\nSuggested CIK_MAP edits:")
        for r in by_severity["broken"] + by_severity["stale"]:
            if r.suggested_cik:
                print(f'    "{r.ticker}":  "{r.suggested_cik}",')
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CIK_MAP against SEC's live registry.")
    parser.add_argument(
        "--threshold", type=int, default=30,
        help="Days of staleness before SEC-mismatch is flagged as STALE (default 30).",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.15,
        help="Seconds between SEC API calls (SEC rate limit: 10 req/s; default 0.15s).",
    )
    args = parser.parse_args()

    results = audit(threshold_days=args.threshold, sleep_s=args.sleep)
    return _print_report(results, threshold_days=args.threshold)


if __name__ == "__main__":
    sys.exit(main())
