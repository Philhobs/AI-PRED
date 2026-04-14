"""
EDGAR XBRL Historical Fundamentals Ingestion

Fetches ~12–15 years of quarterly financial data from SEC EDGAR for all 24
watchlist tickers. Computes 9 fundamental features (same schema as
fundamental_ingestion.py) and writes to:
  data/raw/financials/fundamentals/<TICKER>/quarterly.parquet

No downstream changes required — fundamental_features.py reads the same path.
"""
import logging
import time
import datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests

_LOG = logging.getLogger(__name__)

# ── CIK map ───────────────────────────────────────────────────────────────────

CIK_MAP: dict[str, str] = {
    "MSFT":  "0000789019",
    "AMZN":  "0001018724",
    "GOOGL": "0001652044",
    "META":  "0001326801",
    "NVDA":  "0001045810",
    "AMD":   "0000002488",
    "AVGO":  "0001730168",
    "MRVL":  "0001058057",
    "TSM":   "0001046179",
    "ASML":  "0000937556",
    "AMAT":  "0000796343",
    "LRCX":  "0000707549",
    "KLAC":  "0000319201",
    "VRT":   "0001748157",
    "SMCI":  "0000910638",
    "DELL":  "0001571996",
    "HPE":   "0001645590",
    "EQIX":  "0001101239",
    "DLR":   "0001297996",
    "AMT":   "0001053507",
    "CEG":   "0001868275",
    "VST":   "0001692819",
    "NRG":   "0001013871",
    "TLN":   "0000099590",
}

ANNUAL_FILERS: set[str] = {"TSM", "ASML"}

# ── XBRL concept lists (tried in order; first non-empty wins) ─────────────────

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]
EQUITY_CONCEPTS = [
    "StockholdersEquity",
    "StockholdersEquityAttributableToParent",
]
DEBT_CONCEPTS = ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"]
SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]

# ── PyArrow output schema (identical to fundamental_ingestion.py) ─────────────

_SCHEMA = pa.schema([
    pa.field("ticker",              pa.string()),
    pa.field("period_end",          pa.date32()),
    pa.field("pe_ratio_trailing",   pa.float64()),
    pa.field("price_to_sales",      pa.float64()),
    pa.field("price_to_book",       pa.float64()),
    pa.field("revenue_growth_yoy",  pa.float64()),
    pa.field("gross_margin",        pa.float64()),
    pa.field("operating_margin",    pa.float64()),
    pa.field("capex_to_revenue",    pa.float64()),
    pa.field("debt_to_equity",      pa.float64()),
    pa.field("current_ratio",       pa.float64()),
])

_HEADERS = {"User-Agent": "ai-infra-predictor research@example.com"}


# ── XBRL fetch ────────────────────────────────────────────────────────────────

def _fetch_xbrl(cik: str, concept: str) -> list[dict]:
    """
    Fetch all USD values for one XBRL concept from SEC EDGAR.
    Returns only 10-K, 10-Q, and 20-F records.
    Returns [] on 404 (concept not used by this company).
    """
    url = (
        f"https://data.sec.gov/api/xbrl/companyconcept/"
        f"{cik}/us-gaap/{concept}.json"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        _LOG.warning("[EDGAR] fetch failed for %s/%s: %s", cik, concept, exc)
        return []

    units = resp.json().get("units", {}).get("USD", [])
    return [
        {
            "val":   item["val"],
            "start": item.get("start"),
            "end":   item["end"],
            "form":  item["form"],
            "filed": item["filed"],
            "accn":  item["accn"],
        }
        for item in units
        if item.get("form") in ("10-K", "10-Q", "20-F")
    ]


def _try_concepts(cik: str, concepts: list[str]) -> list[dict]:
    """Try each concept in order. Return first non-empty result, or []."""
    for concept in concepts:
        records = _fetch_xbrl(cik, concept)
        time.sleep(0.15)  # stay well under SEC 10 req/s limit
        if records:
            return records
    return []


# ── Period filters ────────────────────────────────────────────────────────────

def _filter_quarterly(records: list[dict]) -> list[dict]:
    """
    Keep records whose period is 75–105 days (standalone quarter).
    Excludes: YTD cumulative (180/270 days), annual (365 days),
              records with no start date (balance-sheet snapshots).
    Deduplicates by end date — keeps the latest 'filed' (amended filings).
    Returns list sorted by end date ascending.
    """
    best: dict[str, dict] = {}
    for r in records:
        if r.get("start") is None:
            continue
        start = datetime.date.fromisoformat(r["start"])
        end   = datetime.date.fromisoformat(r["end"])
        days  = (end - start).days
        if not (75 <= days <= 105):
            continue
        key = r["end"]
        if key not in best or r["filed"] > best[key]["filed"]:
            best[key] = r
    return sorted(best.values(), key=lambda x: x["end"])


def _filter_annual(records: list[dict]) -> list[dict]:
    """
    Keep records whose period is 350–380 days (full fiscal year).
    Used for TSM and ASML (20-F annual filers).
    Deduplicates by end date — keeps latest filed.
    Returns list sorted by end date ascending.
    """
    best: dict[str, dict] = {}
    for r in records:
        if r.get("start") is None:
            continue
        start = datetime.date.fromisoformat(r["start"])
        end   = datetime.date.fromisoformat(r["end"])
        days  = (end - start).days
        if not (350 <= days <= 380):
            continue
        key = r["end"]
        if key not in best or r["filed"] > best[key]["filed"]:
            best[key] = r
    return sorted(best.values(), key=lambda x: x["end"])


# ── Placeholder stubs (implemented in later tasks) ────────────────────────────

def _compute_derived(income: pl.DataFrame, balance: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError("implemented in Task 3")


def _compute_valuation_ratios(df: pl.DataFrame, ticker: str, ohlcv_dir: Path) -> pl.DataFrame:
    raise NotImplementedError("implemented in Task 4")
