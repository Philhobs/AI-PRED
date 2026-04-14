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


def _to_period_series(records: list[dict], value_col: str, annual: bool) -> pl.DataFrame:
    """
    Apply quarterly or annual filter to raw EDGAR records.
    Return DataFrame with [period_end (Date), value_col (Float64)].
    Returns empty DataFrame with correct schema if no records pass the filter.
    """
    filtered = _filter_annual(records) if annual else _filter_quarterly(records)
    if not filtered:
        return pl.DataFrame(
            {"period_end": pl.Series([], dtype=pl.Date),
             value_col:    pl.Series([], dtype=pl.Float64)}
        )
    return pl.DataFrame({
        "period_end": pl.Series(
            [datetime.date.fromisoformat(r["end"]) for r in filtered],
            dtype=pl.Date,
        ),
        value_col: pl.Series(
            [float(r["val"]) for r in filtered],
            dtype=pl.Float64,
        ),
    })


def _build_income_df(cik: str, ticker: str, annual: bool) -> pl.DataFrame:
    """
    Fetch income statement + capex from EDGAR for one ticker.
    Returns DataFrame: [period_end, revenue, gross_profit, operating_income,
                        net_income, capex]
    Missing concepts produce null columns. Returns empty DataFrame if revenue
    data is unavailable (revenue is the required anchor column).
    """
    print(f"[EDGAR] {ticker}: fetching income statement...")

    revenue_records        = _try_concepts(cik, REVENUE_CONCEPTS)
    gross_profit_records   = _fetch_xbrl(cik, "GrossProfit");           time.sleep(0.15)
    op_income_records      = _fetch_xbrl(cik, "OperatingIncomeLoss");   time.sleep(0.15)
    net_income_records     = _try_concepts(cik, NET_INCOME_CONCEPTS)
    capex_records          = _try_concepts(cik, CAPEX_CONCEPTS)

    revenue = _to_period_series(revenue_records, "revenue", annual)
    if revenue.is_empty():
        print(f"[EDGAR] {ticker}: no revenue data found — skipping")
        return pl.DataFrame()

    gross_profit      = _to_period_series(gross_profit_records,  "gross_profit",      annual)
    operating_income  = _to_period_series(op_income_records,     "operating_income",  annual)
    net_income        = _to_period_series(net_income_records,     "net_income",        annual)
    capex             = _to_period_series(capex_records,          "capex",             annual)

    # Outer-join all series on period_end; missing quarters become null
    df = revenue
    for other in [gross_profit, operating_income, net_income, capex]:
        if not other.is_empty():
            df = df.join(other, on="period_end", how="left")
        else:
            col_name = [c for c in other.columns if c != "period_end"][0] if other.columns else None
            if col_name:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

    # Ensure all expected columns exist
    for col in ["gross_profit", "operating_income", "net_income", "capex"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Capex is a cash outflow (negative in XBRL) — store as positive
    if "capex" in df.columns:
        df = df.with_columns(pl.col("capex").abs())

    return df.sort("period_end")


def _build_balance_df(cik: str, ticker: str) -> pl.DataFrame:
    """
    Fetch balance sheet snapshot items from EDGAR for one ticker.
    Balance sheet items have no start date — deduplicate by end date (latest filed).
    Returns DataFrame: [period_end, equity, long_term_debt,
                        current_assets, current_liabilities, shares_outstanding]
    """
    print(f"[EDGAR] {ticker}: fetching balance sheet...")

    def _dedup_snapshot(records: list[dict], value_col: str) -> pl.DataFrame:
        """Deduplicate snapshot (no start) records by end date, keep latest filed."""
        best: dict[str, dict] = {}
        for r in records:
            key = r["end"]
            if key not in best or r["filed"] > best[key]["filed"]:
                best[key] = r
        filtered = sorted(best.values(), key=lambda x: x["end"])
        if not filtered:
            return pl.DataFrame({
                "period_end": pl.Series([], dtype=pl.Date),
                value_col:    pl.Series([], dtype=pl.Float64),
            })
        return pl.DataFrame({
            "period_end": pl.Series(
                [datetime.date.fromisoformat(r["end"]) for r in filtered],
                dtype=pl.Date,
            ),
            value_col: pl.Series(
                [float(r["val"]) for r in filtered],
                dtype=pl.Float64,
            ),
        })

    equity_records     = _try_concepts(cik, EQUITY_CONCEPTS)
    debt_records       = _try_concepts(cik, DEBT_CONCEPTS)
    cur_assets_records = _fetch_xbrl(cik, "AssetsCurrent");       time.sleep(0.15)
    cur_liab_records   = _fetch_xbrl(cik, "LiabilitiesCurrent");  time.sleep(0.15)
    shares_records     = _try_concepts(cik, SHARES_CONCEPTS)

    equity       = _dedup_snapshot(equity_records,     "equity")
    debt         = _dedup_snapshot(debt_records,       "long_term_debt")
    cur_assets   = _dedup_snapshot(cur_assets_records, "current_assets")
    cur_liab     = _dedup_snapshot(cur_liab_records,   "current_liabilities")
    shares       = _dedup_snapshot(shares_records,     "shares_outstanding")

    if equity.is_empty():
        print(f"[EDGAR] {ticker}: no equity data — balance sheet will be null")
        return pl.DataFrame()

    df = equity
    for other in [debt, cur_assets, cur_liab, shares]:
        if not other.is_empty():
            df = df.join(other, on="period_end", how="left")
        else:
            col_name = [c for c in other.columns if c != "period_end"][0] if other.columns else None
            if col_name:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

    for col in ["long_term_debt", "current_assets", "current_liabilities", "shares_outstanding"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return df.sort("period_end")


# ── Placeholder stubs (implemented in later tasks) ────────────────────────────

def _compute_derived(income: pl.DataFrame, balance: pl.DataFrame) -> pl.DataFrame:
    """
    Join income + balance sheet on period_end (left join — keep all income dates).
    Compute 6 derived metrics + revenue_growth_yoy (4-quarter lag).

    Returns DataFrame: [period_end, revenue, net_income, shares_outstanding,
                        revenue_growth_yoy, gross_margin, operating_margin,
                        capex_to_revenue, debt_to_equity, current_ratio]
    plus income/balance columns needed for valuation ratios downstream.
    """
    if income.is_empty() or balance.is_empty():
        return pl.DataFrame()

    df = income.join(balance, on="period_end", how="left").sort("period_end")

    # Revenue YoY: shift(4) gives the value 4 rows back (same quarter prior year)
    df = df.with_columns(
        pl.col("revenue").shift(4).alias("revenue_4q_prior")
    )

    df = df.with_columns([
        # revenue_growth_yoy
        pl.when(
            pl.col("revenue_4q_prior").is_not_null() & (pl.col("revenue_4q_prior") != 0)
        )
        .then(
            (pl.col("revenue") - pl.col("revenue_4q_prior")) / pl.col("revenue_4q_prior").abs()
        )
        .otherwise(None)
        .alias("revenue_growth_yoy"),

        # gross_margin
        pl.when(pl.col("revenue").is_not_null() & (pl.col("revenue") != 0))
        .then(pl.col("gross_profit") / pl.col("revenue"))
        .otherwise(None)
        .alias("gross_margin"),

        # operating_margin
        pl.when(pl.col("revenue").is_not_null() & (pl.col("revenue") != 0))
        .then(pl.col("operating_income") / pl.col("revenue"))
        .otherwise(None)
        .alias("operating_margin"),

        # capex_to_revenue
        pl.when(pl.col("revenue").is_not_null() & (pl.col("revenue") != 0))
        .then(pl.col("capex") / pl.col("revenue"))
        .otherwise(None)
        .alias("capex_to_revenue"),

        # debt_to_equity
        pl.when(pl.col("equity").is_not_null() & (pl.col("equity") > 0))
        .then(pl.col("long_term_debt") / pl.col("equity"))
        .otherwise(None)
        .alias("debt_to_equity"),

        # current_ratio
        pl.when(
            pl.col("current_liabilities").is_not_null() & (pl.col("current_liabilities") != 0)
        )
        .then(pl.col("current_assets") / pl.col("current_liabilities"))
        .otherwise(None)
        .alias("current_ratio"),
    ])

    return df.drop("revenue_4q_prior")


def _compute_valuation_ratios(df: pl.DataFrame, ticker: str, ohlcv_dir: Path) -> pl.DataFrame:
    raise NotImplementedError("implemented in Task 4")
