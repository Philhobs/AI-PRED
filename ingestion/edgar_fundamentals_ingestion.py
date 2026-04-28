"""
EDGAR XBRL Historical Fundamentals Ingestion

Fetches ~12–15 years of quarterly financial data from SEC EDGAR for all 83
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
    # ── Existing 24 tickers (unchanged) ──────────────────────────────────────
    "MSFT":  "0000789019",
    "AMZN":  "0001018724",
    "GOOGL": "0001652044",
    "META":  "0001326801",
    "NVDA":  "0001045810",
    "AMD":   "0000002488",
    "AVGO":  "0001730168",
    "MRVL":  "0001835632",   # 2025 audit: was 0001058057 (5y stale, last 8-K 2021-04)
    "TSM":   "0001046179",
    "ASML":  "0000937556",
    "AMAT":  "0000796343",
    "LRCX":  "0000707549",
    "KLAC":  "0000319201",
    "VRT":   "0001674101",   # 2025 audit: was 0001748157 (no 8-Ks in old CIK)
    "SMCI":  "0000910638",
    "DELL":  "0001571996",
    "HPE":   "0001645590",
    "EQIX":  "0001101239",
    "DLR":   "0001297996",
    "AMT":   "0001053507",
    "CEG":   "0001868275",
    "VST":   "0001692819",
    "NRG":   "0001013871",
    "TLN":   "0001622536",   # post-Chapter-11 entity (Talen Energy Corp); old CIK 0000099590 was 404
    # ── Layer 1 — New Cloud tickers ───────────────────────────────────────────
    "ORCL":  "0001341439",
    "IBM":   "0000051143",
    # ── Layer 2 — New Compute tickers ─────────────────────────────────────────
    "INTC":  "0000050863",
    "MU":    "0000723254",
    "SNPS":  "0000883241",
    "CDNS":  "0000813672",
    # ARM (0001980994) and ASML already present — foreign filers, no Form 4
    # ── Layer 3 — Semiconductor Equipment & Materials ─────────────────────────
    "ENTG":  "0001101302",   # current Entegris Inc registration; old CIK 0001101781 was 404
    "MKSI":  "0000062996",
    "UCTT":  "0001275014",
    "ICHR":  "0001677576",
    "TER":   "0000097476",
    "ONTO":  "0000704532",   # 2025 audit: was 0000315374 (36d stale)
    "APD":   "0000002969",
    "LIN":   "0001707925",   # 2025 audit: was 0001707092 (5y stale, last 8-K 2020-10)
    # ── Layer 4 — Networking / Interconnect ───────────────────────────────────
    "ANET":  "0001596532",   # 2025 audit: was 0001313925 (no 8-Ks in old CIK)
    "CSCO":  "0000858877",
    "CIEN":  "0000936395",
    "COHR":  "0000820318",
    "LITE":  "0001633978",   # 2025 audit: was 0001439231 (no 8-Ks in old CIK)
    "INFN":  "0001101680",
    "VIAV":  "0000912093",   # 2025 audit: was 0000936744 (no 8-Ks in old CIK)
    # NOK is a foreign private issuer — no Form 4
    # ── Layer 5 — Servers / Storage / Systems ─────────────────────────────────
    "NTAP":  "0001002047",   # 2025 audit: was 0001108320 (5y stale, last 8-K 2021-04)
    "PSTG":  "0001474432",
    "STX":   "0001137789",
    "WDC":   "0000106040",
    # ── Layer 6 — Data Center Operators / REITs ───────────────────────────────
    "CCI":   "0001051512",
    "APLD":  "0001070050",
    # IREN is an Australian company — no US Form 4 filings
    # ── Layer 7 — Power / Energy / Nuclear ────────────────────────────────────
    "NEE":   "0000753308",
    "SO":    "0000092122",
    "EXC":   "0001109357",
    "ETR":   "0000049600",
    "GEV":   "0001996810",   # 2025 audit: was 0001986936 (no 8-Ks in old CIK)
    "BWX":   "0001643953",
    "OKLO":  "0001849056",   # 2025 audit: was 0001840198 (no 8-Ks in old CIK)
    "SMR":   "0001822928",
    "FSLR":  "0001274494",
    # ── Layer 8 — Cooling / Facilities / Backup Power ─────────────────────────
    "NVENT": "0001681903",
    "JCI":   "0000833444",
    "TT":    "0001466258",
    "CARR":  "0001783180",   # 2025 audit: was 0001783398 (31d stale)
    "GNRC":  "0001474735",
    "HUBB":  "0000048898",
    # ── Layer 9 — Grid / Construction / Electrical ────────────────────────────
    "PWR":   "0001050915",   # 2025 audit: was 0001108827 (5y stale, last 8-K 2021-03)
    "MTZ":   "0000015615",
    "EME":   "0000105634",
    "MYR":   "0000700923",
    "IESC":  "0001048268",   # 2025 audit: was 0000049588 (24y stale, last 8-K 2002-03)
    "AGX":   "0000100591",   # 2025 audit: was 0001068875 (~2y stale, last 8-K 2023-11)
    # ── Layer 10 — Metals / Materials ─────────────────────────────────────────
    "FCX":   "0000831259",
    "SCCO":  "0001001838",   # 2025 audit: was 0001001290 (no 8-Ks in old CIK)
    "AA":    "0000004281",
    "NUE":   "0000073309",
    "STLD":  "0001022671",   # 2025 audit: was 0001022652 (61d stale)
    "MP":    "0001801368",   # 2025 audit: was 0001801762 (~3y stale, last 8-K 2022-11)
    "UUUU":  "0001477845",
    "ECL":   "0000031462",
    # ── Strategy alignment additions (2026-04-26): power, cooling, cyber, SaaS, govt-AI ──
    "CCJ":   "0001009001",   # Cameco — uranium fuel cycle
    "ETN":   "0001551182",   # Eaton — DC electrical equipment
    "NET":   "0001477333",   # Cloudflare — edge security + AI Workers
    "PLTR":  "0001321655",   # Palantir — govt + commercial AI OS
    "NOW":   "0001373715",   # ServiceNow
    "CRM":   "0001108524",   # Salesforce — Agentforce
    "ADBE":  "0000796343",   # Adobe
    "INTU":  "0000896878",   # Intuit
    "DDOG":  "0001561550",   # Datadog
    "SNOW":  "0001640147",   # Snowflake — Cortex AI
    "GTLB":  "0001653482",   # GitLab
    "TEAM":  "0001650372",   # Atlassian
    "PATH":  "0001734722",   # UiPath
    "MNDY":  "0001845338",   # Monday.com
    # ── 2026-04-27 watchlist coverage gap fill (looked up via SEC company_tickers.json) ──
    "ADI":   "0000006281",   # Analog Devices
    "AKAM":  "0001086222",   # Akamai Technologies
    "ARM":   "0001973239",   # Arm Holdings ADR
    "CGNX":  "0000851205",   # Cognex Corp
    "CHKP":  "0001015922",   # Check Point Software
    "CRWD":  "0001535527",   # CrowdStrike Holdings
    "EMR":   "0000032604",   # Emerson Electric
    "ERIC":  "0000717826",   # Ericsson ADR
    "FTNT":  "0001262039",   # Fortinet
    "IREN":  "0001878848",   # Iris Energy
    "ISRG":  "0001035267",   # Intuitive Surgical
    "MCHP":  "0000827054",   # Microchip Technology
    "NOK":   "0000924613",   # Nokia ADR
    "OKTA":  "0001660134",   # Okta
    "PANW":  "0001327567",   # Palo Alto Networks
    "QLYS":  "0001107843",   # Qualys
    "ROK":   "0001024478",   # Rockwell Automation
    "RPD":   "0001560327",   # Rapid7
    "S":     "0001583708",   # SentinelOne
    "STM":   "0000932787",   # STMicroelectronics ADR
    "SYM":   "0001837240",   # Symbotic
    "TENB":  "0001660280",   # Tenable Holdings
    "TSLA":  "0001318605",   # Tesla
    "TXN":   "0000097476",   # Texas Instruments
    "VRNS":  "0001361113",   # Varonis Systems
    "ZBRA":  "0000877212",   # Zebra Technologies
    "ZS":    "0001713683",   # Zscaler
    # CYBR (CyberArk) and JNPR (Juniper, acquired by HPE 2025) are not in
    # SEC company_tickers.json. Add manually if they reappear.
    # ── 2026-04-28 medical-robotics expansion ────────────────────────────────
    "SYK":   "0000310764",   # Stryker Corp — Mako ortho robot
    "MDT":   "0001613103",   # Medtronic plc — Hugo surgical
    "GMED":  "0001237831",   # Globus Medical — ExcelsiusGPS spine
    "PRCT":  "0001588978",   # PROCEPT BioRobotics — AquaBeam
    # ── 2026-04-28 layer-breadth expansion (25 tickers across 8 layers) ──────
    # Networking
    "HLIT":  "0000851310",   # Harmonic
    "CALX":  "0001406666",   # Calix
    "AAOI":  "0001158114",   # Applied Optoelectronics
    "EXTR":  "0001078271",   # Extreme Networks
    # Servers
    "CDW":   "0001402057",   # CDW Corp
    "ARW":   "0000007536",   # Arrow Electronics
    # Datacenter
    "SBAC":  "0001034054",   # SBA Communications
    "DBRG":  "0001679688",   # DigitalBridge
    "GLW":   "0000024741",   # Corning
    "DOCN":  "0001582961",   # DigitalOcean
    # Power
    "DUK":   "0001326160",   # Duke Energy
    "AEP":   "0000004904",   # American Electric Power
    "XEL":   "0000072903",   # Xcel Energy
    "LEU":   "0001065059",   # Centrus Energy
    "PLUG":  "0001093691",   # Plug Power
    # Grid
    "FLR":   "0001124198",   # Fluor
    "ACM":   "0000868857",   # AECOM
    "KBR":   "0001357615",   # KBR Inc
    # Cyber pureplay
    "VRSN":  "0001014473",   # Verisign — DNS infra
    # Cyber platform
    "LDOS":  "0001336920",   # Leidos
    "CACI":  "0000016058",   # CACI International
    "BAH":   "0001443646",   # Booz Allen Hamilton
    # Robotics MCU/sensor chips
    "ON":    "0001097864",   # onsemi
    "NXPI":  "0001413447",   # NXP Semiconductors
    "MPWR":  "0001280452",   # Monolithic Power Systems
}

ANNUAL_FILERS: set[str] = {"TSM", "ASML", "NOK", "ARM"}

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
    pa.field("ticker",                pa.string()),
    pa.field("period_end",            pa.date32()),
    pa.field("pe_ratio_trailing",     pa.float64()),
    pa.field("price_to_sales",        pa.float64()),
    pa.field("price_to_book",         pa.float64()),
    pa.field("revenue_growth_yoy",    pa.float64()),
    pa.field("gross_margin",          pa.float64()),
    pa.field("operating_margin",      pa.float64()),
    pa.field("capex_to_revenue",      pa.float64()),
    pa.field("debt_to_equity",        pa.float64()),
    pa.field("current_ratio",         pa.float64()),
    # 5 new TTM-based metrics
    pa.field("net_income_margin",     pa.float64()),
    pa.field("free_cash_flow_margin", pa.float64()),
    pa.field("capex_growth_yoy",      pa.float64()),
    pa.field("revenue_growth_accel",  pa.float64()),
    pa.field("research_to_revenue",   pa.float64()),
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
        f"CIK{cik}/us-gaap/{concept}.json"
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
    Fetch income statement + capex + R&D from EDGAR for one ticker.
    Returns DataFrame: [period_end, revenue, gross_profit, operating_income,
                        net_income, capex, rd_expense]
    Missing concepts produce null columns (rd_expense = null when concept returns 404).
    Returns empty DataFrame if revenue data is unavailable.
    """
    print(f"[EDGAR] {ticker}: fetching income statement...")

    revenue_records        = _try_concepts(cik, REVENUE_CONCEPTS)
    gross_profit_records   = _fetch_xbrl(cik, "GrossProfit");                   time.sleep(0.15)
    op_income_records      = _fetch_xbrl(cik, "OperatingIncomeLoss");           time.sleep(0.15)
    net_income_records     = _try_concepts(cik, NET_INCOME_CONCEPTS)
    capex_records          = _try_concepts(cik, CAPEX_CONCEPTS)
    rd_records             = _fetch_xbrl(cik, "ResearchAndDevelopmentExpense"); time.sleep(0.15)

    revenue = _to_period_series(revenue_records, "revenue", annual)
    if revenue.is_empty():
        print(f"[EDGAR] {ticker}: no revenue data found — skipping")
        return pl.DataFrame()

    gross_profit      = _to_period_series(gross_profit_records,  "gross_profit",      annual)
    operating_income  = _to_period_series(op_income_records,     "operating_income",  annual)
    net_income        = _to_period_series(net_income_records,     "net_income",        annual)
    capex             = _to_period_series(capex_records,          "capex",             annual)
    rd_expense        = _to_period_series(rd_records,             "rd_expense",        annual)

    # Outer-join all series on period_end; missing quarters become null
    df = revenue
    for other in [gross_profit, operating_income, net_income, capex, rd_expense]:
        if not other.is_empty():
            df = df.join(other, on="period_end", how="left")
        else:
            col_name = [c for c in other.columns if c != "period_end"][0] if other.columns else None
            if col_name:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

    # Ensure all expected columns exist
    for col in ["gross_profit", "operating_income", "net_income", "capex", "rd_expense"]:
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
    Compute 11 derived metrics: 6 ratio metrics + revenue_growth_yoy (4-quarter
    calendar join) + 5 TTM-based metrics (net_income_margin, free_cash_flow_margin,
    capex_growth_yoy, revenue_growth_accel, research_to_revenue).

    Returns DataFrame with income/balance columns plus:
        revenue_growth_yoy, gross_margin, operating_margin,
        capex_to_revenue, debt_to_equity, current_ratio,
        net_income_margin, free_cash_flow_margin, capex_growth_yoy,
        revenue_growth_accel, research_to_revenue
    (income/balance columns retained for valuation ratio computation downstream)
    """
    if income.is_empty() or balance.is_empty():
        return pl.DataFrame()

    df = income.join(balance, on="period_end", how="left").sort("period_end")

    # Revenue YoY: calendar-aware join on year+quarter to handle gaps in history.
    # shift(4) is position-based and produces wrong results when quarters are missing.
    df = df.with_columns([
        pl.col("period_end").dt.year().alias("_year"),
        pl.col("period_end").dt.quarter().alias("_quarter"),
    ])

    prior_year = df.select([
        (pl.col("_year") + 1).alias("_year"),   # prior year → current year
        pl.col("_quarter"),
        pl.col("revenue").alias("revenue_4q_prior"),
    ])

    df = df.join(prior_year, on=["_year", "_quarter"], how="left")
    df = df.drop(["_year", "_quarter"])

    df = df.with_columns([
        # revenue_growth_yoy
        pl.when(
            pl.col("revenue_4q_prior").is_not_null() & (pl.col("revenue_4q_prior") > 0)
        )
        .then(
            (pl.col("revenue") - pl.col("revenue_4q_prior")) / pl.col("revenue_4q_prior")
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
            pl.col("current_liabilities").is_not_null() & (pl.col("current_liabilities") > 0)
        )
        .then(pl.col("current_assets") / pl.col("current_liabilities"))
        .otherwise(None)
        .alias("current_ratio"),
    ])

    df = df.drop("revenue_4q_prior")

    # ── 5 new TTM-based metrics ───────────────────────────────────────────────
    df = df.sort("period_end").with_columns([
        pl.col("net_income").rolling_sum(window_size=4, min_samples=4).alias("_ttm_net_income"),
        pl.col("operating_income").rolling_sum(window_size=4, min_samples=4).alias("_ttm_op_income"),
        pl.col("capex").rolling_sum(window_size=4, min_samples=4).alias("_ttm_capex"),
        pl.col("revenue").rolling_sum(window_size=4, min_samples=4).alias("_ttm_revenue"),
        pl.col("rd_expense").fill_null(0.0).rolling_sum(window_size=4, min_samples=4).alias("_ttm_rd"),
    ])

    df = df.with_columns(
        pl.col("_ttm_capex").shift(4).alias("_prior_ttm_capex")
    )

    df = df.with_columns([
        # net_income_margin: 0.0 when TTM revenue unavailable
        pl.when(
            pl.col("_ttm_revenue").is_not_null() & (pl.col("_ttm_revenue") != 0)
            & pl.col("_ttm_net_income").is_not_null()
        )
        .then(pl.col("_ttm_net_income") / pl.col("_ttm_revenue"))
        .otherwise(0.0)
        .alias("net_income_margin"),

        # free_cash_flow_margin: 0.0 when TTM revenue unavailable
        pl.when(
            pl.col("_ttm_revenue").is_not_null() & (pl.col("_ttm_revenue") != 0)
            & pl.col("_ttm_op_income").is_not_null()
            & pl.col("_ttm_capex").is_not_null()
        )
        .then((pl.col("_ttm_op_income") - pl.col("_ttm_capex")) / pl.col("_ttm_revenue"))
        .otherwise(0.0)
        .alias("free_cash_flow_margin"),

        # capex_growth_yoy: null when <8 quarters of capex history
        pl.when(
            pl.col("_prior_ttm_capex").is_not_null() & (pl.col("_prior_ttm_capex") > 0)
            & pl.col("_ttm_capex").is_not_null()
        )
        .then((pl.col("_ttm_capex") / pl.col("_prior_ttm_capex")) - 1.0)
        .otherwise(None)
        .alias("capex_growth_yoy"),

        # research_to_revenue: 0.0 when R&D concept unavailable (rd_expense = 0/null)
        pl.when(pl.col("_ttm_revenue").is_not_null() & (pl.col("_ttm_revenue") != 0))
        .then(pl.col("_ttm_rd") / pl.col("_ttm_revenue"))
        .otherwise(0.0)
        .alias("research_to_revenue"),
    ])

    # revenue_growth_accel: second derivative of YoY growth
    df = df.with_columns(
        pl.col("revenue_growth_yoy").shift(1).alias("_prior_yoy")
    )
    df = df.with_columns(
        pl.when(
            pl.col("revenue_growth_yoy").is_not_null()
            & pl.col("_prior_yoy").is_not_null()
        )
        .then(pl.col("revenue_growth_yoy") - pl.col("_prior_yoy"))
        .otherwise(0.0)
        .alias("revenue_growth_accel")
    )

    df = df.drop(["_ttm_net_income", "_ttm_op_income", "_ttm_capex", "_ttm_revenue",
                  "_ttm_rd", "_prior_ttm_capex", "_prior_yoy"])

    return df


def _load_ohlcv(ticker: str, ohlcv_dir: Path) -> pl.DataFrame:
    """
    Load full OHLCV price history for one ticker.
    Returns DataFrame [date (Date), close_price (Float64)] sorted by date.
    Returns empty DataFrame if no parquet files found.
    """
    ticker_dir = ohlcv_dir / ticker
    files = list(ticker_dir.glob("*.parquet")) if ticker_dir.exists() else []
    if not files:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date),
                             "close_price": pl.Series([], dtype=pl.Float64)})
    return (
        pl.concat([pl.read_parquet(str(f)) for f in files])
        .select(["date", "close_price"])
        .with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
    )


def _compute_valuation_ratios(
    df: pl.DataFrame,
    ticker: str,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """
    Add pe_ratio_trailing, price_to_sales, price_to_book to df.

    For each period_end:
      - Look up close_price (backward asof join on OHLCV)
      - market_cap = close_price × shares_outstanding
      - TTM_revenue = rolling_sum(4) of quarterly revenues
      - TTM_net_income = rolling_sum(4) of quarterly net incomes
      - pe_ratio_trailing = market_cap / TTM_net_income  (null if <= 0)
      - price_to_sales    = market_cap / TTM_revenue      (null if <= 0)
      - price_to_book     = market_cap / equity            (null if <= 0)
    """
    ohlcv = _load_ohlcv(ticker, ohlcv_dir)

    if ohlcv.is_empty() or "shares_outstanding" not in df.columns:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("pe_ratio_trailing"),
            pl.lit(None).cast(pl.Float64).alias("price_to_sales"),
            pl.lit(None).cast(pl.Float64).alias("price_to_book"),
        ])

    # TTM income figures via rolling sum of 4 quarters
    df = df.sort("period_end").with_columns([
        pl.col("revenue").rolling_sum(window_size=4, min_samples=4).alias("ttm_revenue"),
        pl.col("net_income").rolling_sum(window_size=4, min_samples=4).alias("ttm_net_income"),
    ])

    # Backward asof join: for each period_end, find most recent close_price
    df = df.join_asof(
        ohlcv.rename({"date": "period_end"}),
        on="period_end",
        strategy="backward",
    )

    # market_cap
    df = df.with_columns(
        (pl.col("close_price") * pl.col("shares_outstanding")).alias("market_cap")
    )

    # Valuation ratios
    df = df.with_columns([
        pl.when(
            pl.col("ttm_net_income").is_not_null() & (pl.col("ttm_net_income") > 0)
            & pl.col("market_cap").is_not_null()
        )
        .then(pl.col("market_cap") / pl.col("ttm_net_income"))
        .otherwise(None)
        .alias("pe_ratio_trailing"),

        pl.when(
            pl.col("ttm_revenue").is_not_null() & (pl.col("ttm_revenue") > 0)
            & pl.col("market_cap").is_not_null()
        )
        .then(pl.col("market_cap") / pl.col("ttm_revenue"))
        .otherwise(None)
        .alias("price_to_sales"),

        pl.when(
            pl.col("equity").is_not_null() & (pl.col("equity") > 0)
            & pl.col("market_cap").is_not_null()
        )
        .then(pl.col("market_cap") / pl.col("equity"))
        .otherwise(None)
        .alias("price_to_book"),
    ])

    return df.drop(["market_cap", "ttm_revenue", "ttm_net_income",
                    "close_price"], strict=False)


def fetch_edgar_fundamentals(ticker: str, ohlcv_dir: Path) -> pl.DataFrame:
    """
    Main per-ticker entry point.

    Fetches income statement + balance sheet from SEC EDGAR, computes 6 derived
    metrics and 3 valuation ratios, and returns a DataFrame matching the schema
    of fundamental_ingestion.py output.

    Returns empty DataFrame if the ticker has no EDGAR data (e.g. too new).
    """
    cik = CIK_MAP.get(ticker)
    if cik is None:
        _LOG.warning("[EDGAR] %s: no CIK in map — skipping", ticker)
        return pl.DataFrame()

    annual = ticker in ANNUAL_FILERS

    income  = _build_income_df(cik, ticker, annual)
    balance = _build_balance_df(cik, ticker)

    if income.is_empty():
        return pl.DataFrame()

    df = _compute_derived(income, balance)
    if df.is_empty():
        return pl.DataFrame()

    df = _compute_valuation_ratios(df, ticker, ohlcv_dir)

    # Select and rename to final output schema
    output_cols = [
        "period_end", "pe_ratio_trailing", "price_to_sales", "price_to_book",
        "revenue_growth_yoy", "gross_margin", "operating_margin",
        "capex_to_revenue", "debt_to_equity", "current_ratio",
        # 5 new TTM metrics
        "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
        "revenue_growth_accel", "research_to_revenue",
    ]
    # Only keep columns that exist (some may be null if EDGAR had no data)
    present = [c for c in output_cols if c in df.columns]
    missing = [c for c in output_cols if c not in df.columns]

    df = df.select(present)
    for col in missing:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    df = df.select(output_cols).with_columns(
        pl.lit(ticker).alias("ticker")
    ).select(["ticker"] + output_cols)

    _LOG.info("[EDGAR] %s: %d periods, %s → %s",
              ticker, len(df), df["period_end"].min(), df["period_end"].max())
    return df


def save_edgar_fundamentals(df: pl.DataFrame, ticker: str, output_dir: Path) -> None:
    """
    Write to data/raw/financials/fundamentals/<TICKER>/quarterly.parquet
    (overwrites yfinance-only data). Uses same PyArrow schema as
    fundamental_ingestion.py so fundamental_features.py needs no changes.
    """
    if df.is_empty():
        _LOG.warning("[EDGAR] %s: no data to write", ticker)
        return

    path = output_dir / "financials" / "fundamentals" / ticker / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(df.to_dicts(), schema=_SCHEMA)
    pq.write_table(table, str(path), compression="snappy")
    _LOG.info("[EDGAR] %s: %d rows → %s", ticker, len(df), path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(message)s")
    from dotenv import load_dotenv
    load_dotenv()

    ohlcv_dir  = Path("data/raw/financials/ohlcv")
    output_dir = Path("data/raw")

    tickers = list(CIK_MAP.keys())
    print(f"[EDGAR] Fetching fundamentals for {len(tickers)} tickers from SEC EDGAR...")
    print("[EDGAR] Expected runtime: ~5 minutes (rate-limited to SEC fair use)")

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[EDGAR] [{i}/{len(tickers)}] {ticker}")
        df = fetch_edgar_fundamentals(ticker, ohlcv_dir)
        save_edgar_fundamentals(df, ticker, output_dir)
        time.sleep(1)  # 1s between tickers (on top of per-concept sleeps)

    print("\n[EDGAR] Done. Run `python models/train.py` to retrain with historical fundamentals.")
