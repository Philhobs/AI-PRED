"""Central registry of all 127 AI infrastructure + robotics supply chain tickers.

Single source of truth for layer assignments, exchange metadata, and currency.
CIK_MAP stays in edgar_fundamentals_ingestion.py (backward-compatible).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickerInfo:
    """Metadata for a single ticker in the prediction universe."""

    symbol: str    # yfinance-compatible: "NVDA", "ABBN.SW", "6954.T"
    layer: str     # layer name (one of LAYER_IDS keys)
    exchange: str  # "US","DE","PA","SW","MI","CO","ST","OL","L","AS","BR","MC","T"
    currency: str  # ISO 4217: "USD","EUR","CHF","JPY","DKK","SEK","NOK","GBP"
    country: str   # ISO 3166-1 alpha-2


TICKERS_INFO: list[TickerInfo] = [
    # ── Layer 1: Hyperscalers / Cloud (9) ─────────────────────────────────────
    TickerInfo("MSFT",      "cloud",          "US", "USD", "US"),
    TickerInfo("AMZN",      "cloud",          "US", "USD", "US"),
    TickerInfo("GOOGL",     "cloud",          "US", "USD", "US"),
    TickerInfo("META",      "cloud",          "US", "USD", "US"),
    TickerInfo("ORCL",      "cloud",          "US", "USD", "US"),
    TickerInfo("IBM",       "cloud",          "US", "USD", "US"),
    TickerInfo("SAP.DE",    "cloud",          "DE", "EUR", "DE"),
    TickerInfo("CAP.PA",    "cloud",          "PA", "EUR", "FR"),
    TickerInfo("OVH.PA",    "cloud",          "PA", "EUR", "FR"),
    # ── Layer 2: AI Compute / Chips (13) ──────────────────────────────────────
    TickerInfo("NVDA",      "compute",        "US", "USD", "US"),
    TickerInfo("AMD",       "compute",        "US", "USD", "US"),
    TickerInfo("AVGO",      "compute",        "US", "USD", "US"),
    TickerInfo("MRVL",      "compute",        "US", "USD", "US"),
    TickerInfo("TSM",       "compute",        "US", "USD", "TW"),
    TickerInfo("ASML",      "compute",        "US", "USD", "NL"),
    TickerInfo("INTC",      "compute",        "US", "USD", "US"),
    TickerInfo("ARM",       "compute",        "US", "USD", "GB"),
    TickerInfo("MU",        "compute",        "US", "USD", "US"),
    TickerInfo("SNPS",      "compute",        "US", "USD", "US"),
    TickerInfo("CDNS",      "compute",        "US", "USD", "US"),
    TickerInfo("IFX.DE",    "compute",        "DE", "EUR", "DE"),
    TickerInfo("STM",       "compute",        "US", "USD", "NL"),
    # ── Layer 3: Semiconductor Equipment & Materials (15) ─────────────────────
    TickerInfo("AMAT",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("LRCX",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("KLAC",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("ENTG",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("MKSI",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("UCTT",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("ICHR",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("TER",       "semi_equipment", "US", "USD", "US"),
    TickerInfo("ONTO",      "semi_equipment", "US", "USD", "US"),
    TickerInfo("APD",       "semi_equipment", "US", "USD", "US"),
    TickerInfo("LIN",       "semi_equipment", "US", "USD", "IE"),
    TickerInfo("8035.T",    "semi_equipment", "T",  "JPY", "JP"),
    TickerInfo("6920.T",    "semi_equipment", "T",  "JPY", "JP"),
    TickerInfo("ASMI.AS",   "semi_equipment", "AS", "EUR", "NL"),
    TickerInfo("BESI.AS",   "semi_equipment", "AS", "EUR", "NL"),
    # ── Layer 4: Networking / Interconnect (11) ────────────────────────────────
    TickerInfo("ANET",      "networking",     "US", "USD", "US"),
    TickerInfo("CSCO",      "networking",     "US", "USD", "US"),
    TickerInfo("CIEN",      "networking",     "US", "USD", "US"),
    TickerInfo("COHR",      "networking",     "US", "USD", "US"),
    TickerInfo("LITE",      "networking",     "US", "USD", "US"),
    TickerInfo("INFN",      "networking",     "US", "USD", "US"),
    TickerInfo("NOK",       "networking",     "US", "USD", "FI"),
    TickerInfo("VIAV",      "networking",     "US", "USD", "US"),
    TickerInfo("ERIC",      "networking",     "US", "USD", "SE"),
    TickerInfo("JNPR",      "networking",     "US", "USD", "US"),
    TickerInfo("SPT.L",     "networking",     "L",  "GBP", "GB"),
    # ── Layer 5: Servers / Storage / Systems (9) ──────────────────────────────
    TickerInfo("SMCI",      "servers",        "US", "USD", "US"),
    TickerInfo("DELL",      "servers",        "US", "USD", "US"),
    TickerInfo("HPE",       "servers",        "US", "USD", "US"),
    TickerInfo("NTAP",      "servers",        "US", "USD", "US"),
    TickerInfo("PSTG",      "servers",        "US", "USD", "US"),
    TickerInfo("STX",       "servers",        "US", "USD", "IE"),
    TickerInfo("WDC",       "servers",        "US", "USD", "US"),
    TickerInfo("6702.T",    "servers",        "T",  "JPY", "JP"),
    TickerInfo("KTN.DE",    "servers",        "DE", "EUR", "DE"),
    # ── Layer 6: Data Center Operators / REITs (8) ────────────────────────────
    TickerInfo("EQIX",      "datacenter",     "US", "USD", "US"),
    TickerInfo("DLR",       "datacenter",     "US", "USD", "US"),
    TickerInfo("AMT",       "datacenter",     "US", "USD", "US"),
    TickerInfo("CCI",       "datacenter",     "US", "USD", "US"),
    TickerInfo("IREN",      "datacenter",     "US", "USD", "AU"),
    TickerInfo("APLD",      "datacenter",     "US", "USD", "US"),
    TickerInfo("9432.T",    "datacenter",     "T",  "JPY", "JP"),
    TickerInfo("CLNX.MC",   "datacenter",     "MC", "EUR", "ES"),
    # ── Layer 7: Power / Energy / Nuclear (19) ────────────────────────────────
    TickerInfo("CEG",       "power",          "US", "USD", "US"),
    TickerInfo("VST",       "power",          "US", "USD", "US"),
    TickerInfo("NRG",       "power",          "US", "USD", "US"),
    TickerInfo("TLN",       "power",          "US", "USD", "US"),
    TickerInfo("NEE",       "power",          "US", "USD", "US"),
    TickerInfo("SO",        "power",          "US", "USD", "US"),
    TickerInfo("EXC",       "power",          "US", "USD", "US"),
    TickerInfo("ETR",       "power",          "US", "USD", "US"),
    TickerInfo("GEV",       "power",          "US", "USD", "US"),
    TickerInfo("BWX",       "power",          "US", "USD", "US"),
    TickerInfo("OKLO",      "power",          "US", "USD", "US"),
    TickerInfo("SMR",       "power",          "US", "USD", "US"),
    TickerInfo("FSLR",      "power",          "US", "USD", "US"),
    TickerInfo("ENR.DE",    "power",          "DE", "EUR", "DE"),
    TickerInfo("VWS.CO",    "power",          "CO", "DKK", "DK"),
    TickerInfo("RWE.DE",    "power",          "DE", "EUR", "DE"),
    TickerInfo("ENEL.MI",   "power",          "MI", "EUR", "IT"),
    TickerInfo("ORSTED.CO", "power",          "CO", "DKK", "DK"),
    TickerInfo("ENGI.PA",   "power",          "PA", "EUR", "FR"),
    # ── Layer 8: Cooling / Facilities / Backup Power (10) ─────────────────────
    TickerInfo("VRT",       "cooling",        "US", "USD", "US"),
    TickerInfo("NVENT",     "cooling",        "US", "USD", "IE"),
    TickerInfo("JCI",       "cooling",        "US", "USD", "IE"),
    TickerInfo("TT",        "cooling",        "US", "USD", "IE"),
    TickerInfo("CARR",      "cooling",        "US", "USD", "US"),
    TickerInfo("GNRC",      "cooling",        "US", "USD", "US"),
    TickerInfo("HUBB",      "cooling",        "US", "USD", "US"),
    TickerInfo("ALFA.ST",   "cooling",        "ST", "SEK", "SE"),
    TickerInfo("ASETEK.OL", "cooling",        "OL", "NOK", "NO"),
    TickerInfo("SU.PA",     "cooling",        "PA", "EUR", "FR"),
    # ── Layer 9: Grid / Construction / Electrical Contracting (10) ────────────
    TickerInfo("PWR",       "grid",           "US", "USD", "US"),
    TickerInfo("MTZ",       "grid",           "US", "USD", "US"),
    TickerInfo("EME",       "grid",           "US", "USD", "US"),
    TickerInfo("MYR",       "grid",           "US", "USD", "US"),
    TickerInfo("IESC",      "grid",           "US", "USD", "US"),
    TickerInfo("AGX",       "grid",           "US", "USD", "US"),
    TickerInfo("PRY.MI",    "grid",           "MI", "EUR", "IT"),
    TickerInfo("NEX.PA",    "grid",           "PA", "EUR", "FR"),
    TickerInfo("NG.L",      "grid",           "L",  "GBP", "GB"),
    TickerInfo("TRN.MI",    "grid",           "MI", "EUR", "IT"),
    # ── Layer 10: Metals / Materials (12) ─────────────────────────────────────
    TickerInfo("FCX",       "metals",         "US", "USD", "US"),
    TickerInfo("SCCO",      "metals",         "US", "USD", "US"),
    TickerInfo("AA",        "metals",         "US", "USD", "US"),
    TickerInfo("NUE",       "metals",         "US", "USD", "US"),
    TickerInfo("STLD",      "metals",         "US", "USD", "US"),
    TickerInfo("MP",        "metals",         "US", "USD", "US"),
    TickerInfo("UUUU",      "metals",         "US", "USD", "US"),
    TickerInfo("ECL",       "metals",         "US", "USD", "US"),
    TickerInfo("UMI.BR",    "metals",         "BR", "EUR", "BE"),
    TickerInfo("GLEN.L",    "metals",         "L",  "GBP", "CH"),
    TickerInfo("RIO.L",     "metals",         "L",  "GBP", "AU"),
    TickerInfo("5713.T",    "metals",         "T",  "JPY", "JP"),
    # ── Layer 11: Robotics / Automation / Industrial AI (11) ──────────────────
    TickerInfo("ISRG",      "robotics",       "US", "USD", "US"),
    TickerInfo("ROK",       "robotics",       "US", "USD", "US"),
    TickerInfo("ZBRA",      "robotics",       "US", "USD", "US"),
    TickerInfo("CGNX",      "robotics",       "US", "USD", "US"),
    TickerInfo("SYM",       "robotics",       "US", "USD", "US"),
    TickerInfo("ABBN.SW",   "robotics",       "SW", "CHF", "CH"),
    TickerInfo("KGX.DE",    "robotics",       "DE", "EUR", "DE"),
    TickerInfo("HEXA-B.ST", "robotics",       "ST", "SEK", "SE"),
    TickerInfo("6954.T",    "robotics",       "T",  "JPY", "JP"),
    TickerInfo("6506.T",    "robotics",       "T",  "JPY", "JP"),
    TickerInfo("6861.T",    "robotics",       "T",  "JPY", "JP"),
    # ── Layer 12: AI Cybersecurity — Pure Plays (5) ───────────────────────────────
    TickerInfo("CRWD",   "cyber_pureplay", "US", "USD", "US"),
    TickerInfo("ZS",     "cyber_pureplay", "US", "USD", "US"),
    TickerInfo("S",      "cyber_pureplay", "US", "USD", "US"),
    TickerInfo("DARK.L", "cyber_pureplay", "L",  "GBP", "GB"),
    TickerInfo("VRNS",   "cyber_pureplay", "US", "USD", "US"),
    # ── Layer 13: AI Cybersecurity — Platform Vendors (9) ─────────────────────────
    TickerInfo("PANW",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("FTNT",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("CHKP",   "cyber_platform", "US", "USD", "IL"),
    TickerInfo("CYBR",   "cyber_platform", "US", "USD", "IL"),
    TickerInfo("TENB",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("QLYS",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("OKTA",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("AKAM",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("RPD",    "cyber_platform", "US", "USD", "US"),
]

# ── Layer metadata ──────────────────────────────────────────────────────────

LAYER_IDS: dict[str, int] = {
    "cloud": 1, "compute": 2, "semi_equipment": 3, "networking": 4,
    "servers": 5, "datacenter": 6, "power": 7, "cooling": 8,
    "grid": 9, "metals": 10, "robotics": 11, "cyber_pureplay": 12, "cyber_platform": 13,
}

LAYER_LABELS: dict[str, str] = {
    "cloud":          "Hyperscalers / Cloud",
    "compute":        "AI Compute / Chips",
    "semi_equipment": "Semiconductor Equipment & Materials",
    "networking":     "Networking / Interconnect",
    "servers":        "Servers / Storage / Systems",
    "datacenter":     "Data Center Operators / REITs",
    "power":          "Power / Energy / Nuclear",
    "cooling":        "Cooling / Facilities / Backup Power",
    "grid":           "Grid / Construction / Electrical",
    "metals":         "Metals / Materials",
    "robotics":       "Robotics / Automation / Industrial AI",
    "cyber_pureplay": "AI Cybersecurity — Pure Plays",
    "cyber_platform": "AI Cybersecurity — Platform Vendors",
}

# ── Generated lookups (single source of truth: TICKERS_INFO) ───────────────

TICKER_LAYERS:   dict[str, str] = {t.symbol: t.layer    for t in TICKERS_INFO}
TICKER_CURRENCY: dict[str, str] = {t.symbol: t.currency for t in TICKERS_INFO}
TICKER_EXCHANGE: dict[str, str] = {t.symbol: t.exchange for t in TICKERS_INFO}
TICKER_COUNTRY:  dict[str, str] = {t.symbol: t.country  for t in TICKERS_INFO}
TICKERS:         list[str]       = sorted(t.symbol for t in TICKERS_INFO)

# Hyperscalers are the demand root — used for graph hop-distance feature.
HYPERSCALERS: frozenset[str] = frozenset({"MSFT", "AMZN", "GOOGL", "META"})


def tickers_in_layer(layer: str) -> list[str]:
    """Return sorted list of tickers assigned to a given layer name."""
    if layer not in LAYER_IDS:
        raise ValueError(f"Unknown layer {layer!r}. Valid layers: {list(LAYER_IDS)}")
    return sorted(t.symbol for t in TICKERS_INFO if t.layer == layer)


def layers() -> list[str]:
    """Return all layer names in ascending layer_id order."""
    return sorted(LAYER_IDS.keys(), key=lambda la: LAYER_IDS[la])


def non_usd_tickers() -> list[str]:
    """Return sorted list of tickers that trade in non-USD currencies."""
    return sorted(t.symbol for t in TICKERS_INFO if t.currency != "USD")


def ticker_currency(symbol: str) -> str:
    """Return the ISO 4217 currency code for a ticker symbol."""
    try:
        return TICKER_CURRENCY[symbol]
    except KeyError:
        raise KeyError(f"Unknown ticker {symbol!r}. Not in registry.") from None
