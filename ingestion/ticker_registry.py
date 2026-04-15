"""Central registry of all 83 AI infrastructure supply chain tickers.

Single source of truth for layer assignments. CIK_MAP stays in
edgar_fundamentals_ingestion.py (backward-compatible).
"""
from __future__ import annotations

TICKER_LAYERS: dict[str, str] = {
    # Layer 1 — Hyperscalers / Cloud
    "MSFT": "cloud", "AMZN": "cloud", "GOOGL": "cloud",
    "META": "cloud", "ORCL": "cloud", "IBM": "cloud",
    # Layer 2 — AI Compute / Chips
    "NVDA": "compute", "AMD": "compute", "AVGO": "compute",
    "MRVL": "compute", "TSM": "compute", "ASML": "compute",
    "INTC": "compute", "ARM": "compute", "MU": "compute",
    "SNPS": "compute", "CDNS": "compute",
    # Layer 3 — Semiconductor Equipment & Materials
    "AMAT": "semi_equipment", "LRCX": "semi_equipment", "KLAC": "semi_equipment",
    "ENTG": "semi_equipment", "MKSI": "semi_equipment", "UCTT": "semi_equipment",
    "ICHR": "semi_equipment", "TER": "semi_equipment", "ONTO": "semi_equipment",
    "APD": "semi_equipment", "LIN": "semi_equipment",
    # Layer 4 — Networking / Interconnect
    "ANET": "networking", "CSCO": "networking", "CIEN": "networking",
    "COHR": "networking", "LITE": "networking", "INFN": "networking",
    "NOK": "networking", "VIAV": "networking",
    # Layer 5 — Servers / Storage / Systems
    "SMCI": "servers", "DELL": "servers", "HPE": "servers",
    "NTAP": "servers", "PSTG": "servers", "STX": "servers", "WDC": "servers",
    # Layer 6 — Data Center Operators / REITs
    "EQIX": "datacenter", "DLR": "datacenter", "AMT": "datacenter",
    "CCI": "datacenter", "IREN": "datacenter", "APLD": "datacenter",
    # Layer 7 — Power / Energy / Nuclear
    "CEG": "power", "VST": "power", "NRG": "power", "TLN": "power",
    "NEE": "power", "SO": "power", "EXC": "power", "ETR": "power",
    "GEV": "power", "BWX": "power", "OKLO": "power", "SMR": "power",
    "FSLR": "power",
    # Layer 8 — Cooling / Facilities / Backup Power
    "VRT": "cooling", "NVENT": "cooling", "JCI": "cooling",
    "TT": "cooling", "CARR": "cooling", "GNRC": "cooling", "HUBB": "cooling",
    # Layer 9 — Grid / Construction / Electrical Contracting
    "PWR": "grid", "MTZ": "grid", "EME": "grid",
    "MYR": "grid", "IESC": "grid", "AGX": "grid",
    # Layer 10 — Metals / Materials
    "FCX": "metals", "SCCO": "metals", "AA": "metals", "NUE": "metals",
    "STLD": "metals", "MP": "metals", "UUUU": "metals", "ECL": "metals",
}

LAYER_IDS: dict[str, int] = {
    "cloud": 1, "compute": 2, "semi_equipment": 3, "networking": 4,
    "servers": 5, "datacenter": 6, "power": 7, "cooling": 8,
    "grid": 9, "metals": 10,
}

LAYER_LABELS: dict[str, str] = {
    "cloud": "Hyperscalers / Cloud",
    "compute": "AI Compute / Chips",
    "semi_equipment": "Semiconductor Equipment & Materials",
    "networking": "Networking / Interconnect",
    "servers": "Servers / Storage / Systems",
    "datacenter": "Data Center Operators / REITs",
    "power": "Power / Energy / Nuclear",
    "cooling": "Cooling / Facilities / Backup Power",
    "grid": "Grid / Construction / Electrical",
    "metals": "Metals / Materials",
}

# Hyperscalers are the demand root — used for graph hop-distance feature.
HYPERSCALERS: frozenset[str] = frozenset({"MSFT", "AMZN", "GOOGL", "META"})

TICKERS: list[str] = sorted(TICKER_LAYERS.keys())


def tickers_in_layer(layer: str) -> list[str]:
    """Return sorted list of tickers assigned to a given layer name."""
    if layer not in LAYER_IDS:
        raise ValueError(f"Unknown layer {layer!r}. Valid layers: {list(LAYER_IDS)}")
    return sorted(t for t, la in TICKER_LAYERS.items() if la == layer)


def layers() -> list[str]:
    """Return all layer names in ascending layer_id order."""
    return sorted(LAYER_IDS.keys(), key=lambda la: LAYER_IDS[la])
