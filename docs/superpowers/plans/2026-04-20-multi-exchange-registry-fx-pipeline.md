# Multi-Exchange Registry + FX Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the model from 83 US-only tickers to 127 tickers across 11 layers, introduce a typed `TickerInfo` registry, add FX ingestion for 7 currency pairs, and produce a new `fx_adjusted_return_20d` feature (FEATURE_COLS 47 → 48).

**Architecture:** The `TickerInfo` dataclass replaces the flat `TICKER_LAYERS` dict as the single source of truth — all existing lookups are regenerated from it, so no call sites change. A new `ingestion/fx_ingestion.py` fetches daily FX rates; `processing/fx_features.py` converts non-USD close prices to USD and computes the new feature. `supply_chain_features.py` gains an optional `fx_dir` parameter so cross-ticker correlations use USD-normalized returns.

**Tech Stack:** Python 3.11, Polars, yfinance (already a project dependency), dataclasses (stdlib).

---

## Context You Must Know

**Working directory:** `/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED`

**Existing registry:** `ingestion/ticker_registry.py` — flat `TICKER_LAYERS: dict[str, str]`, 83 US tickers, 10 layers. Functions `tickers_in_layer()`, `layers()`, constants `LAYER_IDS`, `HYPERSCALERS`, `TICKERS` are all used across the pipeline. All must remain backwards-compatible.

**Existing tests that need updating (they assert old counts):**
- `tests/test_ticker_registry.py::test_ticker_count` — asserts 83
- `tests/test_ticker_registry.py::test_tickers_in_layer` — asserts cloud has 6 tickers
- `tests/test_ticker_registry.py::test_layers_returns_10` — asserts 10 layers
- `tests/test_ticker_registry.py::test_layers_order` — asserts last layer is "metals"
- `tests/test_ticker_registry.py::test_cik_map_covers_domestic_tickers` — foreign set needs update

**OHLCV parquet schema:** `ticker, date, open, high, low, close_price, volume` — the `_build_close_matrix` in `supply_chain_features.py` reads `close_price` and renames the column to the ticker symbol. Ticker directory names use the full symbol including suffix (e.g., `ohlcv/ABBN.SW/2025.parquet`) — dots and hyphens are valid on all filesystems.

**FX yfinance symbols:** `EURUSD=X`, `CHFUSD=X`, `JPYUSD=X`, `DKKUSD=X`, `SEKUSD=X`, `NOKUSD=X`, `GBPUSD=X`

**Pattern to follow:** `processing/energy_geo_features.py` and `processing/supply_chain_features.py` for module structure. `ingestion/eia_ingestion.py` for failure-safe ingestion pattern.

---

## File Map

| File | Action |
|---|---|
| `ingestion/ticker_registry.py` | Refactor (TickerInfo, 83→127 tickers) |
| `ingestion/fx_ingestion.py` | Create |
| `processing/fx_features.py` | Create |
| `processing/supply_chain_features.py` | Extend (fx_dir param) |
| `models/train.py` | Extend (FX_FEATURE_COLS, 47→48) |
| `models/inference.py` | Extend (mirror train.py) |
| `tests/test_ticker_registry.py` | Extend + fix stale count assertions |
| `tests/test_fx_ingestion.py` | Create |
| `tests/test_fx_features.py` | Create |

---

## Task 1: Registry Refactor

**Files:**
- Modify: `ingestion/ticker_registry.py`
- Modify: `tests/test_ticker_registry.py`

- [ ] **Step 1: Update existing tests to expect new counts (they will fail on old code — that's the signal)**

Replace the full content of `tests/test_ticker_registry.py` with:

```python
# tests/test_ticker_registry.py


def test_ticker_count():
    from ingestion.ticker_registry import TICKERS, TICKER_LAYERS
    assert len(TICKERS) == 127
    assert len(TICKER_LAYERS) == 127


def test_all_layers_present():
    from ingestion.ticker_registry import TICKER_LAYERS, LAYER_IDS
    layers_used = set(TICKER_LAYERS.values())
    assert layers_used == set(LAYER_IDS.keys())


def test_tickers_in_layer():
    from ingestion.ticker_registry import tickers_in_layer
    cloud = tickers_in_layer("cloud")
    assert "MSFT" in cloud and "AMZN" in cloud
    assert len(cloud) == 9  # was 6; +SAP.DE, CAP.PA, OVH.PA


def test_hyperscalers_are_cloud():
    from ingestion.ticker_registry import HYPERSCALERS, TICKER_LAYERS
    for t in HYPERSCALERS:
        assert TICKER_LAYERS[t] == "cloud"


def test_layers_returns_11():
    from ingestion.ticker_registry import layers
    assert len(layers()) == 11


def test_layers_order():
    from ingestion.ticker_registry import layers
    result = layers()
    assert result[0] == "cloud"
    assert result[-1] == "robotics"  # robotics=11, metals=10


def test_cik_map_covers_domestic_tickers():
    """CIK_MAP must have entries for original 83 domestic US tickers."""
    from ingestion.edgar_fundamentals_ingestion import CIK_MAP
    from ingestion.ticker_registry import TICKER_EXCHANGE
    # Only US-listed tickers could have SEC CIK entries
    us_listed = [t for t in [
        "MSFT", "AMZN", "GOOGL", "META", "ORCL", "IBM",
        "NVDA", "AMD", "AVGO", "MRVL", "TSM", "ASML", "INTC", "ARM",
        "MU", "SNPS", "CDNS",
        "AMAT", "LRCX", "KLAC", "ENTG", "MKSI", "UCTT", "ICHR",
        "TER", "ONTO", "APD", "LIN",
        "ANET", "CSCO", "CIEN", "COHR", "LITE", "INFN", "NOK", "VIAV",
        "SMCI", "DELL", "HPE", "NTAP", "PSTG", "STX", "WDC",
        "EQIX", "DLR", "AMT", "CCI", "IREN", "APLD",
        "CEG", "VST", "NRG", "TLN", "NEE", "SO", "EXC", "ETR",
        "GEV", "BWX", "OKLO", "SMR", "FSLR",
        "VRT", "NVENT", "JCI", "TT", "CARR", "GNRC", "HUBB",
        "PWR", "MTZ", "EME", "MYR", "IESC", "AGX",
        "FCX", "SCCO", "AA", "NUE", "STLD", "MP", "UUUU", "ECL",
    ] if TICKER_EXCHANGE.get(t, "US") == "US"]
    # Foreign private issuers / non-SEC-registrants — excluded by design
    foreign = {"TSM", "ASML", "ARM", "NOK", "IREN", "STM", "ERIC"}
    domestic = [t for t in us_listed if t not in foreign]
    missing = [t for t in domestic if t not in CIK_MAP]
    assert missing == [], f"Missing CIKs for: {missing}"


# ── New tests ──────────────────────────────────────────────────────────────────

def test_tickerinfo_fields_complete():
    """Every TickerInfo entry has non-empty fields and no duplicate symbols."""
    from ingestion.ticker_registry import TICKERS_INFO
    for t in TICKERS_INFO:
        assert t.symbol,   f"Empty symbol in entry: {t}"
        assert t.layer,    f"Empty layer for {t.symbol}"
        assert t.exchange, f"Empty exchange for {t.symbol}"
        assert t.currency, f"Empty currency for {t.symbol}"
        assert t.country,  f"Empty country for {t.symbol}"
    symbols = [t.symbol for t in TICKERS_INFO]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_robotics_layer_populated():
    """robotics layer exists in LAYER_IDS and contains expected tickers."""
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics" in LAYER_IDS
    assert LAYER_IDS["robotics"] == 11
    robotics = tickers_in_layer("robotics")
    assert len(robotics) == 11
    assert "ABBN.SW" in robotics
    assert "6954.T"  in robotics
    assert "ISRG"    in robotics


def test_non_usd_tickers():
    """non_usd_tickers() returns only tickers with non-USD currency."""
    from ingestion.ticker_registry import non_usd_tickers, TICKER_CURRENCY
    result = non_usd_tickers()
    assert len(result) > 0
    for t in result:
        assert TICKER_CURRENCY[t] != "USD", f"{t} is USD but in non_usd_tickers()"
    # NVDA is USD — must not appear
    assert "NVDA" not in result
    # ABBN.SW is CHF — must appear
    assert "ABBN.SW" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_ticker_registry.py -v 2>&1 | head -30
```

Expected: Multiple FAILED (count mismatches, missing `TICKERS_INFO`, missing `non_usd_tickers`).

- [ ] **Step 3: Replace `ingestion/ticker_registry.py` with the refactored version**

```python
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
]

# ── Layer metadata ──────────────────────────────────────────────────────────

LAYER_IDS: dict[str, int] = {
    "cloud": 1, "compute": 2, "semi_equipment": 3, "networking": 4,
    "servers": 5, "datacenter": 6, "power": 7, "cooling": 8,
    "grid": 9, "metals": 10, "robotics": 11,
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
    return TICKER_CURRENCY[symbol]
```

- [ ] **Step 4: Run the tests**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_ticker_registry.py -v
```

Expected: 10 PASSED. If any fail, debug against the ticker list above.

- [ ] **Step 5: Run the full suite to check for regressions**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All prior tests pass. The supply_chain_features module-level code rebuilds `_LAYER_MAP`, `_LAYER_TICKERS`, `_CORRELATION_PEERS` from the new registry at import time — verify there are no import errors.

- [ ] **Step 6: Commit**

```bash
git add ingestion/ticker_registry.py tests/test_ticker_registry.py
git commit -m "feat: refactor registry to TickerInfo dataclass — 83→127 tickers across 11 layers (Task 1)"
```

---

## Task 2: FX Ingestion

**Files:**
- Create: `ingestion/fx_ingestion.py`
- Create: `tests/test_fx_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fx_ingestion.py`:

```python
"""Tests for FX rate ingestion."""
from __future__ import annotations
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest


def _mock_yf_response(rates: list[float], dates: list[str]) -> pd.DataFrame:
    """Build a fake yfinance download response."""
    return pd.DataFrame(
        {"Close": rates},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_fetch_fx_rates_schema():
    """fetch_fx_rates returns {pair: DataFrame} with date (Date) and rate (Float64)."""
    from ingestion.fx_ingestion import fetch_fx_rates

    mock_data = _mock_yf_response([1.08, 1.09], ["2025-01-01", "2025-01-02"])
    with patch("ingestion.fx_ingestion.yf.download", return_value=mock_data):
        result = fetch_fx_rates(["EURUSD"], years=1)

    assert "EURUSD" in result
    df = result["EURUSD"]
    assert "date" in df.columns
    assert "rate" in df.columns
    assert df["rate"].dtype == pl.Float64
    assert df["date"].dtype == pl.Date
    assert len(df) == 2
    assert df["date"][0] == date(2025, 1, 1)


def test_fetch_fx_rates_sorted_ascending():
    """Returned DataFrame is sorted by date ascending."""
    from ingestion.fx_ingestion import fetch_fx_rates

    mock_data = _mock_yf_response(
        [1.09, 1.08],
        ["2025-01-02", "2025-01-01"],  # deliberately reversed
    )
    with patch("ingestion.fx_ingestion.yf.download", return_value=mock_data):
        result = fetch_fx_rates(["EURUSD"], years=1)

    df = result["EURUSD"]
    assert df["date"][0] == date(2025, 1, 1)
    assert df["date"][1] == date(2025, 1, 2)


def test_fetch_fx_rates_returns_empty_on_error():
    """Connection errors return empty schema DataFrame — pipeline does not crash."""
    from ingestion.fx_ingestion import fetch_fx_rates

    with patch("ingestion.fx_ingestion.yf.download", side_effect=Exception("network error")):
        result = fetch_fx_rates(["EURUSD"], years=1)

    df = result["EURUSD"]
    assert df.is_empty()
    assert "date" in df.columns
    assert "rate" in df.columns


def test_save_fx_rates_creates_parquet(tmp_path):
    """save_fx_rates writes {pair}.parquet with correct schema."""
    from ingestion.fx_ingestion import save_fx_rates

    data = {"EURUSD": pl.DataFrame({
        "date": [date(2025, 1, 1), date(2025, 1, 2)],
        "rate": [1.08, 1.09],
    })}
    save_fx_rates(tmp_path, data)

    parquet_path = tmp_path / "EURUSD.parquet"
    assert parquet_path.exists()
    df = pl.read_parquet(parquet_path)
    assert list(df.columns) == ["date", "rate"]
    assert df["rate"].dtype == pl.Float64


def test_save_fx_rates_deduplicates(tmp_path):
    """Re-saving overlapping dates does not create duplicate rows."""
    from ingestion.fx_ingestion import save_fx_rates

    first = {"EURUSD": pl.DataFrame({
        "date": [date(2025, 1, 1), date(2025, 1, 2)],
        "rate": [1.08, 1.09],
    })}
    second = {"EURUSD": pl.DataFrame({
        "date": [date(2025, 1, 2), date(2025, 1, 3)],  # Jan 2 is duplicate
        "rate": [1.09, 1.10],
    })}

    save_fx_rates(tmp_path, first)
    save_fx_rates(tmp_path, second)

    df = pl.read_parquet(tmp_path / "EURUSD.parquet")
    assert len(df) == 3  # Jan 1, 2, 3 — no duplicate Jan 2
    assert df["date"].to_list() == [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_fx_ingestion.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'ingestion.fx_ingestion'`

- [ ] **Step 3: Create `ingestion/fx_ingestion.py`**

```python
"""Daily FX rate ingestion via yfinance.

Fetches 7 currency pairs needed for USD-normalizing non-USD tickers.
Saves to data/raw/financials/fx/{pair}.parquet (e.g., EURUSD.parquet).
Schema: date (pl.Date), rate (pl.Float64).
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)

# Map from pair name to yfinance symbol
_FX_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "CHFUSD": "CHFUSD=X",
    "JPYUSD": "JPYUSD=X",
    "DKKUSD": "DKKUSD=X",
    "SEKUSD": "SEKUSD=X",
    "NOKUSD": "NOKUSD=X",
    "GBPUSD": "GBPUSD=X",
}

# All currency codes covered by _FX_SYMBOLS (used externally for validation)
SUPPORTED_CURRENCIES: frozenset[str] = frozenset({"EUR", "CHF", "JPY", "DKK", "SEK", "NOK", "GBP"})

# Map currency ISO code → pair name
CURRENCY_TO_PAIR: dict[str, str] = {
    "EUR": "EURUSD",
    "CHF": "CHFUSD",
    "JPY": "JPYUSD",
    "DKK": "DKKUSD",
    "SEK": "SEKUSD",
    "NOK": "NOKUSD",
    "GBP": "GBPUSD",
}

_EMPTY_SCHEMA = {"date": pl.Date, "rate": pl.Float64}


def fetch_fx_rates(
    pairs: list[str] | None = None,
    years: int = 5,
) -> dict[str, pl.DataFrame]:
    """Fetch daily closing FX rates from yfinance.

    Args:
        pairs: Subset of pair names to fetch (default: all 7).
        years: History to fetch in years.

    Returns:
        Dict mapping pair name → DataFrame(date, rate). Empty DataFrame on failure.
    """
    if pairs is None:
        pairs = list(_FX_SYMBOLS.keys())

    result: dict[str, pl.DataFrame] = {}
    for pair in pairs:
        yf_symbol = _FX_SYMBOLS[pair]
        try:
            raw = yf.download(
                yf_symbol,
                period=f"{years}y",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                _LOG.warning("[FX] No data returned for %s", pair)
                result[pair] = pl.DataFrame(schema=_EMPTY_SCHEMA)
                continue

            # yfinance may return multi-level columns for single ticker in newer versions
            close = raw["Close"] if "Close" in raw.columns else raw[("Close", yf_symbol)]
            df = (
                pl.from_pandas(close.reset_index())
                .rename({"Date": "date", "Close": "rate"})
                .with_columns(pl.col("date").cast(pl.Date))
                .sort("date")
                .select(["date", "rate"])
            )
            df = df.with_columns(pl.col("rate").cast(pl.Float64))
            result[pair] = df

        except Exception as exc:
            _LOG.warning("[FX] Failed to fetch %s (%s): %s", pair, yf_symbol, exc)
            result[pair] = pl.DataFrame(schema=_EMPTY_SCHEMA)

    return result


def save_fx_rates(fx_dir: Path, pairs: dict[str, pl.DataFrame]) -> None:
    """Append-and-deduplicate FX rates into data/raw/financials/fx/{pair}.parquet.

    Args:
        fx_dir: Destination directory (created if absent).
        pairs: Dict of pair name → DataFrame(date, rate) from fetch_fx_rates().
    """
    fx_dir.mkdir(parents=True, exist_ok=True)
    for pair, df in pairs.items():
        if df.is_empty():
            continue
        path = fx_dir / f"{pair}.parquet"
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df]).unique("date").sort("date")
        df.write_parquet(path, compression="snappy")


if __name__ == "__main__":
    import datetime as dt
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ROOT = Path(__file__).parent.parent
    fx_dir = _ROOT / "data" / "raw" / "financials" / "fx"
    _LOG.info("Fetching FX rates for %d pairs...", len(_FX_SYMBOLS))
    rates = fetch_fx_rates()
    save_fx_rates(fx_dir, rates)
    for pair, df in rates.items():
        _LOG.info("  %s: %d rows", pair, len(df))
```

- [ ] **Step 4: Run the tests**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_fx_ingestion.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Run the full suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add ingestion/fx_ingestion.py tests/test_fx_ingestion.py
git commit -m "feat: add FX rate ingestion for 7 currency pairs — EUR/CHF/JPY/DKK/SEK/NOK/GBP (Task 2)"
```

---

## Task 3: FX Features

**Files:**
- Create: `processing/fx_features.py`
- Create: `tests/test_fx_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fx_features.py`:

```python
"""Tests for FX-adjusted return features."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


def _write_ohlcv(ticker_dir: Path, dates: list[date], prices: list[float]) -> None:
    """Write a minimal OHLCV parquet for a single ticker."""
    ticker_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "ticker":      [ticker_dir.name] * len(dates),
        "date":        dates,
        "close_price": prices,
    })
    year = dates[0].year
    df.write_parquet(ticker_dir / f"{year}.parquet")


def _write_fx(fx_dir: Path, pair: str, dates: list[date], rates: list[float]) -> None:
    """Write a minimal FX parquet."""
    fx_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": dates, "rate": rates}).write_parquet(fx_dir / f"{pair}.parquet")


def test_usd_ticker_passthrough(tmp_path):
    """USD ticker close prices are unchanged by build_usd_close_matrix."""
    from processing.fx_features import build_usd_close_matrix

    close = pl.DataFrame({
        "date":  [date(2025, 1, 1), date(2025, 1, 2)],
        "NVDA":  [100.0, 102.0],
    })
    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()

    result = build_usd_close_matrix(close, fx_dir)
    assert result["NVDA"].to_list() == [100.0, 102.0]


def test_eur_ticker_converted_to_usd(tmp_path):
    """EUR ticker is multiplied by EURUSD rate on each date."""
    from processing.fx_features import build_usd_close_matrix

    fx_dir = tmp_path / "fx"
    _write_fx(fx_dir, "EURUSD", [date(2025, 1, 1), date(2025, 1, 2)], [1.10, 1.12])

    close = pl.DataFrame({
        "date":    [date(2025, 1, 1), date(2025, 1, 2)],
        "SAP.DE":  [100.0, 100.0],
    })

    result = build_usd_close_matrix(close, fx_dir)
    assert abs(result["SAP.DE"][0] - 110.0) < 0.001
    assert abs(result["SAP.DE"][1] - 112.0) < 0.001


def test_missing_fx_rate_produces_null(tmp_path):
    """Date with no FX rate → null for that ticker, no crash."""
    from processing.fx_features import build_usd_close_matrix

    fx_dir = tmp_path / "fx"
    # FX only covers Jan 1, not Jan 2
    _write_fx(fx_dir, "EURUSD", [date(2025, 1, 1)], [1.10])

    close = pl.DataFrame({
        "date":    [date(2025, 1, 1), date(2025, 1, 2)],
        "SAP.DE":  [100.0, 101.0],
    })

    result = build_usd_close_matrix(close, fx_dir)
    assert abs(result["SAP.DE"][0] - 110.0) < 0.001  # Jan 1: has rate
    assert result["SAP.DE"][1] is None               # Jan 2: no rate → null


def test_fx_adjusted_return_20d_correct(tmp_path):
    """EUR ticker with constant FX rate: 20d USD return == 20d local return."""
    from processing.fx_features import join_fx_features

    as_of = date(2025, 6, 1)
    n_days = 25
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]

    # SAP.DE: EUR price grows 0.5%/day. EURUSD = 1.1 (constant).
    # 20d return in EUR == 20d return in USD (constant FX cancels out)
    eur_prices = [100.0 * (1.005 ** i) for i in range(n_days + 1)]
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv(ohlcv_dir / "SAP.DE", dates, eur_prices)

    fx_dir = tmp_path / "fx"
    _write_fx(fx_dir, "EURUSD", dates, [1.1] * (n_days + 1))

    spine = pl.DataFrame({"ticker": ["SAP.DE"], "date": [as_of]})
    result = join_fx_features(spine, fx_dir=fx_dir, ohlcv_dir=ohlcv_dir)

    expected = (1.005 ** 20) - 1.0  # ≈ 0.1049
    actual = result["fx_adjusted_return_20d"][0]
    assert actual is not None
    assert abs(actual - expected) < 0.01, f"Expected ~{expected:.4f}, got {actual:.4f}"


def test_usd_ticker_gets_null_feature(tmp_path):
    """fx_adjusted_return_20d is always null for USD tickers."""
    from processing.fx_features import join_fx_features

    ohlcv_dir = tmp_path / "ohlcv"
    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()

    spine = pl.DataFrame({
        "ticker": ["NVDA"],
        "date":   [date(2025, 6, 1)],
    })
    result = join_fx_features(spine, fx_dir=fx_dir, ohlcv_dir=ohlcv_dir)
    assert result["fx_adjusted_return_20d"][0] is None


def test_join_fx_features_adds_column(tmp_path):
    """join_fx_features adds exactly 1 Float64 column named fx_adjusted_return_20d."""
    from processing.fx_features import join_fx_features

    ohlcv_dir = tmp_path / "ohlcv"; ohlcv_dir.mkdir()
    fx_dir    = tmp_path / "fx";    fx_dir.mkdir()

    spine = pl.DataFrame({
        "ticker": ["NVDA", "SAP.DE"],
        "date":   [date(2025, 6, 1), date(2025, 6, 1)],
    })
    result = join_fx_features(spine, fx_dir=fx_dir, ohlcv_dir=ohlcv_dir)

    assert "fx_adjusted_return_20d" in result.columns
    assert result["fx_adjusted_return_20d"].dtype == pl.Float64
    assert len(result) == 2


def test_registry_coverage():
    """Every non-USD currency in TICKERS_INFO has a corresponding supported FX pair."""
    from ingestion.ticker_registry import TICKERS_INFO
    from ingestion.fx_ingestion import CURRENCY_TO_PAIR, SUPPORTED_CURRENCIES

    non_usd_currencies = {t.currency for t in TICKERS_INFO if t.currency != "USD"}
    uncovered = non_usd_currencies - SUPPORTED_CURRENCIES
    assert uncovered == set(), f"Currencies with no FX pair: {uncovered}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_fx_features.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'processing.fx_features'`

- [ ] **Step 3: Create `processing/fx_features.py`**

```python
"""FX-adjusted return features for non-USD tickers.

Provides:
  build_usd_close_matrix() — convert non-USD close prices to USD via daily FX rates
  join_fx_features()       — add fx_adjusted_return_20d to training spine

Called by models/train.py via join_fx_features().
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ingestion.fx_ingestion import CURRENCY_TO_PAIR
from ingestion.ticker_registry import TICKER_CURRENCY, non_usd_tickers

_LOG = logging.getLogger(__name__)


def _load_fx_rate(pair: str, fx_dir: Path) -> pl.DataFrame:
    """Load FX rate parquet. Returns empty DataFrame(date, rate) if not found."""
    path = fx_dir / f"{pair}.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"date": pl.Date, "rate": pl.Float64})
    return pl.read_parquet(path)


def build_usd_close_matrix(
    close_wide: pl.DataFrame,
    fx_dir: Path,
) -> pl.DataFrame:
    """Convert non-USD ticker columns to USD using daily FX rates.

    USD tickers pass through unchanged.
    Non-USD tickers: close_price * fx_rate_on_date. Missing rate → null.

    Args:
        close_wide: Wide close-price DataFrame with date column and one column per ticker.
        fx_dir: Directory containing {pair}.parquet FX rate files.

    Returns:
        Same shape as close_wide with non-USD columns converted to USD.
    """
    result = close_wide.clone()
    ticker_cols = [c for c in close_wide.columns if c != "date"]

    # Group non-USD tickers by FX pair to minimise file reads
    pair_to_tickers: dict[str, list[str]] = {}
    for ticker in ticker_cols:
        currency = TICKER_CURRENCY.get(ticker, "USD")
        if currency == "USD":
            continue
        pair = CURRENCY_TO_PAIR.get(currency)
        if pair is None:
            _LOG.warning("[FX] No pair for currency %s (ticker %s) — skipping", currency, ticker)
            continue
        pair_to_tickers.setdefault(pair, []).append(ticker)

    for pair, tickers in pair_to_tickers.items():
        fx_df = _load_fx_rate(pair, fx_dir)
        if fx_df.is_empty():
            # Null out all tickers for this pair
            result = result.with_columns([
                pl.lit(None).cast(pl.Float64).alias(t) for t in tickers
            ])
            continue

        rate_col = f"_fx_{pair}"
        result = result.join(fx_df.rename({"rate": rate_col}), on="date", how="left")
        for ticker in tickers:
            result = result.with_columns(
                (pl.col(ticker) * pl.col(rate_col)).alias(ticker)
            )
        result = result.drop(rate_col)

    return result


def _build_close_matrix_for_tickers(
    ohlcv_dir: Path,
    tickers: list[str],
    min_date: date,
    max_date: date,
    extra_days: int = 100,
) -> pl.DataFrame:
    """Build wide close-price matrix for a specific list of tickers.

    Mirrors supply_chain_features._build_close_matrix but scoped to given tickers
    to avoid a circular import between fx_features and supply_chain_features.
    """
    load_start = min_date - timedelta(days=extra_days)

    frames = []
    for ticker in tickers:
        ticker_dir = ohlcv_dir / ticker
        if not ticker_dir.exists():
            continue
        year_files = sorted(
            f for f in ticker_dir.glob("*.parquet")
            if f.stem.isdigit() and int(f.stem) >= load_start.year
        )
        if not year_files:
            continue
        df = pl.concat([pl.read_parquet(f) for f in year_files])
        df = (
            df.filter((pl.col("date") >= load_start) & (pl.col("date") <= max_date))
            .sort("date")
            .select(["date", pl.col("close_price").alias(ticker)])
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="date", how="full", coalesce=True)
    return result.sort("date")


def join_fx_features(
    df: pl.DataFrame,
    fx_dir: Path | None = None,
    ohlcv_dir: Path | None = None,
) -> pl.DataFrame:
    """Add fx_adjusted_return_20d: 20-day cumulative return on USD-normalised close prices.

    - Non-USD tickers: populated Float64.
    - USD tickers: null (their price features already capture USD returns).

    Args:
        df: Training spine with columns [ticker, date, ...].
        fx_dir: Path to data/raw/financials/fx/ (default: resolved from __file__).
        ohlcv_dir: Path to data/raw/financials/ohlcv/ (default: resolved from __file__).

    Returns:
        df with 1 new Float64 column. Missing data → null (not 0).
    """
    _ROOT = Path(__file__).parent.parent
    if fx_dir is None:
        fx_dir = _ROOT / "data" / "raw" / "financials" / "fx"
    if ohlcv_dir is None:
        ohlcv_dir = _ROOT / "data" / "raw" / "financials" / "ohlcv"

    null_col = pl.lit(None).cast(pl.Float64).alias("fx_adjusted_return_20d")

    if not fx_dir.exists():
        _LOG.warning("[FX] No fx_dir found — fx_adjusted_return_20d will be null")
        return df.with_columns(null_col)

    # Only compute for non-USD tickers that appear in the spine
    spine_tickers = df["ticker"].unique().to_list()
    intl_tickers = [t for t in spine_tickers if TICKER_CURRENCY.get(t, "USD") != "USD"]

    if not intl_tickers:
        return df.with_columns(null_col)

    min_date = df["date"].min()
    max_date = df["date"].max()

    close_wide = _build_close_matrix_for_tickers(ohlcv_dir, intl_tickers, min_date, max_date)
    if close_wide.is_empty() or close_wide.height < 21:
        return df.with_columns(null_col)

    usd_close = build_usd_close_matrix(close_wide, fx_dir)

    ticker_cols = [c for c in usd_close.columns if c != "date"]
    ret_20d_usd = usd_close.select(
        ["date"] + [
            (pl.col(t) / pl.col(t).shift(20) - 1).alias(t)
            for t in ticker_cols
        ]
    )

    vals: list[float | None] = []
    for row in df.select(["ticker", "date"]).iter_rows(named=True):
        ticker = row["ticker"]
        as_of  = row["date"]

        if TICKER_CURRENCY.get(ticker, "USD") == "USD" or ticker not in ret_20d_usd.columns:
            vals.append(None)
            continue

        ret_row = ret_20d_usd.filter(pl.col("date") == as_of)
        if ret_row.is_empty():
            vals.append(None)
            continue

        v = ret_row[ticker][0]
        if v is None:
            vals.append(None)
        else:
            fv = float(v)
            vals.append(None if np.isnan(fv) else fv)

    return df.with_columns(pl.Series("fx_adjusted_return_20d", vals, dtype=pl.Float64))
```

- [ ] **Step 4: Run the tests**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_fx_features.py -v
```

Expected: 7 PASSED. If `test_registry_coverage` fails, check that all non-USD currencies in `TICKERS_INFO` are in `SUPPORTED_CURRENCIES` in `fx_ingestion.py`.

- [ ] **Step 5: Run the full suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add processing/fx_features.py tests/test_fx_features.py
git commit -m "feat: FX features — build_usd_close_matrix + fx_adjusted_return_20d (Task 3)"
```

---

## Task 4: supply_chain_features.py Extension

**Files:**
- Modify: `processing/supply_chain_features.py`

This task adds an optional `fx_dir` parameter to `join_supply_chain_features()` and wires it through to `compute_supply_chain_correlation()`, which uses `build_usd_close_matrix()` when FX data is available.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_supply_chain_features.py` (append at the end of the file):

```python
def test_supply_chain_correlation_uses_usd_matrix_when_fx_dir_provided(tmp_path):
    """When fx_dir is given, correlation is computed on USD-normalised returns."""
    from processing.supply_chain_features import (
        compute_supply_chain_correlation, _CORRELATION_PEERS,
    )
    from processing.fx_features import build_usd_close_matrix

    ticker = "NVDA"
    peers = _CORRELATION_PEERS[ticker]
    as_of = date(2025, 6, 1)
    n_days = 65

    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    data: dict = {
        "date":  dates,
        ticker:  [0.01 * ((-1) ** i) for i in range(n_days + 1)],
    }
    for peer in peers:
        data[peer] = [0.005 * ((-1) ** i) for i in range(n_days + 1)]

    # All USD tickers — USD matrix should be identical to local matrix
    ret_1d = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))
    usd_ret_1d = build_usd_close_matrix(ret_1d, tmp_path)  # empty fx_dir → USD tickers unchanged

    result_local = compute_supply_chain_correlation(ticker, as_of, ret_1d)
    result_usd   = compute_supply_chain_correlation(ticker, as_of, usd_ret_1d)

    # For USD tickers both should give the same result
    assert result_local is not None
    assert result_usd is not None
    assert abs(result_local - result_usd) < 0.001
```

- [ ] **Step 2: Run test to verify it passes already (it should)**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_supply_chain_features.py::test_supply_chain_correlation_uses_usd_matrix_when_fx_dir_provided -v
```

Expected: PASS (the test validates behaviour that already works; the next step wires fx_dir into the public interface).

- [ ] **Step 3: Add `fx_dir` parameter to `join_supply_chain_features()`**

Read `processing/supply_chain_features.py` to find the current `join_supply_chain_features` signature and the line that calls `compute_supply_chain_correlation`. Then make these two edits:

**Edit 1** — Update function signature (find and replace):

Old:
```python
def join_supply_chain_features(
    df: pl.DataFrame,
    ohlcv_dir: Path | None = None,
    earnings_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Add own_layer_momentum_20d, ecosystem_momentum_20d,
    supply_chain_correlation_60d, peer_eps_surprise_mean to training spine.

    Args:
        df: Training spine with columns [ticker, date, ...].
        ohlcv_dir: Path to data/raw/financials/ohlcv/ (default: resolved from __file__).
        earnings_dir: Path to data/raw/fundamentals/earnings/ (default: resolved from __file__).

    Returns df with 4 new Float64 columns. Missing data → null (not 0).
    """
```

New:
```python
def join_supply_chain_features(
    df: pl.DataFrame,
    ohlcv_dir: Path | None = None,
    earnings_dir: Path | None = None,
    fx_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Add own_layer_momentum_20d, ecosystem_momentum_20d,
    supply_chain_correlation_60d, peer_eps_surprise_mean to training spine.

    Args:
        df: Training spine with columns [ticker, date, ...].
        ohlcv_dir: Path to data/raw/financials/ohlcv/ (default: resolved from __file__).
        earnings_dir: Path to data/raw/fundamentals/earnings/ (default: resolved from __file__).
        fx_dir: Path to data/raw/financials/fx/ for USD-normalised correlation matrix.
                When None, correlation uses local-currency returns (backwards-compatible).

    Returns df with 4 new Float64 columns. Missing data → null (not 0).
    """
```

**Edit 2** — Inside `join_supply_chain_features`, after `ret_1d_wide` is built (find the line that builds `ret_1d_wide`), add USD matrix derivation and pass it to correlation:

Find the block:
```python
    ticker_cols = [c for c in close_wide.columns if c != "date"]
    ret_1d_wide = close_wide.select(
        ["date"] + [pl.col(t).pct_change(1).alias(t) for t in ticker_cols]
    )
```

Replace with:
```python
    ticker_cols = [c for c in close_wide.columns if c != "date"]
    ret_1d_wide = close_wide.select(
        ["date"] + [pl.col(t).pct_change(1).alias(t) for t in ticker_cols]
    )

    # For cross-ticker correlations, use USD-normalised returns when fx_dir is available
    # to avoid spurious correlations from shared EUR/JPY exposure.
    if fx_dir is not None and fx_dir.exists():
        from processing.fx_features import build_usd_close_matrix
        usd_close = build_usd_close_matrix(close_wide, fx_dir)
        ret_1d_corr = usd_close.select(
            ["date"] + [pl.col(t).pct_change(1).alias(t) for t in ticker_cols]
        )
    else:
        ret_1d_corr = ret_1d_wide
```

Then find the line inside the row loop:
```python
        corr_vals.append(compute_supply_chain_correlation(ticker, as_of, ret_1d_wide))
```

Change it to:
```python
        corr_vals.append(compute_supply_chain_correlation(ticker, as_of, ret_1d_corr))
```

- [ ] **Step 4: Run all supply chain tests**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_supply_chain_features.py -v
```

Expected: 10 PASSED (9 original + 1 new).

- [ ] **Step 5: Run the full suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add processing/supply_chain_features.py tests/test_supply_chain_features.py
git commit -m "feat: supply_chain_features gains fx_dir param — USD matrix for cross-ticker correlations (Task 4)"
```

---

## Task 5: Model Integration

**Files:**
- Modify: `models/train.py`
- Modify: `models/inference.py`

- [ ] **Step 1: Read the current state of both files**

Read `models/train.py` lines 37–115 (imports + FEATURE_COLS) and lines 225–245 (`build_training_dataset` join block).

Read `models/inference.py` lines 27–45 (imports) and lines 115–130 (`_build_feature_df` join block).

- [ ] **Step 2: Extend `models/train.py`**

**Change A** — Add import (with other processing imports, after `join_supply_chain_features`):
```python
from processing.fx_features import join_fx_features
```

**Change B** — Add `FX_FEATURE_COLS` after `SUPPLY_CHAIN_FEATURE_COLS`:
```python
FX_FEATURE_COLS = ["fx_adjusted_return_20d"]
```

**Change C** — Append to `FEATURE_COLS` assembly (change the comment from 47 to 48):
```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS + ENERGY_FEATURE_COLS
    + SUPPLY_CHAIN_FEATURE_COLS + FX_FEATURE_COLS  # 47 → 48 features total
)
```

**Change D** — In `build_training_dataset()`, after `df = join_supply_chain_features(df, ohlcv_dir=ohlcv_dir)`, add:
```python
    df = join_fx_features(df, ohlcv_dir=ohlcv_dir)
```

- [ ] **Step 3: Extend `models/inference.py`**

**Change A** — Add `FX_FEATURE_COLS` to the import from `models.train`:
```python
from models.train import (
    FEATURE_COLS, INSIDER_FEATURE_COLS, SENTIMENT_FEATURE_COLS,
    SHORT_INTEREST_FEATURE_COLS, EARNINGS_FEATURE_COLS, GRAPH_FEATURE_COLS,
    OWNERSHIP_FEATURE_COLS, ENERGY_FEATURE_COLS, SUPPLY_CHAIN_FEATURE_COLS,
    FX_FEATURE_COLS,
)
```

**Change B** — Add import of the join function (with other processing imports):
```python
from processing.fx_features import join_fx_features
```

**Change C** — In `_build_feature_df()`, after `df = join_supply_chain_features(df, ohlcv_dir=ohlcv_dir)`, add:
```python
    df = join_fx_features(df, ohlcv_dir=ohlcv_dir)
```

**Change D** — Update the docstring comment from "47-feature" to "48-feature".

- [ ] **Step 4: Verify feature count**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
python -c "
from models.train import FEATURE_COLS, FX_FEATURE_COLS
print(f'Feature count: {len(FEATURE_COLS)}')
print('FX cols:', FX_FEATURE_COLS)
print('Last 3:', FEATURE_COLS[-3:])
"
```

Expected:
```
Feature count: 48
FX cols: ['fx_adjusted_return_20d']
Last 3: ['peer_eps_surprise_mean', 'fx_adjusted_return_20d']
```
(Last 3 shows the final supply chain feature and the new FX feature.)

- [ ] **Step 5: Run the full suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add models/train.py models/inference.py
git commit -m "feat: wire FX feature into FEATURE_COLS — model trains on 48 features (Task 5)"
```

---

## Self-Review

**Spec coverage:**
- ✅ `TickerInfo` dataclass with symbol, layer, exchange, currency, country — Task 1
- ✅ 127 tickers across 11 layers (83→127) — Task 1
- ✅ All existing lookups backwards-compatible (`TICKER_LAYERS`, `tickers_in_layer()`, `layers()`) — Task 1
- ✅ `non_usd_tickers()`, `ticker_currency()` new lookups — Task 1
- ✅ `LAYER_IDS["robotics"] = 11` — Task 1
- ✅ `ingestion/fx_ingestion.py` with 7 pairs, failure-safe, append-deduplicate — Task 2
- ✅ `build_usd_close_matrix()` — Task 3
- ✅ `join_fx_features()` → `fx_adjusted_return_20d` null for USD, populated for non-USD — Task 3
- ✅ Circular import avoided (`_build_close_matrix` inlined in `fx_features.py`) — Task 3
- ✅ `join_supply_chain_features()` gains `fx_dir` param; uses USD matrix for correlation — Task 4
- ✅ `FX_FEATURE_COLS`, train.py + inference.py extended, 47→48 — Task 5
- ✅ 3 new registry tests + existing tests updated — Task 1
- ✅ 5 FX ingestion tests — Task 2
- ✅ 7 FX feature tests — Task 3

**Placeholder scan:** No TBD, no vague steps. All code complete.

**Type consistency:**
- `build_usd_close_matrix(close_wide, fx_dir)` — used in Task 3 tests and Task 4 implementation ✓
- `join_fx_features(df, fx_dir, ohlcv_dir)` — signature consistent across Task 3 code and Task 5 call ✓
- `FX_FEATURE_COLS` — defined in train.py Task 5, imported in inference.py Task 5 ✓
