# AI Cybersecurity Pillar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Layer 12 (`cyber_pureplay`, 5 tickers) and Layer 13 (`cyber_platform`, 9 tickers) to the prediction universe, plus 7 cyber threat regime features ingested from free public APIs (NVD CVE, CISA KEV, AlienVault OTX).

**Architecture:** Ticker registry gets two new layers following the existing TickerInfo pattern; a new pluggable ingestion module fetches raw threat events `(date, source, metric, value)` from three free APIs and writes Hive-partitioned parquet; a new processing module aggregates those events into 7- and 30-day rolling window features joined by date to all tickers; `FEATURE_COLS` grows from 48 → 55 with cyber threat features routed to short + medium tiers only (excluded from long — noise at year+ horizons).

**Tech Stack:** Python 3.11+, Polars, requests, Parquet/snappy, pytest, `unittest.mock.patch` for HTTP mocking.

---

## File Map

| File | Change |
|---|---|
| `ingestion/ticker_registry.py` | Add `cyber_pureplay` (Layer 12) + `cyber_platform` (Layer 13) tickers; update `LAYER_IDS`, `LAYER_LABELS` |
| `ingestion/cyber_threat_ingestion.py` | **New** — `CyberThreatSource` protocol; `NVDSource`, `CISASource`, `OTXSource`; `ingest_cyber_threats()` |
| `processing/cyber_threat_features.py` | **New** — `CYBER_THREAT_FEATURE_COLS`; `build_cyber_threat_features()`; `join_cyber_threat_features()` |
| `models/train.py` | Import `CYBER_THREAT_FEATURE_COLS` + `join_cyber_threat_features`; extend `FEATURE_COLS` + `TIER_FEATURE_COLS`; add join in `build_training_dataset` |
| `models/inference.py` | Import `join_cyber_threat_features`; add call in `_build_feature_df` |
| `tests/test_ticker_registry.py` | Update counts (127→141, 11→13 layers, 36→37 non-USD); add cyber layer tests |
| `tests/test_cyber_threat_ingestion.py` | **New** — source schema, OTX degradation, ingest writes parquet |
| `tests/test_cyber_threat_features.py` | **New** — rolling sums, index in [0,1], missing dir zero-fills, join adds columns |

---

### Task 1: Ticker Registry — Layers 12 + 13

**Files:**
- Modify: `ingestion/ticker_registry.py`
- Modify: `tests/test_ticker_registry.py`

- [ ] **Step 1: Write failing tests**

Replace the existing test file `tests/test_ticker_registry.py` with:

```python
# tests/test_ticker_registry.py


def test_ticker_count():
    from ingestion.ticker_registry import TICKERS, TICKER_LAYERS
    assert len(TICKERS) == 141       # 127 + 5 cyber_pureplay + 9 cyber_platform
    assert len(TICKER_LAYERS) == 141


def test_all_layers_present():
    from ingestion.ticker_registry import TICKER_LAYERS, LAYER_IDS
    layers_used = set(TICKER_LAYERS.values())
    assert layers_used == set(LAYER_IDS.keys())


def test_tickers_in_layer():
    from ingestion.ticker_registry import tickers_in_layer
    cloud = tickers_in_layer("cloud")
    assert "MSFT" in cloud and "AMZN" in cloud
    assert len(cloud) == 9


def test_hyperscalers_are_cloud():
    from ingestion.ticker_registry import HYPERSCALERS, TICKER_LAYERS
    for t in HYPERSCALERS:
        assert TICKER_LAYERS[t] == "cloud"


def test_layers_returns_13():
    from ingestion.ticker_registry import layers
    assert len(layers()) == 13


def test_layers_order():
    from ingestion.ticker_registry import layers
    result = layers()
    assert result[0] == "cloud"
    assert result[-3] == "robotics"    # robotics=11; cyber_pureplay=12, cyber_platform=13
    assert result[-2] == "cyber_pureplay"
    assert result[-1] == "cyber_platform"


def test_cyber_pureplay_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "cyber_pureplay" in LAYER_IDS
    assert LAYER_IDS["cyber_pureplay"] == 12
    tickers = tickers_in_layer("cyber_pureplay")
    assert len(tickers) == 5
    assert "CRWD" in tickers
    assert "ZS" in tickers
    assert "S" in tickers
    assert "DARK.L" in tickers
    assert "VRNS" in tickers


def test_cyber_platform_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "cyber_platform" in LAYER_IDS
    assert LAYER_IDS["cyber_platform"] == 13
    tickers = tickers_in_layer("cyber_platform")
    assert len(tickers) == 9
    for expected in ["PANW", "FTNT", "CHKP", "CYBR", "TENB", "QLYS", "OKTA", "AKAM", "RPD"]:
        assert expected in tickers, f"{expected} missing from cyber_platform"


def test_cik_map_covers_domestic_tickers():
    """CIK_MAP must have entries for original 83 domestic US tickers."""
    from ingestion.edgar_fundamentals_ingestion import CIK_MAP
    from ingestion.ticker_registry import TICKER_EXCHANGE
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
    foreign = {"TSM", "ASML", "ARM", "NOK", "IREN", "STM", "ERIC"}
    domestic = [t for t in us_listed if t not in foreign]
    missing = [t for t in domestic if t not in CIK_MAP]
    assert missing == [], f"Missing CIKs for: {missing}"


def test_tickerinfo_fields_complete():
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
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics" in LAYER_IDS
    assert LAYER_IDS["robotics"] == 11
    robotics = tickers_in_layer("robotics")
    assert len(robotics) == 11
    assert "ABBN.SW" in robotics
    assert "6954.T"  in robotics
    assert "ISRG"    in robotics


def test_non_usd_tickers():
    from ingestion.ticker_registry import non_usd_tickers, TICKER_CURRENCY
    result = non_usd_tickers()
    assert len(result) == 37    # was 36; +DARK.L (GBP)
    for t in result:
        assert TICKER_CURRENCY[t] != "USD", f"{t} is USD but in non_usd_tickers()"
    assert "NVDA" not in result
    assert "ABBN.SW" in result
    assert "DARK.L" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
python -m pytest tests/test_ticker_registry.py -v 2>&1 | head -40
```

Expected: `test_ticker_count`, `test_layers_returns_13`, `test_layers_order`, `test_cyber_pureplay_layer_populated`, `test_cyber_platform_layer_populated`, `test_non_usd_tickers` all FAIL.

- [ ] **Step 3: Add Layers 12 + 13 to ticker_registry.py**

In `ingestion/ticker_registry.py`, add the following after the last `# ── Layer 11: Robotics` block (after the closing `TickerInfo("6861.T", "robotics", ...)` entry):

```python
    # ── Layer 12: AI Cybersecurity — Pure Plays (5) ───────────────────────────
    TickerInfo("CRWD",   "cyber_pureplay", "US", "USD", "US"),
    TickerInfo("ZS",     "cyber_pureplay", "US", "USD", "US"),
    TickerInfo("S",      "cyber_pureplay", "US", "USD", "US"),
    TickerInfo("DARK.L", "cyber_pureplay", "L",  "GBP", "GB"),
    TickerInfo("VRNS",   "cyber_pureplay", "US", "USD", "US"),
    # ── Layer 13: AI Cybersecurity — Platform Vendors (9) ─────────────────────
    TickerInfo("PANW",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("FTNT",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("CHKP",   "cyber_platform", "US", "USD", "IL"),
    TickerInfo("CYBR",   "cyber_platform", "US", "USD", "IL"),
    TickerInfo("TENB",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("QLYS",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("OKTA",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("AKAM",   "cyber_platform", "US", "USD", "US"),
    TickerInfo("RPD",    "cyber_platform", "US", "USD", "US"),
```

Then in the `LAYER_IDS` dict, add:
```python
    "cyber_pureplay": 12,
    "cyber_platform": 13,
```

And in `LAYER_LABELS`, add:
```python
    "cyber_pureplay": "AI Cybersecurity — Pure Plays",
    "cyber_platform": "AI Cybersecurity — Platform Vendors",
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ticker_registry.py -v 2>&1 | tail -25
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Run full suite to catch regressions**

```bash
python -m pytest tests/ -m 'not integration' -q 2>&1 | tail -5
```

Expected: all passing (same count as before + new ticker tests).

- [ ] **Step 6: Commit**

```bash
git add ingestion/ticker_registry.py tests/test_ticker_registry.py
git commit -m "feat: add cyber_pureplay (Layer 12) and cyber_platform (Layer 13) to ticker registry"
```

---

### Task 2: Cyber Threat Ingestion

**Files:**
- Create: `ingestion/cyber_threat_ingestion.py`
- Create: `tests/test_cyber_threat_ingestion.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cyber_threat_ingestion.py`:

```python
"""Tests for cyber_threat_ingestion.py — all HTTP calls are mocked."""
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


# ── Schema helper ─────────────────────────────────────────────────────────────

_EXPECTED_SCHEMA = {"date": pl.Date, "source": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


def _check_schema(df: pl.DataFrame) -> None:
    assert set(df.columns) == set(_EXPECTED_SCHEMA), f"Unexpected columns: {df.columns}"
    for col, dtype in _EXPECTED_SCHEMA.items():
        assert df[col].dtype == dtype, f"Column {col}: expected {dtype}, got {df[col].dtype}"


# ── NVDSource ─────────────────────────────────────────────────────────────────

def _nvd_response(score: float, published: str = "2024-01-15T12:00:00.000") -> dict:
    """Build a minimal NVD API v2 response with a single CVE."""
    return {
        "totalResults": 1,
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-0001",
                "published": published,
                "metrics": {
                    "cvssMetricV31": [{
                        "cvssData": {"baseScore": score}
                    }]
                }
            }
        }]
    }


def test_nvd_source_critical_cve(tmp_path):
    """NVDSource returns metric='cve_critical' for CVSS >= 9.0."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nvd_response(score=9.8)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "cve_critical"
    assert df["value"][0] == 1.0
    assert df["source"][0] == "nvd"


def test_nvd_source_high_cve(tmp_path):
    """NVDSource returns metric='cve_high' for 7.0 <= CVSS < 9.0."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nvd_response(score=7.5)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "cve_high"


def test_nvd_source_below_threshold_excluded(tmp_path):
    """NVDSource drops CVEs with CVSS < 7.0."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nvd_response(score=5.5)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert len(df) == 0


def test_nvd_source_network_error_returns_empty():
    """NVDSource returns empty DataFrame on network error (no crash)."""
    with patch("requests.get", side_effect=Exception("timeout")), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert df.is_empty()


# ── CISASource ────────────────────────────────────────────────────────────────

def _cisa_response(date_added: str = "2024-01-15") -> dict:
    return {
        "vulnerabilities": [{
            "cveID": "CVE-2021-44228",
            "dateAdded": date_added,
            "vendorProject": "Apache",
        }]
    }


def test_cisa_source_returns_kev_rows():
    """CISASource returns metric='cisa_kev', one row per KEV entry in range."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _cisa_response("2024-01-15")
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.cyber_threat_ingestion import CISASource
        source = CISASource()
        df = source.fetch("2024-01-01", "2024-01-31")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "cisa_kev"
    assert df["source"][0] == "cisa"
    assert str(df["date"][0]) == "2024-01-15"


def test_cisa_source_filters_by_date_range():
    """CISASource drops KEVs outside the requested date range."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"vulnerabilities": [
        {"cveID": "CVE-A", "dateAdded": "2024-01-15"},
        {"cveID": "CVE-B", "dateAdded": "2024-02-01"},  # outside range
    ]}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.cyber_threat_ingestion import CISASource
        source = CISASource()
        df = source.fetch("2024-01-01", "2024-01-31")

    assert len(df) == 1
    assert str(df["date"][0]) == "2024-01-15"


# ── OTXSource ─────────────────────────────────────────────────────────────────

def test_otx_source_no_key_returns_empty():
    """OTXSource returns empty DataFrame when OTX_API_KEY is not set."""
    import os
    env_backup = os.environ.pop("OTX_API_KEY", None)
    try:
        from ingestion.cyber_threat_ingestion import OTXSource
        source = OTXSource()
        df = source.fetch("2024-01-01", "2024-01-31")
        _check_schema(df)
        assert df.is_empty()
    finally:
        if env_backup is not None:
            os.environ["OTX_API_KEY"] = env_backup


def test_otx_source_with_key_returns_pulse_rows(monkeypatch):
    """OTXSource returns metric='otx_pulse' rows when key is present."""
    monkeypatch.setenv("OTX_API_KEY", "test-key-abc")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [{"created": "2024-01-15T10:00:00.000000", "name": "Log4Shell"}],
        "next": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion import cyber_threat_ingestion
        import importlib; importlib.reload(cyber_threat_ingestion)
        source = cyber_threat_ingestion.OTXSource()
        df = source.fetch("2024-01-01", "2024-01-31")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "otx_pulse"
    assert df["source"][0] == "otx"


# ── ingest_cyber_threats ──────────────────────────────────────────────────────

def test_ingest_cyber_threats_writes_parquet(tmp_path):
    """ingest_cyber_threats writes threats.parquet partitioned by date."""
    import datetime
    import polars as pl

    class _FakeSource:
        def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
            return pl.DataFrame({
                "date": [datetime.date(2024, 1, 15)],
                "source": ["test"],
                "metric": ["cve_critical"],
                "value": [1.0],
            })

    from ingestion.cyber_threat_ingestion import ingest_cyber_threats
    ingest_cyber_threats(
        start_date="2024-01-15",
        end_date="2024-01-15",
        output_dir=tmp_path,
        sources=[_FakeSource()],
    )

    expected = tmp_path / "date=2024-01-15" / "threats.parquet"
    assert expected.exists(), f"Expected parquet at {expected}"
    df = pl.read_parquet(expected)
    assert "metric" in df.columns
    assert len(df) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cyber_threat_ingestion.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'ingestion.cyber_threat_ingestion'`

- [ ] **Step 3: Create ingestion/cyber_threat_ingestion.py**

```python
"""Cyber threat data ingestion from free public sources.

Fetches CVEs (NVD), known exploited vulnerabilities (CISA), and threat pulses
(AlienVault OTX) for a date range and writes Hive-partitioned parquet files:

    data/raw/cyber_threat/date=YYYY-MM-DD/threats.parquet

Schema: date (Date), source (Utf8), metric (Utf8), value (Float64)

Metrics produced:
    cve_critical  — CVSS >= 9.0 vulnerability published (NVD)
    cve_high      — 7.0 <= CVSS < 9.0 vulnerability published (NVD)
    cisa_kev      — CISA known exploited vulnerability added (CISA)
    otx_pulse     — AlienVault OTX threat pulse created (OTX)

Adding a paid source: implement CyberThreatSource.fetch() in a new class
and pass it to ingest_cyber_threats(sources=[...]).

Usage:
    python ingestion/cyber_threat_ingestion.py             # last 30 days
    python ingestion/cyber_threat_ingestion.py --start 2024-01-01 --end 2024-03-31
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path
from typing import Protocol

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_NVD_URL  = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_CISA_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_OTX_URL  = "https://otx.alienvault.com/api/v1/pulses/subscribed"

_SCHEMA = {"date": pl.Date, "source": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def _get_cvss_score(cve: dict) -> float | None:
    """Extract the highest available CVSS base score from a CVE object."""
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            return entries[0].get("cvssData", {}).get("baseScore")
    return None


class CyberThreatSource(Protocol):
    """Protocol for pluggable threat data sources."""

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Return rows with schema (date Date, source Utf8, metric Utf8, value Float64)."""
        ...


class NVDSource:
    """NIST NVD CVE API v2 — no API key required (rate limited to 5 req/30s)."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("NVD_API_KEY")

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        rows: list[dict] = []
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)

        # NVD accepts at most 120-day windows; chunk to 30 days for safety
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + datetime.timedelta(days=29), end)
            s_str = f"{chunk_start.isoformat()}T00:00:00.000"
            e_str = f"{chunk_end.isoformat()}T23:59:59.999"

            start_idx = 0
            while True:
                params: dict = {
                    "pubStartDate": s_str,
                    "pubEndDate": e_str,
                    "resultsPerPage": 2000,
                    "startIndex": start_idx,
                }
                if self._api_key:
                    params["apiKey"] = self._api_key

                try:
                    resp = requests.get(_NVD_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    _LOG.warning("[NVD] Fetch failed: %s", exc)
                    break

                vulns = data.get("vulnerabilities", [])
                total = data.get("totalResults", 0)

                for v in vulns:
                    cve = v.get("cve", {})
                    published = (cve.get("published") or "")[:10]
                    if not published:
                        continue
                    score = _get_cvss_score(cve)
                    if score is None or score < 7.0:
                        continue
                    metric = "cve_critical" if score >= 9.0 else "cve_high"
                    rows.append({
                        "date": datetime.date.fromisoformat(published),
                        "source": "nvd",
                        "metric": metric,
                        "value": 1.0,
                    })

                start_idx += len(vulns)
                # Respect rate limit: 5 req/30s without key, 50 req/30s with key
                time.sleep(1 if self._api_key else 6)
                if start_idx >= total or not vulns:
                    break

            chunk_start = chunk_end + datetime.timedelta(days=1)

        return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


class CISASource:
    """CISA Known Exploited Vulnerabilities feed — no key required."""

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)

        try:
            resp = requests.get(_CISA_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            _LOG.warning("[CISA] Fetch failed: %s", exc)
            return _empty()

        rows: list[dict] = []
        for entry in data.get("vulnerabilities", []):
            date_str = entry.get("dateAdded", "")
            if not date_str:
                continue
            try:
                d = datetime.date.fromisoformat(date_str)
            except ValueError:
                continue
            if not (start <= d <= end):
                continue
            rows.append({
                "date": d,
                "source": "cisa",
                "metric": "cisa_kev",
                "value": 1.0,
            })

        return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


class OTXSource:
    """AlienVault Open Threat Exchange — free API key required (OTX_API_KEY in .env).

    Degrades gracefully to empty DataFrame when key is absent.
    Register free at https://otx.alienvault.com
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("OTX_API_KEY")

    def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
        if not self._api_key:
            _LOG.debug("[OTX] OTX_API_KEY not set — skipping OTX source")
            return _empty()

        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)
        headers = {"X-OTX-API-KEY": self._api_key}
        rows: list[dict] = []
        url: str | None = _OTX_URL

        while url:
            params = {"modified_since": f"{start_date}T00:00:00", "limit": 50}
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                _LOG.warning("[OTX] Fetch failed: %s", exc)
                break

            for pulse in data.get("results", []):
                created_str = (pulse.get("created") or "")[:10]
                if not created_str:
                    continue
                try:
                    d = datetime.date.fromisoformat(created_str)
                except ValueError:
                    continue
                if not (start <= d <= end):
                    continue
                rows.append({
                    "date": d,
                    "source": "otx",
                    "metric": "otx_pulse",
                    "value": 1.0,
                })

            url = data.get("next")
            time.sleep(1)

        return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


def ingest_cyber_threats(
    start_date: str,
    end_date: str,
    output_dir: Path = Path("data/raw/cyber_threat"),
    sources: list | None = None,
) -> None:
    """Fetch threat events from all sources and write per-date parquet files.

    output_dir: root of Hive partition tree, e.g. data/raw/cyber_threat/
    sources: override list of CyberThreatSource instances (default: NVD + CISA + OTX)
    """
    if sources is None:
        sources = [NVDSource(), CISASource(), OTXSource()]

    frames: list[pl.DataFrame] = []
    for source in sources:
        df = source.fetch(start_date, end_date)
        if not df.is_empty():
            frames.append(df)

    if not frames:
        _LOG.info("[CyberThreat] No threat data fetched for %s–%s", start_date, end_date)
        return

    combined = pl.concat(frames)

    # Write one parquet file per date partition
    for date_val, group in combined.group_by("date"):
        date_str = str(date_val[0]) if isinstance(date_val, tuple) else str(date_val)
        out_path = output_dir / f"date={date_str}" / "threats.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        group.write_parquet(out_path, compression="snappy")
        _LOG.info("[CyberThreat] Wrote %d rows → %s", len(group), out_path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest cyber threat data.")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    today = datetime.date.today()
    end   = datetime.date.fromisoformat(args.end)   if args.end   else today
    start = datetime.date.fromisoformat(args.start) if args.start else today - datetime.timedelta(days=30)

    ingest_cyber_threats(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        output_dir=Path("data/raw/cyber_threat"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cyber_threat_ingestion.py -v 2>&1 | tail -20
```

Expected: all 9 tests PASS. (The OTX key test may require `monkeypatch` — if any fail, read the error carefully.)

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -m 'not integration' -q 2>&1 | tail -5
```

Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add ingestion/cyber_threat_ingestion.py tests/test_cyber_threat_ingestion.py
git commit -m "feat: cyber threat ingestion from NVD, CISA KEV, AlienVault OTX (pluggable sources)"
```

---

### Task 3: Cyber Threat Features

**Files:**
- Create: `processing/cyber_threat_features.py`
- Create: `tests/test_cyber_threat_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cyber_threat_features.py`:

```python
"""Tests for processing/cyber_threat_features.py."""
import datetime
from pathlib import Path

import polars as pl
import pytest


def _write_threats(threats_dir: Path, rows: list[dict]) -> None:
    """Write raw threat rows to the Hive-partitioned parquet structure."""
    import polars as pl
    df = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "source": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )
    for date_val, group in df.group_by("date"):
        date_str = str(date_val[0]) if isinstance(date_val, tuple) else str(date_val)
        out = threats_dir / f"date={date_str}" / "threats.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        group.write_parquet(out, compression="snappy")


def test_build_cyber_threat_features_has_all_columns(tmp_path):
    """build_cyber_threat_features returns all 7 CYBER_THREAT_FEATURE_COLS."""
    threats_dir = tmp_path / "cyber_threat"
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 15), "source": "cisa", "metric": "cisa_kev", "value": 1.0},
    ])

    from processing.cyber_threat_features import build_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = build_cyber_threat_features(threats_dir)

    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"
    assert "date" in result.columns


def test_build_cyber_threat_features_7d_rolling_sum(tmp_path):
    """cve_critical_7d is the 7-day rolling count of cve_critical events."""
    threats_dir = tmp_path / "cyber_threat"
    # 3 critical CVEs on Jan 15, 2 on Jan 16
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 16), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 16), "source": "nvd", "metric": "cve_critical", "value": 1.0},
    ])

    from processing.cyber_threat_features import build_cyber_threat_features
    result = build_cyber_threat_features(threats_dir).sort("date")

    jan15 = result.filter(pl.col("date") == datetime.date(2024, 1, 15))
    jan16 = result.filter(pl.col("date") == datetime.date(2024, 1, 16))

    assert jan15["cve_critical_7d"][0] == pytest.approx(3.0)
    assert jan16["cve_critical_7d"][0] == pytest.approx(5.0)  # 3 + 2 within 7 days


def test_build_cyber_threat_features_index_in_bounds(tmp_path):
    """cyber_threat_index_7d is always in [0, 1]."""
    threats_dir = tmp_path / "cyber_threat"
    # Large spike to trigger normalization
    rows = []
    for d in range(1, 20):
        for _ in range(50):
            rows.append({
                "date": datetime.date(2024, 1, d),
                "source": "nvd",
                "metric": "cve_critical",
                "value": 1.0,
            })
    _write_threats(threats_dir, rows)

    from processing.cyber_threat_features import build_cyber_threat_features
    result = build_cyber_threat_features(threats_dir)

    assert result["cyber_threat_index_7d"].min() >= 0.0
    assert result["cyber_threat_index_7d"].max() <= 1.0


def test_build_cyber_threat_features_missing_dir_returns_empty(tmp_path):
    """Returns empty DataFrame (with correct schema) when threats_dir doesn't exist."""
    from processing.cyber_threat_features import build_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = build_cyber_threat_features(tmp_path / "nonexistent")

    assert result.is_empty()
    for col in ["date"] + CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns, f"Missing column in empty result: {col}"


def test_join_cyber_threat_features_adds_columns(tmp_path):
    """join_cyber_threat_features adds all 7 feature columns to the input df."""
    threats_dir = tmp_path / "cyber_threat"
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
    ])

    df = pl.DataFrame({
        "ticker": ["NVDA", "MSFT"],
        "date": [datetime.date(2024, 1, 15), datetime.date(2024, 1, 15)],
    })

    from processing.cyber_threat_features import join_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = join_cyber_threat_features(df, threats_dir)

    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns, f"Missing column after join: {col}"
    # Original columns preserved
    assert "ticker" in result.columns
    assert len(result) == 2


def test_join_cyber_threat_features_missing_dir_zero_fills(tmp_path):
    """When threats_dir missing, all cyber threat columns are 0.0 (not null)."""
    df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": [datetime.date(2024, 1, 15)],
    })

    from processing.cyber_threat_features import join_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = join_cyber_threat_features(df, tmp_path / "nonexistent")

    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns
        assert result[col][0] == pytest.approx(0.0), f"{col} should be 0.0 when no data"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cyber_threat_features.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'processing.cyber_threat_features'`

- [ ] **Step 3: Create processing/cyber_threat_features.py**

```python
"""Cyber threat regime features — market-wide signals joined by date.

Reads raw threat events from data/raw/cyber_threat/date=*/threats.parquet
(written by ingestion/cyber_threat_ingestion.py) and produces 7-day and
30-day rolling window aggregate features.

Features:
    cve_critical_7d      — CVSS >= 9.0 CVEs published, 7-day rolling sum
    cve_high_7d          — CVSS 7-8.9 CVEs published, 7-day rolling sum
    cisa_kev_7d          — CISA KEV entries added, 7-day rolling sum
    otx_pulse_7d         — AlienVault OTX threat pulses, 7-day rolling sum
    cyber_threat_index_7d — composite normalized score in [0, 1]
    cve_critical_30d     — CVSS >= 9.0 CVEs published, 30-day rolling sum
    cisa_kev_30d         — CISA KEV entries added, 30-day rolling sum

These are date-keyed (not ticker-specific). All tickers on a given date
receive the same threat feature values.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

CYBER_THREAT_FEATURE_COLS: list[str] = [
    "cve_critical_7d",
    "cve_high_7d",
    "cisa_kev_7d",
    "otx_pulse_7d",
    "cyber_threat_index_7d",
    "cve_critical_30d",
    "cisa_kev_30d",
]

_EMPTY_SCHEMA = {"date": pl.Date} | {c: pl.Float64 for c in CYBER_THREAT_FEATURE_COLS}


def build_cyber_threat_features(threats_dir: Path) -> pl.DataFrame:
    """Aggregate raw threat events into daily rolling-window features.

    Args:
        threats_dir: Root of the Hive-partitioned raw threat data,
                     e.g. data/raw/cyber_threat/ (contains date=*/threats.parquet).

    Returns:
        DataFrame with columns: [date] + CYBER_THREAT_FEATURE_COLS.
        Empty DataFrame (correct schema) if no data found.
    """
    files = sorted(threats_dir.glob("date=*/threats.parquet")) if threats_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    raw = pl.concat([pl.read_parquet(f) for f in files])
    if raw.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    # Pivot to wide format: one row per date, one column per metric
    daily = (
        raw.group_by(["date", "metric"])
        .agg(pl.col("value").sum())
        .pivot(on="metric", values="value", index="date")
        .fill_null(0.0)
        .sort("date")
    )

    # Ensure all expected raw metric columns exist
    for col in ("cve_critical", "cve_high", "cisa_kev", "otx_pulse"):
        if col not in daily.columns:
            daily = daily.with_columns(pl.lit(0.0).alias(col))

    # 7-day rolling sums
    daily = daily.with_columns([
        pl.col("cve_critical").rolling_sum(window_size=7, min_periods=1).alias("cve_critical_7d"),
        pl.col("cve_high").rolling_sum(window_size=7, min_periods=1).alias("cve_high_7d"),
        pl.col("cisa_kev").rolling_sum(window_size=7, min_periods=1).alias("cisa_kev_7d"),
        pl.col("otx_pulse").rolling_sum(window_size=7, min_periods=1).alias("otx_pulse_7d"),
    ])

    # 30-day rolling sums
    daily = daily.with_columns([
        pl.col("cve_critical").rolling_sum(window_size=30, min_periods=1).alias("cve_critical_30d"),
        pl.col("cisa_kev").rolling_sum(window_size=30, min_periods=1).alias("cisa_kev_30d"),
    ])

    # Composite threat index: weighted sum normalised to [0, 1]
    # weighted = cve_critical_7d * 3 + cve_high_7d + cisa_kev_7d * 2
    # divided by 30-day rolling max of weighted (floor 1 to avoid div-by-zero)
    daily = daily.with_columns(
        (pl.col("cve_critical_7d") * 3 + pl.col("cve_high_7d") + pl.col("cisa_kev_7d") * 2)
        .alias("_weighted")
    )
    daily = daily.with_columns(
        pl.col("_weighted").rolling_max(window_size=30, min_periods=1).alias("_weighted_max_30d")
    )
    daily = daily.with_columns(
        (pl.col("_weighted") / pl.col("_weighted_max_30d").clip(lower_bound=1.0))
        .clip(lower_bound=0.0, upper_bound=1.0)
        .alias("cyber_threat_index_7d")
    )

    return daily.select(["date"] + CYBER_THREAT_FEATURE_COLS)


def join_cyber_threat_features(
    df: pl.DataFrame,
    threats_dir: Path,
) -> pl.DataFrame:
    """Left-join cyber threat features to df by date. Missing dates zero-fill.

    Args:
        df: Input DataFrame with a 'date' column (pl.Date).
        threats_dir: Root of raw cyber threat parquet tree.

    Returns:
        df with CYBER_THREAT_FEATURE_COLS appended. All values are Float64.
        Zero-filled (not null) when no threat data is available.
    """
    features = build_cyber_threat_features(threats_dir)

    if features.is_empty():
        for col in CYBER_THREAT_FEATURE_COLS:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        return df

    result = df.join(features, on="date", how="left")

    # Zero-fill any dates not in the threat data
    for col in CYBER_THREAT_FEATURE_COLS:
        result = result.with_columns(pl.col(col).fill_null(0.0))

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cyber_threat_features.py -v 2>&1 | tail -15
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -m 'not integration' -q 2>&1 | tail -5
```

Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add processing/cyber_threat_features.py tests/test_cyber_threat_features.py
git commit -m "feat: cyber threat features — 7d/30d rolling CVE, CISA KEV, OTX pulse + composite index"
```

---

### Task 4: Model Integration

**Files:**
- Modify: `models/train.py`
- Modify: `models/inference.py`
- Modify: `tests/test_train.py` (add FEATURE_COLS count test)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train.py` (after the existing import block, alongside the other constant tests):

```python
def test_feature_cols_includes_cyber_threat():
    """FEATURE_COLS must contain all 7 CYBER_THREAT_FEATURE_COLS after integration."""
    from models.train import FEATURE_COLS
    from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS
    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 55, f"Expected 55 features, got {len(FEATURE_COLS)}"


def test_tier_short_includes_cyber_threat_7d_features():
    """Short tier must include the 5 cyber threat _7d features, not the _30d ones."""
    from models.train import TIER_FEATURE_COLS
    short = TIER_FEATURE_COLS["short"]
    short_cyber = ["cve_critical_7d", "cve_high_7d", "cisa_kev_7d", "otx_pulse_7d", "cyber_threat_index_7d"]
    for col in short_cyber:
        assert col in short, f"{col} missing from short tier"
    # 30d features should NOT be in short tier
    assert "cve_critical_30d" not in short
    assert "cisa_kev_30d" not in short


def test_tier_long_excludes_cyber_threat():
    """Long tier must NOT include any cyber threat features."""
    from models.train import TIER_FEATURE_COLS
    from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS
    for col in CYBER_THREAT_FEATURE_COLS:
        assert col not in TIER_FEATURE_COLS["long"], f"{col} should not be in long tier"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_train.py::test_feature_cols_includes_cyber_threat \
                 tests/test_train.py::test_tier_short_includes_cyber_threat_7d_features \
                 tests/test_train.py::test_tier_long_excludes_cyber_threat -v 2>&1 | tail -15
```

Expected: FAIL — `FEATURE_COLS` has 48 features, cyber threat columns absent.

- [ ] **Step 3: Update models/train.py — add import and extend FEATURE_COLS**

At the top of `models/train.py`, in the `from processing.*` import block, add:

```python
from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS, join_cyber_threat_features
```

Then find the `FEATURE_COLS` definition (currently ends with `+ FX_FEATURE_COLS`) and extend it:

```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS + ENERGY_FEATURE_COLS
    + SUPPLY_CHAIN_FEATURE_COLS + FX_FEATURE_COLS
    + CYBER_THREAT_FEATURE_COLS  # 48 → 55 features total
)
```

- [ ] **Step 4: Update TIER_FEATURE_COLS in train.py**

Find the `TIER_FEATURE_COLS` dict definition and replace it with:

```python
# The 5 cyber threat features with 7d windows belong in short + medium tiers.
# The 30d features (cve_critical_30d, cisa_kev_30d) are medium-only.
_CYBER_THREAT_SHORT_COLS = [c for c in CYBER_THREAT_FEATURE_COLS if c.endswith("_7d")]

TIER_FEATURE_COLS: dict[str, list[str]] = {
    "short": (
        PRICE_FEATURE_COLS
        + SENTIMENT_FEATURE_COLS
        + INSIDER_FEATURE_COLS
        + SHORT_INTEREST_FEATURE_COLS
        + _CYBER_THREAT_SHORT_COLS   # 5 features: *_7d only
    ),
    "medium": FEATURE_COLS,          # all 55 features
    "long": (
        PRICE_FEATURE_COLS
        + FUND_FEATURE_COLS
        + EARNINGS_FEATURE_COLS
        + GRAPH_FEATURE_COLS
        + OWNERSHIP_FEATURE_COLS
        + ENERGY_FEATURE_COLS
        + SUPPLY_CHAIN_FEATURE_COLS
        + FX_FEATURE_COLS
        # cyber threat features excluded — noise at year+ horizons
    ),
}
```

- [ ] **Step 5: Add join_cyber_threat_features call to build_training_dataset in train.py**

In `build_training_dataset`, after the `join_fx_features` call and before the final `if horizon_tag is not None:` block, add:

```python
    # Join cyber threat regime features (date-keyed market-wide signals)
    cyber_threat_dir = fundamentals_dir.parent.parent / "cyber_threat"
    df = join_cyber_threat_features(df, cyber_threat_dir)
```

`fundamentals_dir.parent.parent` resolves `data/raw/financials/fundamentals` → `data/raw/financials` → `data/raw`. So `cyber_threat_dir` = `data/raw/cyber_threat`.

- [ ] **Step 6: Run train tests to verify they pass**

```bash
python -m pytest tests/test_train.py::test_feature_cols_includes_cyber_threat \
                 tests/test_train.py::test_tier_short_includes_cyber_threat_7d_features \
                 tests/test_train.py::test_tier_long_excludes_cyber_threat -v 2>&1 | tail -15
```

Expected: all 3 PASS.

- [ ] **Step 7: Update models/inference.py — add import and join call**

At the top of `models/inference.py`, add to the `from models.train import (...)` block:

```python
from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS, join_cyber_threat_features
```

In `_build_feature_df`, after the `join_fx_features` call and before `return df`, add:

```python
    cyber_threat_dir = data_dir / "cyber_threat"
    df = join_cyber_threat_features(df, cyber_threat_dir)
```

`data_dir` is the function's parameter (defaults to `Path("data/raw")`), so `data_dir / "cyber_threat"` = `data/raw/cyber_threat`.

Also add the missing-directory fallback (it's already handled inside `join_cyber_threat_features` via zero-fill, so no additional if/else block is needed — the join function is safe to call regardless).

- [ ] **Step 8: Run full suite**

```bash
python -m pytest tests/ -m 'not integration' -q 2>&1 | tail -5
```

Expected: all passing (count ≥ 263 + 3 new train tests).

- [ ] **Step 9: Commit**

```bash
git add models/train.py models/inference.py tests/test_train.py
git commit -m "feat: wire cyber threat features into FEATURE_COLS, TIER_FEATURE_COLS, train, inference"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Layer 12 `cyber_pureplay` (5 tickers) | Task 1 |
| Layer 13 `cyber_platform` (9 tickers) | Task 1 |
| `CyberThreatSource` protocol | Task 2 |
| `NVDSource`, `CISASource`, `OTXSource` | Task 2 |
| OTX degrades gracefully without key | Task 2 |
| `ingest_cyber_threats()` writes per-date parquet | Task 2 |
| 7 `CYBER_THREAT_FEATURE_COLS` | Task 3 |
| `build_cyber_threat_features()` rolling windows | Task 3 |
| `cyber_threat_index_7d` in [0, 1] | Task 3 |
| `join_cyber_threat_features()` zero-fills missing | Task 3 |
| `FEATURE_COLS` 48 → 55 | Task 4 |
| Short tier: 5 cyber `_7d` features only | Task 4 |
| Medium tier: all 7 cyber features (via FEATURE_COLS) | Task 4 |
| Long tier: no cyber features | Task 4 |
| `build_training_dataset` joins cyber threat | Task 4 |
| `_build_feature_df` joins cyber threat | Task 4 |
| DARK.L exchange="L", currency="GBP" (non-USD pattern) | Task 1 |
| CHKP + CYBR country="IL" | Task 1 |

All requirements covered.

**Placeholder scan:** No TBD, TODO, or vague steps. All code blocks are complete and syntactically correct.

**Type consistency:**
- `build_cyber_threat_features(threats_dir: Path) -> pl.DataFrame` — used consistently in Task 3 and 4.
- `join_cyber_threat_features(df: pl.DataFrame, threats_dir: Path) -> pl.DataFrame` — consistent across Tasks 3 and 4.
- `CYBER_THREAT_FEATURE_COLS: list[str]` — imported in train.py and inference.py from the same module.
- `_CYBER_THREAT_SHORT_COLS` is defined locally in train.py using a list comprehension on `CYBER_THREAT_FEATURE_COLS` — consistent.
