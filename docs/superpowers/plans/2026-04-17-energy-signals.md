# Energy Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new features (39 → 43 `FEATURE_COLS`) capturing US grid tightness, international energy capacity tailwinds, and energy deal quality to the AI infrastructure stock prediction model.

**Architecture:** Standard `ingestion/ → processing/ → FEATURE_COLS` pipeline. One new ingestion module (`eia_ingestion.py`), one new processing module (`energy_geo_features.py`), two targeted extensions to existing modules (`deal_ingestion.py`, `graph_features.py`), one manual CSV for per-ticker geographic exposure, and FEATURE_COLS wired in `train.py`.

**Tech Stack:** Python 3.11, Polars, requests, openpyxl (PJM Excel), EIA open API (free key at `api.eia.gov`), NumPy — all already in the project except openpyxl.

---

## Context You Must Know

**Existing modules this plan touches:**
- `ingestion/deal_ingestion.py` — parses SEC 8-K Item 1.01 deals. Key function: `_parse_8k_for_deals(text, filing_date)` returns `list[dict]` with fields: `party_a`, `party_b`, `deal_type`, `description`, `source`, `confidence`. We add `deal_mw` and `buyer_type` to each dict.
- `processing/graph_features.py` — computes `graph_partner_momentum_30d`, `graph_deal_count_90d`, `graph_hops_to_hyperscaler` per ticker per date. Has `_SCHEMA` dict and a per-ticker loop (lines 177–190). We add two functions + two schema entries.
- `ingestion/energy_geo_ingestion.py` — fetches OWID country energy data. Pattern: `fetch_*()` → `_parse_*()` → `save_*()`. Follow this exactly.
- `models/train.py` — has grouped FEATURE_COLS: `GRAPH_FEATURE_COLS`, `OWNERSHIP_FEATURE_COLS`, etc. We add `ENERGY_FEATURE_COLS` and a call to `join_energy_geo_features()` in `build_training_dataset()`.

**Spec:** `docs/superpowers/specs/2026-04-17-energy-signals.md`

---

## File Map

| File | Action |
|---|---|
| `ingestion/eia_ingestion.py` | Create |
| `processing/energy_geo_features.py` | Create |
| `data/manual/ticker_geo_exposure.csv` | Create |
| `ingestion/deal_ingestion.py` | Extend |
| `processing/graph_features.py` | Extend |
| `models/train.py` | Extend |
| `tests/test_eia_ingestion.py` | Create |
| `tests/test_energy_geo_features.py` | Create |
| `tests/test_deal_enrichment.py` | Create |

---

## Task 1: EIA + PJM Ingestion

**Files:**
- Create: `ingestion/eia_ingestion.py`
- Create: `tests/test_eia_ingestion.py`

- [ ] **Step 1: Install openpyxl (needed for PJM Excel)**

```bash
pip install openpyxl
```

Check it's added to requirements if the project has one:
```bash
grep -r "openpyxl\|requirements" pyproject.toml requirements*.txt 2>/dev/null | head -5
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_eia_ingestion.py
"""Tests for EIA capacity + PJM queue ingestion."""
from __future__ import annotations
import io
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_eia_response() -> dict:
    """Minimal valid EIA API v2 response."""
    return {
        "response": {
            "data": [
                {"period": "2025-01", "fueltypeid": "NUC", "fueltypeDescription": "Nuclear",
                 "capacity": 95.4, "capacity-units": "gigawatts"},
                {"period": "2025-01", "fueltypeid": "NG",  "fueltypeDescription": "Natural Gas",
                 "capacity": 618.5, "capacity-units": "gigawatts"},
                {"period": "2024-12", "fueltypeid": "NUC", "fueltypeDescription": "Nuclear",
                 "capacity": 95.2, "capacity-units": "gigawatts"},
                {"period": "2024-12", "fueltypeid": "NG",  "fueltypeDescription": "Natural Gas",
                 "capacity": 617.1, "capacity-units": "gigawatts"},
            ]
        }
    }


def _mock_pjm_excel() -> bytes:
    """Minimal PJM queue Excel file as bytes."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Queue Number", "Name", "Zone", "MW", "Status"])
    ws.append(["Q001", "Solar Farm A", "MAAC",  500,  "Active"])
    ws.append(["Q002", "Gas Plant B",  "AECO",  800,  "Active"])
    ws.append(["Q003", "Wind Farm C",  "SWVA",  300,  "Withdrawn"])
    ws.append(["Q004", "Nuclear D",    "ComEd", 1000, "Active"])  # not Virginia — should be excluded
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── Tests ──────────────────────────────────────────────────────────────────

def test_eia_capacity_schema(tmp_path):
    """EIA ingestion writes parquet with correct columns."""
    from ingestion.eia_ingestion import fetch_eia_capacity, save_eia_capacity

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: _mock_eia_response(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_eia_capacity(api_key="test_key")

    assert set(df.columns) == {"date", "fuel_type", "capacity_gw"}
    assert df["capacity_gw"].dtype == pl.Float64
    assert len(df) == 4  # 2 periods × 2 fuel types


def test_eia_capacity_fuel_types(tmp_path):
    """EIA ingestion includes nuclear and natural_gas rows."""
    from ingestion.eia_ingestion import fetch_eia_capacity

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: _mock_eia_response(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_eia_capacity(api_key="test_key")

    assert "nuclear" in df["fuel_type"].to_list()
    assert "natural_gas" in df["fuel_type"].to_list()


def test_pjm_queue_schema(tmp_path):
    """PJM ingestion writes parquet with correct columns."""
    from ingestion.eia_ingestion import fetch_pjm_queue

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=_mock_pjm_excel(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_pjm_queue()

    assert set(df.columns) == {"date", "zone", "queue_backlog_gw", "project_count"}
    assert df["queue_backlog_gw"].dtype == pl.Float64
    assert df["project_count"].dtype == pl.Int32


def test_pjm_filters_virginia_zones_only(tmp_path):
    """PJM ingestion excludes non-Virginia zones and withdrawn projects."""
    from ingestion.eia_ingestion import fetch_pjm_queue

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=_mock_pjm_excel(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_pjm_queue()

    # ComEd zone should be excluded; Withdrawn Q003 should be excluded
    # Active: Q001 (MAAC, 500MW) + Q002 (AECO, 800MW) = 1300 MW = 1.3 GW
    zones = df["zone"].to_list()
    assert "ComEd" not in zones
    assert "ALL_VIRGINIA" in zones
    virginia_row = df.filter(pl.col("zone") == "ALL_VIRGINIA")
    assert abs(float(virginia_row["queue_backlog_gw"][0]) - 1.3) < 0.01
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor"
pytest tests/test_eia_ingestion.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'ingestion.eia_ingestion'`

- [ ] **Step 4: Create `ingestion/eia_ingestion.py`**

```python
"""
EIA capacity + PJM interconnection queue ingestion.

Sources:
  - EIA API v2: monthly US electricity generation capacity by fuel type
  - PJM: Virginia corridor interconnection queue backlog

Usage:
    python ingestion/eia_ingestion.py

Requires EIA_API_KEY in .env (register free at https://www.eia.gov/opendata/)
"""
from __future__ import annotations

import io
import logging
import os
import time
from datetime import date
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_EIA_URL = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
_PJM_URL = (
    "https://www.pjm.com/-/media/planning/rtep/rtep-documents/"
    "active-interconnection-requests-report.ashx"
)
_VIRGINIA_ZONES = {"MAAC", "AECO", "SWVA"}
_FUEL_TYPE_MAP = {
    "NUC": "nuclear",
    "NG": "natural_gas",
    "SUN": "solar",
    "WND": "wind",
}


def fetch_eia_capacity(api_key: str, months: int = 36) -> pl.DataFrame:
    """
    Fetch monthly US installed capacity by fuel type from EIA API v2.

    Returns DataFrame with columns: date (date), fuel_type (str), capacity_gw (float64).
    """
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[]": "capacity",
        "facets[fueltypeid][]": list(_FUEL_TYPE_MAP.keys()),
        "facets[sectorid][]": "99",  # all sectors
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": str(months * len(_FUEL_TYPE_MAP) + 10),
    }

    resp = requests.get(_EIA_URL, params=params, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("response", {}).get("data", [])

    rows = []
    for r in records:
        fuel_id = r.get("fueltypeid", "")
        if fuel_id not in _FUEL_TYPE_MAP:
            continue
        period = r.get("period", "")
        cap = r.get("capacity")
        if not period or cap is None:
            continue
        # EIA returns capacity in gigawatts
        rows.append({
            "date": date.fromisoformat(period + "-01"),
            "fuel_type": _FUEL_TYPE_MAP[fuel_id],
            "capacity_gw": float(cap),
        })

    if not rows:
        _LOG.warning("[EIA] No capacity records returned — check API key and params")
        return pl.DataFrame(schema={"date": pl.Date, "fuel_type": pl.Utf8, "capacity_gw": pl.Float64})

    return pl.DataFrame(rows).sort("date", descending=True)


def fetch_pjm_queue() -> pl.DataFrame:
    """
    Fetch PJM interconnection queue, filter to Virginia corridor zones.

    Returns DataFrame with columns:
      date (date), zone (str), queue_backlog_gw (float64), project_count (int32)
    """
    import openpyxl

    resp = requests.get(_PJM_URL, timeout=60)
    resp.raise_for_status()

    wb = openpyxl.load_workbook(io.BytesIO(resp.content), read_only=True, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c else "" for c in next(rows_iter)]

    # Find column indices — PJM column names vary slightly by version
    def _find_col(candidates: list[str]) -> int | None:
        for name in candidates:
            for i, h in enumerate(header):
                if name.lower() in h.lower():
                    return i
        return None

    zone_col   = _find_col(["Zone", "LDA", "Zone Name"])
    mw_col     = _find_col(["MW", "Capacity MW", "Interconnection Request MW", "MW Requested"])
    status_col = _find_col(["Status", "Queue Status", "Interconnection Status"])

    if any(c is None for c in [zone_col, mw_col, status_col]):
        _LOG.warning(
            "[PJM] Could not find expected columns in queue report. Header: %s", header
        )
        return pl.DataFrame(
            schema={"date": pl.Date, "zone": pl.Utf8, "queue_backlog_gw": pl.Float64, "project_count": pl.Int32}
        )

    virginia_mw = 0.0
    project_count = 0
    for row in rows_iter:
        zone   = str(row[zone_col]).strip() if row[zone_col] else ""
        status = str(row[status_col]).strip() if row[status_col] else ""
        mw_val = row[mw_col]

        if zone not in _VIRGINIA_ZONES:
            continue
        if "withdrawn" in status.lower():
            continue
        if mw_val is None:
            continue

        try:
            virginia_mw += float(mw_val)
            project_count += 1
        except (ValueError, TypeError):
            continue

    wb.close()

    today = date.today()
    return pl.DataFrame([{
        "date": today,
        "zone": "ALL_VIRGINIA",
        "queue_backlog_gw": round(virginia_mw / 1000.0, 3),
        "project_count": project_count,
    }]).with_columns(pl.col("project_count").cast(pl.Int32))


def save_eia_capacity(df: pl.DataFrame, output_dir: Path) -> None:
    """Append-and-deduplicate EIA capacity records."""
    out_path = output_dir / "eia_capacity.parquet"
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        df = pl.concat([existing, df]).unique(subset=["date", "fuel_type"], keep="last")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    df.sort("date", descending=True).write_parquet(out_path, compression="snappy")
    _LOG.info("[EIA] Saved %d capacity records to %s", len(df), out_path)


def save_pjm_queue(df: pl.DataFrame, output_dir: Path) -> None:
    """Append-and-deduplicate PJM queue records."""
    out_path = output_dir / "pjm_queue.parquet"
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        df = pl.concat([existing, df]).unique(subset=["date", "zone"], keep="last")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    df.sort("date", descending=True).write_parquet(out_path, compression="snappy")
    _LOG.info("[PJM] Saved %d queue records to %s", len(df), out_path)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        _LOG.error("EIA_API_KEY not set in .env — skipping EIA capacity fetch")
        sys.exit(1)

    output_dir = Path(__file__).parent.parent / "data" / "raw" / "energy"

    _LOG.info("Fetching EIA capacity data...")
    eia_df = fetch_eia_capacity(api_key)
    save_eia_capacity(eia_df, output_dir)
    time.sleep(1)

    _LOG.info("Fetching PJM queue data...")
    pjm_df = fetch_pjm_queue()
    save_pjm_queue(pjm_df, output_dir)

    _LOG.info("Done. Run python processing/energy_geo_features.py to compute features.")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_eia_ingestion.py -v
```

Expected: 4 PASSED

- [ ] **Step 6: Run full suite to check for regressions**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass.

- [ ] **Step 7: Add EIA_API_KEY to .env.example**

Open `.env.example` and add:
```
EIA_API_KEY=your_key_here   # Register free at https://www.eia.gov/opendata/
```

- [ ] **Step 8: Commit**

```bash
git add ingestion/eia_ingestion.py tests/test_eia_ingestion.py .env.example
git commit -m "feat: add EIA capacity + PJM Virginia queue ingestion (Task 1)"
```

---

## Task 2: Energy Geo Features

**Files:**
- Create: `data/manual/ticker_geo_exposure.csv`
- Create: `processing/energy_geo_features.py`
- Create: `tests/test_energy_geo_features.py`

- [ ] **Step 1: Create `data/manual/ticker_geo_exposure.csv`**

```csv
ticker,region,weight
EQIX,north_america,0.55
EQIX,emea,0.12
EQIX,nordics,0.18
EQIX,asia_pacific,0.15
DLR,north_america,0.60
DLR,emea,0.25
DLR,asia_pacific,0.15
AMT,north_america,0.70
AMT,emea,0.20
AMT,asia_pacific,0.10
APLD,north_america,1.00
IREN,north_america,1.00
MSFT,north_america,0.55
MSFT,emea,0.25
MSFT,asia_pacific,0.20
AMZN,north_america,0.60
AMZN,emea,0.20
AMZN,asia_pacific,0.20
GOOGL,north_america,0.55
GOOGL,emea,0.25
GOOGL,asia_pacific,0.20
META,north_america,0.65
META,emea,0.20
META,nordics,0.15
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_energy_geo_features.py
"""Tests for energy geo feature computation."""
from __future__ import annotations
from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _make_pjm(backlog_gw: float, as_of: date) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [as_of],
        "zone": ["ALL_VIRGINIA"],
        "queue_backlog_gw": [backlog_gw],
        "project_count": [10],
    }).with_columns(pl.col("project_count").cast(pl.Int32))


def _make_eia(nuc_gw: float, gas_gw: float, as_of: date) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [as_of, as_of],
        "fuel_type": ["nuclear", "natural_gas"],
        "capacity_gw": [nuc_gw, gas_gw],
    })


def _make_owid(tmp_path: Path) -> Path:
    """Write a minimal OWID country energy parquet."""
    df = pl.DataFrame({
        "country": ["United States", "United States", "Norway", "Norway"],
        "year": [2023, 2022, 2023, 2022],
        "renewables_pct": [0.22, 0.21, 0.98, 0.97],
        "carbon_intensity_gco2_per_kwh": [386.0, 390.0, 18.0, 19.0],
    })
    out = tmp_path / "energy_geo" / "country_energy.parquet"
    out.parent.mkdir(parents=True)
    df.write_parquet(out)
    return tmp_path


def test_us_power_moat_score_range(tmp_path):
    """us_power_moat_score is in [0, 1]."""
    from processing.energy_geo_features import compute_us_power_moat_score

    pjm = _make_pjm(backlog_gw=200.0, as_of=date(2025, 1, 1))
    eia = _make_eia(nuc_gw=95.0, gas_gw=618.0, as_of=date(2025, 1, 1))
    score = compute_us_power_moat_score(pjm, eia, as_of=date(2025, 1, 1))
    assert 0.0 <= score <= 1.0, f"Expected [0, 1], got {score}"


def test_power_moat_zero_when_no_data():
    """us_power_moat_score is 0.0 when EIA or PJM data is missing."""
    from processing.energy_geo_features import compute_us_power_moat_score

    empty_pjm = pl.DataFrame(schema={"date": pl.Date, "zone": pl.Utf8,
                                      "queue_backlog_gw": pl.Float64, "project_count": pl.Int32})
    eia = _make_eia(nuc_gw=95.0, gas_gw=618.0, as_of=date(2025, 1, 1))
    score = compute_us_power_moat_score(empty_pjm, eia, as_of=date(2025, 1, 1))
    assert score == 0.0


def test_geo_tailwind_uses_exposure_weights(tmp_path):
    """geo_weighted_tailwind_score is a weighted average of regional tailwinds."""
    from processing.energy_geo_features import compute_geo_tailwind_score

    # OWID: US has moderate tailwind, Norway has very high tailwind
    owid_path = _make_owid(tmp_path)
    exposure = {"north_america": 0.5, "nordics": 0.5}  # equal split

    score_equal = compute_geo_tailwind_score(exposure, owid_path / "energy_geo" / "country_energy.parquet", as_of=date(2025, 1, 1))

    # All nordics should yield higher score than all north_america
    score_nordic  = compute_geo_tailwind_score({"nordics": 1.0}, owid_path / "energy_geo" / "country_energy.parquet", as_of=date(2025, 1, 1))
    score_us_only = compute_geo_tailwind_score({"north_america": 1.0}, owid_path / "energy_geo" / "country_energy.parquet", as_of=date(2025, 1, 1))

    assert score_nordic > score_us_only, "Nordic (98% renewable) should score higher than US"
    assert score_us_only <= score_equal <= score_nordic, "Equal-split score should be between the two"


def test_missing_ticker_defaults_to_north_america(tmp_path):
    """Ticker not in CSV gets north_america = 1.0 exposure."""
    from processing.energy_geo_features import load_geo_exposure

    csv_path = tmp_path / "ticker_geo_exposure.csv"
    csv_path.write_text("ticker,region,weight\nEQIX,nordics,1.0\n")

    exposure = load_geo_exposure(csv_path, ticker="NVDA")
    assert exposure == {"north_america": 1.0}, f"Expected north_america default, got {exposure}"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_energy_geo_features.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'processing.energy_geo_features'`

- [ ] **Step 4: Create `processing/energy_geo_features.py`**

```python
"""
Energy geography features — grid moat signal and international tailwind score.

Produces 2 features per ticker per date:
  us_power_moat_score      — PJM Virginia queue / baseload capacity (normalized 0-1)
                             High = demand >> supply = moat for power generators
  geo_weighted_tailwind_score — weighted average of regional energy tailwinds
                             using each ticker's geographic revenue exposure

Usage (standalone, generates features parquet):
    python processing/energy_geo_features.py

Called by models/train.py via join_energy_geo_features().
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

_LOG = logging.getLogger(__name__)

# Power layer tickers — only these get the us_power_moat_score signal
_POWER_TICKERS = frozenset({
    "CEG", "VST", "NRG", "TLN", "NEE", "SO", "EXC", "ETR",
    "GEV", "BWX", "OKLO", "SMR", "FSLR",
})

# OWID region → list of country names in that region
_REGION_COUNTRIES: dict[str, list[str]] = {
    "north_america": ["United States", "Canada"],
    "emea":          ["Germany", "United Kingdom", "France", "Netherlands"],
    "nordics":       ["Norway", "Sweden", "Iceland"],
    "asia_pacific":  ["Japan", "South Korea", "Singapore", "Malaysia"],
}


def load_geo_exposure(csv_path: Path, ticker: str) -> dict[str, float]:
    """
    Load {region: weight} for a ticker from the manual exposure CSV.
    Defaults to {"north_america": 1.0} if ticker not present.
    """
    if not csv_path.exists():
        return {"north_america": 1.0}
    df = pl.read_csv(csv_path).filter(pl.col("ticker") == ticker)
    if df.is_empty():
        return {"north_america": 1.0}
    return dict(zip(df["region"].to_list(), df["weight"].to_list()))


def compute_us_power_moat_score(
    pjm: pl.DataFrame,
    eia: pl.DataFrame,
    as_of: date,
    lookback_days: int = 365 * 3,
) -> float:
    """
    Compute US power moat score for as_of date.

    Formula: PJM Virginia queue backlog / (nuclear + gas capacity), normalized to [0, 1]
    using a rolling 3-year window. Returns 0.0 when data is unavailable.
    """
    # Get most recent PJM Virginia backlog on or before as_of
    recent_pjm = (
        pjm.filter((pl.col("zone") == "ALL_VIRGINIA") & (pl.col("date") <= as_of))
        .sort("date", descending=True)
        .head(1)
    )
    if recent_pjm.is_empty():
        return 0.0

    queue_gw = float(recent_pjm["queue_backlog_gw"][0])

    # Get most recent baseload capacity (nuclear + natural gas)
    recent_eia = (
        eia.filter(
            pl.col("fuel_type").is_in(["nuclear", "natural_gas"]) & (pl.col("date") <= as_of)
        )
        .sort("date", descending=True)
        .group_by("fuel_type")
        .agg(pl.col("capacity_gw").first())
    )
    if recent_eia.is_empty():
        return 0.0
    baseload_gw = float(recent_eia["capacity_gw"].sum())
    if baseload_gw == 0:
        return 0.0

    raw = queue_gw / baseload_gw

    # Normalize using rolling lookback window
    window_start = as_of - timedelta(days=lookback_days)
    history = pjm.filter(
        (pl.col("zone") == "ALL_VIRGINIA") &
        (pl.col("date") >= window_start) &
        (pl.col("date") <= as_of)
    )
    if len(history) < 2:
        return float(min(raw, 1.0))

    # Use constant baseload_gw for normalization (simplified — good enough for ranking)
    ratios = (history["queue_backlog_gw"] / baseload_gw).to_numpy()
    lo, hi = float(ratios.min()), float(ratios.max())
    if hi == lo:
        return 0.5
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


def compute_geo_tailwind_score(
    exposure: dict[str, float],
    owid_path: Path,
    as_of: date,
) -> float:
    """
    Compute geo-weighted energy tailwind score for a ticker.

    For each region, tailwind = 0.6 * renewable_growth_yoy + 0.4 * (1 - carbon_norm).
    Returns weighted average across the ticker's regional exposure.
    Returns 0.0 if OWID data unavailable.
    """
    if not owid_path.exists():
        return 0.0

    owid = pl.read_parquet(owid_path)
    as_of_year = as_of.year

    score_total = 0.0
    weight_total = 0.0

    for region, weight in exposure.items():
        countries = _REGION_COUNTRIES.get(region, [])
        if not countries:
            continue

        region_data = owid.filter(pl.col("country").is_in(countries))
        if region_data.is_empty():
            continue

        # Get most recent year ≤ as_of_year
        curr = region_data.filter(pl.col("year") <= as_of_year).sort("year", descending=True).head(len(countries))
        prev = region_data.filter(pl.col("year") <= as_of_year - 1).sort("year", descending=True).head(len(countries))

        if curr.is_empty():
            continue

        # Average renewables_pct across countries in region
        curr_ren = float(curr["renewables_pct"].mean())
        prev_ren = float(prev["renewables_pct"].mean()) if not prev.is_empty() else curr_ren
        ren_growth = max(0.0, curr_ren - prev_ren)  # YoY absolute growth in renewable share

        # Carbon intensity (normalize to 0-1 where 0=clean, 1=dirty; invert for tailwind)
        curr_carbon = float(curr["carbon_intensity_gco2_per_kwh"].mean())
        # 500 gCO2/kWh ≈ coal-heavy grid; 0 ≈ pure renewable. Clip then invert.
        carbon_norm = float(np.clip(curr_carbon / 500.0, 0.0, 1.0))
        carbon_tailwind = 1.0 - carbon_norm

        region_tailwind = 0.6 * ren_growth * 10 + 0.4 * carbon_tailwind  # scale growth to ~[0,1]
        score_total += weight * region_tailwind
        weight_total += weight

    if weight_total == 0:
        return 0.0
    return float(np.clip(score_total / weight_total, 0.0, 1.0))


def join_energy_geo_features(
    df: pl.DataFrame,
    energy_dir: Path | None = None,
    geo_csv: Path | None = None,
    owid_path: Path | None = None,
) -> pl.DataFrame:
    """
    Add us_power_moat_score and geo_weighted_tailwind_score to the training spine.

    Args:
        df: Training spine with columns [ticker, date, ...].
        energy_dir: Path to data/raw/energy/ (contains eia_capacity.parquet, pjm_queue.parquet).
        geo_csv: Path to data/manual/ticker_geo_exposure.csv.
        owid_path: Path to data/raw/energy/energy_geo/country_energy.parquet.

    Returns df with two new float64 columns.
    """
    _ROOT = Path(__file__).parent.parent

    if energy_dir is None:
        energy_dir = _ROOT / "data" / "raw" / "energy"
    if geo_csv is None:
        geo_csv = _ROOT / "data" / "manual" / "ticker_geo_exposure.csv"
    if owid_path is None:
        owid_path = energy_dir / "energy_geo" / "country_energy.parquet"

    # Load EIA and PJM data (empty DataFrames if files don't exist)
    eia_path = energy_dir / "eia_capacity.parquet"
    pjm_path = energy_dir / "pjm_queue.parquet"

    eia = pl.read_parquet(eia_path) if eia_path.exists() else pl.DataFrame(
        schema={"date": pl.Date, "fuel_type": pl.Utf8, "capacity_gw": pl.Float64}
    )
    pjm = pl.read_parquet(pjm_path) if pjm_path.exists() else pl.DataFrame(
        schema={"date": pl.Date, "zone": pl.Utf8, "queue_backlog_gw": pl.Float64, "project_count": pl.Int32}
    )

    # Pre-load geo exposure for all unique tickers
    unique_tickers = df["ticker"].unique().to_list()
    geo_exposure: dict[str, dict[str, float]] = {
        t: load_geo_exposure(geo_csv, t) for t in unique_tickers
    }

    # Compute features per (ticker, date) row
    moat_scores = []
    tailwind_scores = []

    for row in df.select(["ticker", "date"]).iter_rows(named=True):
        ticker = row["ticker"]
        as_of  = row["date"]

        # Power moat — only meaningful for power layer tickers
        if ticker in _POWER_TICKERS:
            moat = compute_us_power_moat_score(pjm, eia, as_of)
        else:
            moat = 0.0

        # Geo tailwind — all tickers
        tailwind = compute_geo_tailwind_score(geo_exposure[ticker], owid_path, as_of)

        moat_scores.append(moat)
        tailwind_scores.append(tailwind)

    return df.with_columns([
        pl.Series("us_power_moat_score", moat_scores, dtype=pl.Float64),
        pl.Series("geo_weighted_tailwind_score", tailwind_scores, dtype=pl.Float64),
    ])


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _LOG.info("Energy geo features are computed on-demand in train.py — no standalone run needed.")
    _LOG.info("Run python ingestion/eia_ingestion.py first to fetch the source data.")
```

- [ ] **Step 5: Run the tests**

```bash
pytest tests/test_energy_geo_features.py -v
```

Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add processing/energy_geo_features.py tests/test_energy_geo_features.py data/manual/ticker_geo_exposure.csv
git commit -m "feat: energy geo features — power moat score and geo tailwind score (Task 2)"
```

---

## Task 3: Deal Schema Enrichment (MW + Buyer Type)

**Files:**
- Modify: `ingestion/deal_ingestion.py` (extend `_parse_8k_for_deals`)
- Create: `tests/test_deal_enrichment.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deal_enrichment.py
"""Tests for MW extraction and buyer_type classification in deal_ingestion."""
from __future__ import annotations

import polars as pl
import pytest


def test_mw_extraction_standard_format():
    """'500 MW' in text → deal_mw == 500.0"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("The company agreed to a 500 MW power purchase agreement.") == pytest.approx(500.0)


def test_mw_extraction_with_commas():
    """'1,200 megawatts' → deal_mw == 1200.0"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("a 1,200 megawatt facility in Virginia") == pytest.approx(1200.0)


def test_mw_extraction_gigawatt():
    """'2 GW' → deal_mw == 2000.0"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("signed a 2 GW offtake agreement") == pytest.approx(2000.0)


def test_mw_extraction_returns_none_when_absent():
    """No capacity mention → None"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("The company entered into a supply agreement for materials.") is None


def test_buyer_type_hyperscaler():
    """'Microsoft Corporation' → buyer_type == 'hyperscaler'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Microsoft Corporation") == "hyperscaler"


def test_buyer_type_amazon():
    """'Amazon Web Services' → buyer_type == 'hyperscaler'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Amazon Web Services, Inc.") == "hyperscaler"


def test_buyer_type_crypto():
    """'Applied Digital' → buyer_type == 'crypto_miner'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Applied Digital Corporation") == "crypto_miner"


def test_buyer_type_other():
    """Unknown counterparty → buyer_type == 'other'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Acme Corporation Ltd") == "other"


def test_deals_parquet_has_new_columns(tmp_path):
    """After build_deals(), parquet includes deal_mw and buyer_type columns."""
    from ingestion.deal_ingestion import build_deals

    # Minimal 8-K text with a PPA mentioning MW + Microsoft
    mock_8k = (
        "Item 1.01. Microsoft Corporation entered into a 300 MW power purchase "
        "agreement with Constellation Energy Group Inc."
    )

    # Write a minimal manual CSV (required by build_deals)
    manual_csv = tmp_path / "deals_override.csv"
    manual_csv.write_text("date,party_a,party_b,deal_type,description,confidence\n")

    deals = build_deals(
        filings=[{"text": mock_8k, "date": "2026-01-15", "url": "https://example.com/8k"}],
        manual_csv_path=manual_csv,
        output_path=tmp_path / "deals.parquet",
    )

    assert "deal_mw" in deals.columns, "deal_mw column must be present"
    assert "buyer_type" in deals.columns, "buyer_type column must be present"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_deal_enrichment.py::test_mw_extraction_standard_format \
       tests/test_deal_enrichment.py::test_buyer_type_hyperscaler -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name '_extract_deal_mw'`

- [ ] **Step 3: Read the current deal_ingestion.py**

Read `ingestion/deal_ingestion.py` in full to find:
- Where `_parse_8k_for_deals()` is defined
- The exact structure of each dict it returns
- Where `build_deals()` constructs the final DataFrame
- Whether `_classify_deal_type()` already exists (it does — re-use its pattern)

- [ ] **Step 4: Add `_extract_deal_mw` and `_classify_buyer_type` to `deal_ingestion.py`**

Add these two functions immediately after the existing `_classify_deal_type()` function:

```python
import re as _re  # add to top-level imports if not already there

_MW_PATTERNS = [
    _re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*(?:GW|gigawatt)", _re.IGNORECASE),
    _re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*(?:MW|megawatt)", _re.IGNORECASE),
]

def _extract_deal_mw(text: str) -> float | None:
    """
    Extract MW capacity from 8-K deal text.

    Returns MW as float (GW converted × 1000), or None if not found.
    """
    for pat in _MW_PATTERNS:
        m = pat.search(text)
        if m:
            raw = float(m.group(1).replace(",", ""))
            # GW pattern appears first — multiply by 1000
            if "GW" in pat.pattern or "gigawatt" in pat.pattern.lower():
                return raw * 1000.0
            return raw
    return None


_HYPERSCALER_KEYWORDS = {"microsoft", "amazon", "aws", "google", "alphabet", "meta", "apple"}
_CRYPTO_KEYWORDS      = {"iren", "applied digital", "apld", "marathon digital", "riot platforms",
                         "core scientific", "bit digital", "bitfarms"}
_UTILITY_KEYWORDS     = {"duke energy", "dominion", "southern company", "exelon", "entergy",
                         "nextera", "eversource", "ameren", "xcel energy", "sempra"}

def _classify_buyer_type(counterparty: str) -> str:
    """
    Classify counterparty as hyperscaler, crypto_miner, utility, or other.
    """
    lower = counterparty.lower()
    if any(kw in lower for kw in _HYPERSCALER_KEYWORDS):
        return "hyperscaler"
    if any(kw in lower for kw in _CRYPTO_KEYWORDS):
        return "crypto_miner"
    if any(kw in lower for kw in _UTILITY_KEYWORDS):
        return "utility"
    return "other"
```

- [ ] **Step 5: Extend `_parse_8k_for_deals()` to populate new fields**

Inside `_parse_8k_for_deals()`, after each deal dict is built (before `deals.append(deal_dict)`), add:

```python
        # Extract MW capacity and buyer type
        deal_dict["deal_mw"] = _extract_deal_mw(text)
        # Classify the counterparty that is NOT the filing company as the buyer
        # party_b is typically the counterparty
        deal_dict["buyer_type"] = _classify_buyer_type(deal_dict.get("party_b", ""))
```

- [ ] **Step 6: Update `build_deals()` to include new columns in the output DataFrame**

In `build_deals()`, where the DataFrame is constructed from the list of deal dicts, ensure `deal_mw` and `buyer_type` columns are included. If using `pl.DataFrame(rows)`, Polars will infer them automatically from the dicts. Add a schema cast after construction:

```python
    # Cast new columns to their correct types
    df = df.with_columns([
        pl.col("deal_mw").cast(pl.Float64),
        pl.col("buyer_type").fill_null("other").cast(pl.Utf8),
    ])
```

For manual CSV deals that don't have these columns: add defaults in the manual deal loading section:
```python
    if "deal_mw" not in manual_df.columns:
        manual_df = manual_df.with_columns(pl.lit(None).cast(pl.Float64).alias("deal_mw"))
    if "buyer_type" not in manual_df.columns:
        manual_df = manual_df.with_columns(pl.lit("other").alias("buyer_type"))
```

- [ ] **Step 7: Run the deal enrichment tests**

```bash
pytest tests/test_deal_enrichment.py -v
```

Expected: All 9 PASSED.

- [ ] **Step 8: Run full suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

- [ ] **Step 9: Commit**

```bash
git add ingestion/deal_ingestion.py tests/test_deal_enrichment.py
git commit -m "feat: deal enrichment — add deal_mw and buyer_type to 8-K parsing (Task 3)"
```

---

## Task 4: Graph Feature Enrichment

**Files:**
- Modify: `processing/graph_features.py` (add 2 new feature functions + schema entries)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_deal_enrichment.py` (append after existing tests):

```python
def test_energy_deal_mw_90d_feature():
    """energy_deal_mw_90d sums MW of energy deals for the ticker in last 90d."""
    from processing.graph_features import _compute_energy_deal_mw_90d
    from datetime import date, timedelta

    as_of = date(2026, 1, 15)
    deals = pl.DataFrame({
        "date": [as_of - timedelta(days=30), as_of - timedelta(days=200)],
        "party_a": ["CEG", "CEG"],
        "party_b": ["MSFT", "AMZN"],
        "deal_type": ["power_purchase_agreement", "power_purchase_agreement"],
        "deal_mw": [500.0, 300.0],
        "buyer_type": ["hyperscaler", "hyperscaler"],
    })

    result = _compute_energy_deal_mw_90d("CEG", deals, as_of)
    assert result == pytest.approx(500.0), "Only the deal within 90d should be counted"


def test_energy_deal_mw_90d_null_treated_as_zero():
    """deal_mw = None is treated as 0 MW (don't penalize deals missing capacity info)."""
    from processing.graph_features import _compute_energy_deal_mw_90d
    from datetime import date, timedelta

    as_of = date(2026, 1, 15)
    deals = pl.DataFrame({
        "date": [as_of - timedelta(days=30)],
        "party_a": ["CEG"],
        "party_b": ["MSFT"],
        "deal_type": ["power_purchase_agreement"],
        "deal_mw": [None],
        "buyer_type": ["hyperscaler"],
    }).with_columns(pl.col("deal_mw").cast(pl.Float64))

    result = _compute_energy_deal_mw_90d("CEG", deals, as_of)
    assert result == pytest.approx(0.0)


def test_hyperscaler_ppa_count_90d_feature():
    """hyperscaler_ppa_count_90d counts PPAs where buyer_type is hyperscaler."""
    from processing.graph_features import _compute_hyperscaler_ppa_count_90d
    from datetime import date, timedelta

    as_of = date(2026, 1, 15)
    deals = pl.DataFrame({
        "date": [
            as_of - timedelta(days=10),   # within window, hyperscaler PPA
            as_of - timedelta(days=20),   # within window, crypto miner PPA
            as_of - timedelta(days=200),  # outside window, hyperscaler PPA
        ],
        "party_a": ["CEG", "CEG", "CEG"],
        "party_b": ["MSFT", "IREN", "AMZN"],
        "deal_type": ["power_purchase_agreement", "power_purchase_agreement", "power_purchase_agreement"],
        "deal_mw": [500.0, 100.0, 300.0],
        "buyer_type": ["hyperscaler", "crypto_miner", "hyperscaler"],
    })

    result = _compute_hyperscaler_ppa_count_90d("CEG", deals, as_of)
    assert result == 1, "Only 1 hyperscaler PPA within 90d window"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_deal_enrichment.py::test_energy_deal_mw_90d_feature -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name '_compute_energy_deal_mw_90d'`

- [ ] **Step 3: Read `processing/graph_features.py` lines 80–220**

Find:
- The `_compute_deal_count_90d()` function — copy its structure for the new functions
- The `_SCHEMA` dict — add two entries
- The per-ticker loop (lines 177–190) — add two new compute calls

- [ ] **Step 4: Add two new compute functions to `graph_features.py`**

Add immediately after `_compute_deal_count_90d()`:

```python
def _compute_energy_deal_mw_90d(
    ticker: str,
    deals: pl.DataFrame,
    as_of: date,
) -> float:
    """
    Total MW contracted by this ticker in energy deals within the last 90 days.

    Only counts power_purchase_agreement deal_type where ticker is party_a or party_b.
    Null deal_mw values are treated as 0 (deal exists but capacity not disclosed).
    Returns 0.0 when no data.
    """
    if deals.is_empty() or "deal_mw" not in deals.columns:
        return 0.0

    window_start = as_of - timedelta(days=90)
    ticker_deals = deals.filter(
        (pl.col("party_a") == ticker | pl.col("party_b") == ticker) &
        (pl.col("deal_type") == "power_purchase_agreement") &
        (pl.col("date") >= window_start) &
        (pl.col("date") <= as_of)
    )
    if ticker_deals.is_empty():
        return 0.0

    return float(ticker_deals["deal_mw"].fill_null(0.0).sum())


def _compute_hyperscaler_ppa_count_90d(
    ticker: str,
    deals: pl.DataFrame,
    as_of: date,
) -> int:
    """
    Count of PPAs where this ticker is the power seller and buyer is a hyperscaler,
    within the last 90 days.

    Returns 0 for non-power-company tickers (they won't appear in party_a/b for PPAs).
    """
    if deals.is_empty() or "buyer_type" not in deals.columns:
        return 0

    window_start = as_of - timedelta(days=90)
    hyper_deals = deals.filter(
        (pl.col("party_a") == ticker | pl.col("party_b") == ticker) &
        (pl.col("deal_type") == "power_purchase_agreement") &
        (pl.col("buyer_type") == "hyperscaler") &
        (pl.col("date") >= window_start) &
        (pl.col("date") <= as_of)
    )
    return len(hyper_deals)
```

- [ ] **Step 5: Add new features to the per-ticker loop and `_SCHEMA`**

In the per-ticker loop (around line 186), after the existing feature appends, add:

```python
        row["energy_deal_mw_90d"]         = float(_compute_energy_deal_mw_90d(ticker, deals, as_of))
        row["hyperscaler_ppa_count_90d"]  = float(_compute_hyperscaler_ppa_count_90d(ticker, deals, as_of))
```

In `_SCHEMA` (around line 192), add:
```python
    "energy_deal_mw_90d":        pl.Float64,
    "hyperscaler_ppa_count_90d": pl.Float64,
```

- [ ] **Step 6: Run the new graph feature tests**

```bash
pytest tests/test_deal_enrichment.py -v
```

Expected: All 12 tests PASSED (9 existing + 3 new).

- [ ] **Step 7: Run full suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

- [ ] **Step 8: Commit**

```bash
git add processing/graph_features.py tests/test_deal_enrichment.py
git commit -m "feat: graph features — energy_deal_mw_90d and hyperscaler_ppa_count_90d (Task 4)"
```

---

## Task 5: FEATURE_COLS Integration

**Files:**
- Modify: `models/train.py` (add ENERGY_FEATURE_COLS, call join_energy_geo_features)

- [ ] **Step 1: Read the current FEATURE_COLS block in `models/train.py`**

Find the `GRAPH_FEATURE_COLS` and `OWNERSHIP_FEATURE_COLS` definitions and the `FEATURE_COLS = ...` list that combines them all. Also find `build_training_dataset()` to see where feature joins happen.

- [ ] **Step 2: Add `ENERGY_FEATURE_COLS` to `train.py`**

After the existing `GRAPH_FEATURE_COLS` definition, add:

```python
ENERGY_FEATURE_COLS = [
    "us_power_moat_score",
    "geo_weighted_tailwind_score",
    "energy_deal_mw_90d",
    "hyperscaler_ppa_count_90d",
]
```

- [ ] **Step 3: Add to the combined `FEATURE_COLS` list**

Find the line that assembles all feature column groups (something like):
```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS +
    SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS +
    EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS + OWNERSHIP_FEATURE_COLS
)
```

Add `+ ENERGY_FEATURE_COLS` at the end:
```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS +
    SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS +
    EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS + OWNERSHIP_FEATURE_COLS
    + ENERGY_FEATURE_COLS
)
```

- [ ] **Step 4: Add import and join call to `build_training_dataset()`**

Add import near the top of `train.py` (with the other processing imports):
```python
from processing.energy_geo_features import join_energy_geo_features
```

In `build_training_dataset()`, after the existing `join_graph_features(...)` and `join_ownership_features(...)` calls, add:
```python
    df = join_energy_geo_features(df)  # uses default paths via Path(__file__).parent.parent
```

- [ ] **Step 5: Verify feature count**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor"
python -c "from models.train import FEATURE_COLS; print(f'Feature count: {len(FEATURE_COLS)}'); print(FEATURE_COLS[-6:])"
```

Expected:
```
Feature count: 43
['inst_momentum_2q', 'us_power_moat_score', 'geo_weighted_tailwind_score', 'energy_deal_mw_90d', 'hyperscaler_ppa_count_90d']
```

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -m "not integration" -q 2>&1 | tail -5
```

Expected: All pass (200+ tests).

- [ ] **Step 7: Run train.py to verify new features flow through**

```bash
python models/train.py 2>&1 | tail -10
```

Expected: Completes without errors. `[Train] Artifacts → models/artifacts` printed. Feature count in JSON should be 43:
```bash
python -c "import json; r=json.load(open('data/backtest/walk_forward_results.json')); print('feature_count:', r['feature_count'])"
```

Expected: `feature_count: 43`

- [ ] **Step 8: Commit**

```bash
git add models/train.py
git commit -m "feat: wire energy signals into FEATURE_COLS — model now trains on 43 features (Task 5)"
```

---

## Final Verification

```bash
# Run inference to confirm predictions_enriched.parquet includes new features indirectly
python -c "
from models.inference import run_inference
import polars as pl
run_inference('2026-04-17')
df = pl.read_parquet('data/predictions/date=2026-04-17/predictions.parquet')
print(f'Predictions written: {len(df)} tickers')
print(df.sort(\"rank\").head(5).select([\"rank\",\"ticker\",\"layer\",\"expected_annual_return\"]))
" 2>&1 | grep -v "UserWarning\|Sortedness"

# Check backtest summary
python -c "
import json
r = json.load(open('data/backtest/walk_forward_results.json'))
print('feature_count:', r['feature_count'])
print('summary:', r['summary'])
"
```

Expected: `feature_count: 43`, predictions for all 83 tickers, inference runs cleanly.
