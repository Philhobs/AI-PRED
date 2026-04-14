"""Unit tests for ingestion/edgar_fundamentals_ingestion.py — no network calls."""
import datetime
import polars as pl
import pytest

from ingestion.edgar_fundamentals_ingestion import (
    _filter_quarterly,
    _filter_annual,
    _compute_derived,
    _compute_valuation_ratios,
    ANNUAL_FILERS,
    CIK_MAP,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_record(start, end, val, filed="2023-05-01", form="10-Q"):
    """Build an EDGAR-format record."""
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form,
            "accn": "0001234567-23-001"}

Q1 = _make_record("2023-01-01", "2023-03-31", 15_000_000_000)               # 89 days — quarterly ✓
YTD_H1 = _make_record("2023-01-01", "2023-06-30", 30_000_000_000)           # 180 days — YTD ✗
YTD_9M = _make_record("2023-01-01", "2023-09-30", 45_000_000_000)           # 272 days — YTD ✗
ANNUAL = _make_record("2023-01-01", "2023-12-31", 60_000_000_000)           # 364 days — annual ✗
Q1_AMENDED = _make_record("2023-01-01", "2023-03-31", 15_100_000_000,       # same end, later filed
                           filed="2023-05-15")
Q2 = _make_record("2023-04-01", "2023-06-30", 16_000_000_000)               # 90 days — quarterly ✓
ANNUAL_20F = _make_record("2022-01-01", "2022-12-31", 70_000_000_000,       # 364 days — annual ✓ for TSM
                           form="20-F")


# ── Test 1: quarterly filter keeps 90-day periods ─────────────────────────────

def test_filter_quarterly_keeps_90day_periods():
    result = _filter_quarterly([Q1, Q2])
    assert len(result) == 2
    ends = [r["end"] for r in result]
    assert "2023-03-31" in ends
    assert "2023-06-30" in ends


# ── Test 2: quarterly filter excludes YTD and annual ─────────────────────────

def test_filter_quarterly_excludes_ytd_and_annual():
    result = _filter_quarterly([Q1, YTD_H1, YTD_9M, ANNUAL])
    assert len(result) == 1
    assert result[0]["end"] == "2023-03-31"


# ── Test 3: quarterly filter keeps latest amendment ───────────────────────────

def test_filter_quarterly_deduplicates_amendments():
    result = _filter_quarterly([Q1, Q1_AMENDED])
    assert len(result) == 1
    assert result[0]["val"] == 15_100_000_000   # amended value wins


# ── Test 4: annual filter keeps 365-day periods, excludes quarterly ───────────

def test_filter_annual_keeps_365day_and_excludes_quarterly():
    result = _filter_annual([Q1, ANNUAL, ANNUAL_20F])
    assert len(result) == 2
    ends = [r["end"] for r in result]
    assert "2023-12-31" in ends
    assert "2022-12-31" in ends
    assert "2023-03-31" not in ends
