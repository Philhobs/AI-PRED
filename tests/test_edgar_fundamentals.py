"""Unit tests for ingestion/edgar_fundamentals_ingestion.py — no network calls."""
import datetime
from unittest.mock import patch, MagicMock
import polars as pl
import pytest

from ingestion.edgar_fundamentals_ingestion import (
    _filter_quarterly,
    _filter_annual,
    _compute_derived,
    _compute_valuation_ratios,
    _fetch_xbrl,
    _to_period_series,
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


# ── Test: records with start=None are excluded ────────────────────────────────

def test_filter_quarterly_excludes_no_start_records():
    """Balance-sheet snapshot records have no 'start' — they must be excluded."""
    no_start = {"start": None, "end": "2023-03-31", "val": 99_000_000_000,
                "filed": "2023-05-01", "form": "10-Q", "accn": "0001234567-23-001"}
    result = _filter_quarterly([Q1, no_start])
    assert len(result) == 1
    assert result[0]["val"] == 15_000_000_000  # only Q1 kept


# ── Test: _fetch_xbrl returns [] on 404 ──────────────────────────────────────

def test_fetch_xbrl_returns_empty_on_404():
    """404 means concept not reported by this company — should return []."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("ingestion.edgar_fundamentals_ingestion.requests.get", return_value=mock_resp):
        result = _fetch_xbrl("0000789019", "NonExistentConcept")
    assert result == []


# ── Test: _to_period_series returns correct DataFrame ─────────────────────────

def test_to_period_series_returns_polars_df():
    records = [Q1, Q2]  # defined above in Task 1 fixtures
    df = _to_period_series(records, value_col="revenue", annual=False)
    assert isinstance(df, pl.DataFrame)
    assert "period_end" in df.columns
    assert "revenue" in df.columns
    assert len(df) == 2
    assert df.schema["period_end"] == pl.Date
    assert df.schema["revenue"] == pl.Float64
    # Values must match fixture
    assert df.sort("period_end")["revenue"].to_list() == [15_000_000_000.0, 16_000_000_000.0]


# ── Fixtures for derived metrics ──────────────────────────────────────────────

_INCOME_FIXTURE = pl.DataFrame({
    "period_end":       pl.Series([
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
        datetime.date(2023, 3, 31),
    ], dtype=pl.Date),
    "revenue":          [10_000.0, 10_500.0, 11_000.0, 11_500.0, 12_000.0],
    "gross_profit":     [ 6_000.0,  6_300.0,  6_600.0,  6_900.0,  7_200.0],
    "operating_income": [ 3_000.0,  3_150.0,  3_300.0,  3_450.0,  3_600.0],
    "net_income":       [ 2_500.0,  2_625.0,  2_750.0,  2_875.0,  3_000.0],
    "capex":            [   500.0,    525.0,    550.0,    575.0,    600.0],
})

_BALANCE_FIXTURE = pl.DataFrame({
    "period_end":          pl.Series([
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
        datetime.date(2023, 3, 31),
    ], dtype=pl.Date),
    "equity":              [50_000.0, 52_000.0, 54_000.0, 56_000.0, 58_000.0],
    "long_term_debt":      [10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0],
    "current_assets":      [20_000.0, 21_000.0, 22_000.0, 23_000.0, 24_000.0],
    "current_liabilities": [8_000.0,  8_400.0,  8_800.0,  9_200.0,  9_600.0],
    "shares_outstanding":  [1_000.0,  1_000.0,  1_000.0,  1_000.0,  1_000.0],
})


# ── Test: derived margins are computed correctly ───────────────────────────────

def test_compute_derived_metrics():
    df = _compute_derived(_INCOME_FIXTURE, _BALANCE_FIXTURE)
    # Q1 2023 row
    q = df.filter(pl.col("period_end") == datetime.date(2023, 3, 31))
    assert len(q) == 1
    row = q.row(0, named=True)
    # gross_margin = 7200 / 12000
    assert abs(row["gross_margin"] - 0.60) < 1e-6
    # operating_margin = 3600 / 12000
    assert abs(row["operating_margin"] - 0.30) < 1e-6
    # capex_to_revenue = 600 / 12000
    assert abs(row["capex_to_revenue"] - 0.05) < 1e-6
    # debt_to_equity = 10000 / 58000
    assert abs(row["debt_to_equity"] - (10_000.0 / 58_000.0)) < 1e-6
    # current_ratio = 24000 / 9600
    assert abs(row["current_ratio"] - 2.5) < 1e-6


# ── Test: revenue_growth_yoy looks back 4 quarters ───────────────────────────

def test_compute_derived_revenue_growth_yoy():
    df = _compute_derived(_INCOME_FIXTURE, _BALANCE_FIXTURE)
    q = df.filter(pl.col("period_end") == datetime.date(2023, 3, 31))
    row = q.row(0, named=True)
    # Q1 2023 revenue = 12000, Q1 2022 revenue = 10000
    # growth = (12000 - 10000) / 10000 = 0.20
    assert abs(row["revenue_growth_yoy"] - 0.20) < 1e-6
    # Q1 2022 has no prior year — should be null
    q_2022 = df.filter(pl.col("period_end") == datetime.date(2022, 3, 31))
    assert q_2022["revenue_growth_yoy"][0] is None


def test_compute_derived_yoy_handles_missing_quarter():
    """shift(4) would produce wrong results with gaps; calendar join should be immune."""
    # Q3 2022 is missing from the series
    income_with_gap = pl.DataFrame({
        "period_end": pl.Series([
            datetime.date(2022, 3, 31),   # Q1 2022
            datetime.date(2022, 6, 30),   # Q2 2022
            # Q3 2022 MISSING
            datetime.date(2022, 12, 31),  # Q4 2022
            datetime.date(2023, 3, 31),   # Q1 2023
        ], dtype=pl.Date),
        "revenue":          [10_000.0, 10_500.0, 11_500.0, 12_000.0],
        "gross_profit":     [ 6_000.0,  6_300.0,  6_900.0,  7_200.0],
        "operating_income": [ 3_000.0,  3_150.0,  3_450.0,  3_600.0],
        "net_income":       [ 2_500.0,  2_625.0,  2_875.0,  3_000.0],
        "capex":            [   500.0,    525.0,    575.0,    600.0],
    })
    balance_with_gap = pl.DataFrame({
        "period_end": pl.Series([
            datetime.date(2022, 3, 31),
            datetime.date(2022, 6, 30),
            datetime.date(2022, 12, 31),
            datetime.date(2023, 3, 31),
        ], dtype=pl.Date),
        "equity":              [50_000.0, 52_000.0, 56_000.0, 58_000.0],
        "long_term_debt":      [10_000.0, 10_000.0, 10_000.0, 10_000.0],
        "current_assets":      [20_000.0, 21_000.0, 23_000.0, 24_000.0],
        "current_liabilities": [ 8_000.0,  8_400.0,  9_200.0,  9_600.0],
        "shares_outstanding":  [ 1_000.0,  1_000.0,  1_000.0,  1_000.0],
    })
    df = _compute_derived(income_with_gap, balance_with_gap)
    q_2023_q1 = df.filter(pl.col("period_end") == datetime.date(2023, 3, 31))
    assert len(q_2023_q1) == 1
    row = q_2023_q1.row(0, named=True)
    # Q1 2023 = 12000, Q1 2022 = 10000 → growth = 0.20
    # With positional shift(4), this would be None (gap breaks indexing)
    # With calendar-aware join, this correctly computes 0.20
    assert row["revenue_growth_yoy"] is not None
    assert abs(row["revenue_growth_yoy"] - 0.20) < 1e-6
