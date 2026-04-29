"""Tests for the 5 new EDGAR-derived TTM metrics in edgar_fundamentals_ingestion.py."""
import datetime
import pyarrow as pa
import polars as pl
import pytest

from ingestion.edgar_fundamentals_ingestion import _SCHEMA, _compute_derived


# ── 8-quarter fixtures (enough history for capex_growth_yoy) ──────────────────

_INCOME_8Q = pl.DataFrame({
    "period_end": pl.Series([
        datetime.date(2021, 3, 31), datetime.date(2021, 6, 30),
        datetime.date(2021, 9, 30), datetime.date(2021, 12, 31),
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
    ], dtype=pl.Date),
    "revenue":          [10_000.0] * 8,
    "gross_profit":     [ 6_000.0] * 8,
    "operating_income": [ 3_000.0] * 8,
    "net_income":       [ 2_000.0] * 8,
    "capex":            [500.0, 500.0, 500.0, 500.0,   # prior TTM = 2000
                         600.0, 600.0, 600.0, 600.0],  # current TTM = 2400
    "rd_expense":       [1_000.0] * 8,                 # TTM R&D = 4000
})

_BALANCE_8Q = pl.DataFrame({
    "period_end": pl.Series([
        datetime.date(2021, 3, 31), datetime.date(2021, 6, 30),
        datetime.date(2021, 9, 30), datetime.date(2021, 12, 31),
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
    ], dtype=pl.Date),
    "equity":              [50_000.0] * 8,
    "long_term_debt":      [10_000.0] * 8,
    "current_assets":      [20_000.0] * 8,
    "current_liabilities": [ 8_000.0] * 8,
    "shares_outstanding":  [ 1_000.0] * 8,
})


def test_schema_has_17_columns():
    """_SCHEMA must have 17 fields: ticker + period_end + available_date + 14 ratio columns."""
    assert len(_SCHEMA) == 17
    assert _SCHEMA.field("available_date").type == pa.date32()
    assert _SCHEMA.field("net_income_margin").type == pa.float64()
    assert _SCHEMA.field("free_cash_flow_margin").type == pa.float64()
    assert _SCHEMA.field("capex_growth_yoy").type == pa.float64()
    assert _SCHEMA.field("revenue_growth_accel").type == pa.float64()
    assert _SCHEMA.field("research_to_revenue").type == pa.float64()


def test_net_income_margin():
    """net_income_margin = TTM net income / TTM revenue."""
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    # TTM net_income = 2000*4 = 8000; TTM revenue = 10000*4 = 40000 → 0.20
    assert abs(row["net_income_margin"] - 0.20) < 1e-6


def test_free_cash_flow_margin():
    """free_cash_flow_margin = (TTM op income - TTM capex) / TTM revenue."""
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    # TTM op income = 3000*4=12000; TTM capex = 600*4=2400; TTM rev = 40000
    # fcf_margin = (12000 - 2400) / 40000 = 0.24
    assert abs(row["free_cash_flow_margin"] - 0.24) < 1e-6


def test_capex_growth_yoy():
    """capex_growth_yoy = (TTM capex[t] / TTM capex[t-4q]) - 1; null when <8q."""
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    # current TTM capex = 2400; prior TTM capex = 2000 → 0.20
    assert abs(row["capex_growth_yoy"] - 0.20) < 1e-6

    # With only 4 quarters of history, capex_growth_yoy must be null (no prior TTM)
    df_4q = _compute_derived(_INCOME_8Q.tail(4), _BALANCE_8Q.tail(4))
    assert df_4q["capex_growth_yoy"].is_null().all()


def test_research_to_revenue_zero_when_no_rd():
    """research_to_revenue = 0.0 when rd_expense is 0 (R&D concept unavailable)."""
    income_no_rd = _INCOME_8Q.with_columns(pl.lit(0.0).alias("rd_expense"))
    df = _compute_derived(income_no_rd, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    assert row["research_to_revenue"] == pytest.approx(0.0)


def test_revenue_growth_accel():
    """revenue_growth_accel = current YoY growth minus prior quarter YoY growth."""
    # _INCOME_8Q has flat revenue (10000 all quarters), so yoy=0 → accel=0
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    assert q.row(0, named=True)["revenue_growth_accel"] == pytest.approx(0.0)

    # Varying revenue: accelerating growth in 2022
    income_varying = _INCOME_8Q.with_columns(
        pl.Series("revenue", [
            10_000.0, 10_000.0, 10_000.0, 10_000.0,   # 2021: flat prior year
            11_000.0, 12_000.0, 13_000.0, 15_000.0,   # 2022: Q1→Q4 accelerating
        ])
    )
    df2 = _compute_derived(income_varying, _BALANCE_8Q)
    # Q4 2022 yoy = (15000-10000)/10000 = 0.50; Q3 2022 yoy = (13000-10000)/10000 = 0.30
    # accel = 0.50 - 0.30 = 0.20
    q2 = df2.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    assert abs(q2.row(0, named=True)["revenue_growth_accel"] - 0.20) < 1e-6
