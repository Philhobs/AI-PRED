"""Tests for fundamental_features.py — covers both the original 9 and 5 new columns."""
import datetime
import polars as pl
import pytest
from pathlib import Path


def _write_fundamentals_fixture(fund_dir: Path, ticker: str, quarters: list[dict]) -> None:
    """Write fundamental fixture Parquet for one ticker."""
    path = fund_dir / ticker / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(quarters).write_parquet(str(path))


def _make_full_quarter(ticker: str, period_end: datetime.date, **overrides) -> dict:
    """Return a dict with all 14 fundamental columns (+ ticker + period_end + available_date).

    By default available_date = period_end + 30 days (a typical 10-Q filing
    cadence). Override per-test when the lag matters.
    """
    base = {
        "ticker": ticker,
        "period_end": period_end,
        "available_date": period_end + datetime.timedelta(days=30),
        "pe_ratio_trailing": 25.0, "price_to_sales": 8.0, "price_to_book": 3.0,
        "revenue_growth_yoy": 0.15, "gross_margin": 0.60, "operating_margin": 0.25,
        "capex_to_revenue": 0.08, "debt_to_equity": 0.5, "current_ratio": 1.8,
        "net_income_margin": 0.20, "free_cash_flow_margin": 0.15,
        "capex_growth_yoy": 0.10, "revenue_growth_accel": 0.02,
        "research_to_revenue": 0.12,
    }
    base.update(overrides)
    return base


# ── Original tests (updated to use FUNDAMENTAL_FEATURE_COLS) ─────────────────

def test_join_fundamentals_picks_most_recent_quarter_before_date(tmp_path):
    """join_fundamentals selects period_end <= date (backward asof join)."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   [datetime.date(2024, 5, 1), datetime.date(2024, 8, 1)],
        "close_price": [900.0, 950.0],
    })

    quarters = [
        _make_full_quarter("NVDA", datetime.date(2024, 3, 31),
                           pe_ratio_trailing=30.0, price_to_sales=10.0, price_to_book=5.0,
                           revenue_growth_yoy=0.2, gross_margin=0.70, operating_margin=0.45,
                           capex_to_revenue=0.10, debt_to_equity=0.30, current_ratio=2.0),
        _make_full_quarter("NVDA", datetime.date(2024, 6, 30),
                           pe_ratio_trailing=35.0, price_to_sales=12.0, price_to_book=6.0,
                           revenue_growth_yoy=0.3, gross_margin=0.72, operating_margin=0.50,
                           capex_to_revenue=0.12, debt_to_equity=0.25, current_ratio=2.1),
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters)

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    # 2024-05-01: only 2024-03-31 is available (2024-06-30 is in the future)
    row_may = result.filter(pl.col("date") == datetime.date(2024, 5, 1))
    assert row_may["pe_ratio_trailing"][0] == pytest.approx(30.0)
    assert row_may["gross_margin"][0] == pytest.approx(0.70)

    # 2024-08-01: 2024-06-30 is now the most recent
    row_aug = result.filter(pl.col("date") == datetime.date(2024, 8, 1))
    assert row_aug["pe_ratio_trailing"][0] == pytest.approx(35.0)
    assert row_aug["gross_margin"][0] == pytest.approx(0.72)

    # Output columns must be exactly: price columns + 14 fundamental columns (no period_end leak)
    from processing.fundamental_features import FUNDAMENTAL_FEATURE_COLS
    expected_cols = {"ticker", "date", "close_price"} | set(FUNDAMENTAL_FEATURE_COLS)
    assert set(result.columns) == expected_cols


def test_join_fundamentals_returns_null_columns_when_no_data(tmp_path):
    """When fundamentals directory is empty, all fundamental columns are null."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date":   [datetime.date(2024, 5, 1)],
        "close_price": [900.0],
    })

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=empty_dir)

    assert "gross_margin" in result.columns
    assert result["gross_margin"][0] is None


def test_join_fundamentals_preserves_all_price_rows(tmp_path):
    """join_fundamentals returns same number of rows as input price_df."""
    # Spine dates land AFTER Q4-2023's available_date (= 2024-01-30 by the
    # default lag in _make_full_quarter), so the 10-Q is publicly known.
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA", "AMZN"],
        "date":   [datetime.date(2024, 2, 15), datetime.date(2024, 6, 1), datetime.date(2024, 6, 1)],
        "close_price": [500.0, 900.0, 180.0],
    })

    quarters_nvda = [
        _make_full_quarter("NVDA", datetime.date(2023, 12, 31),
                           pe_ratio_trailing=None, price_to_sales=None, price_to_book=None,
                           revenue_growth_yoy=0.1, gross_margin=0.65, operating_margin=0.4,
                           capex_to_revenue=0.08, debt_to_equity=0.4, current_ratio=1.8),
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters_nvda)
    # No fundamentals for AMZN → AMZN rows get all-null fundamental cols

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    assert len(result) == 3
    # NVDA 2024-02-15 (after Q4-2023 10-Q filed) gets the Q4-2023 quarter
    nvda_feb = result.filter((pl.col("ticker") == "NVDA") & (pl.col("date") == datetime.date(2024, 2, 15)))
    assert nvda_feb["gross_margin"][0] == pytest.approx(0.65)
    # Confirm no cross-ticker contamination: NVDA has data, AMZN does not
    nvda_jun = result.filter((pl.col("ticker") == "NVDA") & (pl.col("date") == datetime.date(2024, 6, 1)))
    assert nvda_jun["gross_margin"][0] == pytest.approx(0.65)  # same quarter, still matched
    # AMZN has no fundamentals → null
    amzn = result.filter(pl.col("ticker") == "AMZN")
    assert amzn["gross_margin"][0] is None


def test_join_fundamentals_returns_null_when_date_before_earliest_quarter(tmp_path):
    """When price date precedes all available fundamentals, result is null (no backward match)."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date":   [datetime.date(2023, 12, 31)],  # before 2024-03-31 quarter
        "close_price": [500.0],
    })

    quarters = [
        _make_full_quarter("NVDA", datetime.date(2024, 3, 31),
                           pe_ratio_trailing=30.0, price_to_sales=10.0, price_to_book=5.0,
                           revenue_growth_yoy=0.2, gross_margin=0.70, operating_margin=0.45,
                           capex_to_revenue=0.10, debt_to_equity=0.30, current_ratio=2.0),
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters)

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    assert len(result) == 1
    assert result["gross_margin"][0] is None  # no quarter available before 2023-12-31


# ── New tests for the 5 additional columns ───────────────────────────────────

def test_net_income_margin_asof_picks_most_recent_past_quarter(tmp_path):
    """net_income_margin backward asof join picks most recently AVAILABLE quarter."""
    from processing.fundamental_features import join_fundamentals
    _write_fundamentals_fixture(tmp_path, "MSFT", [
        _make_full_quarter("MSFT", datetime.date(2022, 9, 30), net_income_margin=0.30),
        _make_full_quarter("MSFT", datetime.date(2022, 12, 31), net_income_margin=0.35),
        # Future quarter — must NOT be picked for a date before its available_date
        _make_full_quarter("MSFT", datetime.date(2023, 3, 31), net_income_margin=0.40),
    ])
    # Q4-2022 available_date = 2023-01-30 (period_end + 30d). Spine 2023-02-15
    # is after Q4-2022 availability but before Q1-2023 availability (2023-04-30).
    price_df = pl.DataFrame({
        "ticker": ["MSFT"],
        "date": pl.Series([datetime.date(2023, 2, 15)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["net_income_margin"][0] == pytest.approx(0.35)


def test_free_cash_flow_margin_via_asof_join(tmp_path):
    """free_cash_flow_margin is correctly joined backward by date."""
    from processing.fundamental_features import join_fundamentals
    _write_fundamentals_fixture(tmp_path, "NVDA", [
        _make_full_quarter("NVDA", datetime.date(2022, 12, 31), free_cash_flow_margin=0.25),
    ])
    price_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": pl.Series([datetime.date(2023, 2, 1)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["free_cash_flow_margin"][0] == pytest.approx(0.25)


def test_capex_growth_yoy_positive_when_accelerating(tmp_path):
    """capex_growth_yoy reflects positive growth from the most recent quarter."""
    from processing.fundamental_features import join_fundamentals
    _write_fundamentals_fixture(tmp_path, "AMD", [
        _make_full_quarter("AMD", datetime.date(2022, 12, 31), capex_growth_yoy=0.20),
    ])
    price_df = pl.DataFrame({
        "ticker": ["AMD"],
        "date": pl.Series([datetime.date(2023, 3, 1)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["capex_growth_yoy"][0] == pytest.approx(0.20)


def test_revenue_growth_accel_second_derivative(tmp_path):
    """revenue_growth_accel = current YoY growth minus prior quarter YoY growth."""
    from processing.fundamental_features import join_fundamentals
    _write_fundamentals_fixture(tmp_path, "GOOGL", [
        _make_full_quarter("GOOGL", datetime.date(2022, 9, 30),  revenue_growth_accel=0.0),
        _make_full_quarter("GOOGL", datetime.date(2022, 12, 31), revenue_growth_accel=0.05),
    ])
    # Q4-2022 available_date = 2023-01-30; spine 2023-02-15 is after.
    price_df = pl.DataFrame({
        "ticker": ["GOOGL"],
        "date": pl.Series([datetime.date(2023, 2, 15)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["revenue_growth_accel"][0] == pytest.approx(0.05)


def test_missing_fundamentals_dir_all_14_cols_present(tmp_path):
    """When fundamentals directory is missing, all 14 columns are present in output."""
    from processing.fundamental_features import join_fundamentals, FUNDAMENTAL_FEATURE_COLS
    price_df = pl.DataFrame({
        "ticker": ["MSFT"],
        "date": pl.Series([datetime.date(2023, 1, 1)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path / "nonexistent")
    assert len(FUNDAMENTAL_FEATURE_COLS) == 14
    for col in FUNDAMENTAL_FEATURE_COLS:
        assert col in result.columns, f"{col} missing from result"
    assert len(result) == 1


# ── Point-in-time correctness (architecture v2 / Phase A.2) ───────────────────

def test_join_fundamentals_no_lookahead_pre_filing(tmp_path):
    """Spine date BEFORE available_date must NOT see the quarter — period_end alone is lookahead."""
    from processing.fundamental_features import join_fundamentals
    # Q4 ends 2023-12-31 but the 10-Q isn't filed until 2024-02-08. Spine 2024-01-15
    # is after period_end (so the OLD code would join it) but before the filing
    # date (so the new code correctly returns null).
    _write_fundamentals_fixture(tmp_path, "NVDA", [
        _make_full_quarter("NVDA", datetime.date(2023, 12, 31),
                           available_date=datetime.date(2024, 2, 8),
                           gross_margin=0.65),
    ])
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date": pl.Series([datetime.date(2024, 1, 15),   # PRE-filing → null
                            datetime.date(2024, 2, 15)],   # POST-filing → 0.65
                          dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path).sort("date")
    pre  = result.filter(pl.col("date") == datetime.date(2024, 1, 15))
    post = result.filter(pl.col("date") == datetime.date(2024, 2, 15))
    assert pre["gross_margin"][0] is None, "lookahead bug: pre-filing spine should be null"
    assert post["gross_margin"][0] == pytest.approx(0.65)


def test_join_fundamentals_legacy_parquet_fallback(tmp_path):
    """Parquet without available_date column falls back to period_end + 45 days."""
    from processing.fundamental_features import join_fundamentals, FUNDAMENTAL_FEATURE_COLS
    # Build a fixture WITHOUT available_date (simulates pre-A.2 ingestion output).
    legacy_row = {
        "ticker": "AMD",
        "period_end": datetime.date(2023, 12, 31),
        "pe_ratio_trailing": 20.0, "price_to_sales": 5.0, "price_to_book": 2.0,
        "revenue_growth_yoy": 0.10, "gross_margin": 0.50, "operating_margin": 0.20,
        "capex_to_revenue": 0.05, "debt_to_equity": 0.2, "current_ratio": 1.5,
        "net_income_margin": 0.15, "free_cash_flow_margin": 0.10,
        "capex_growth_yoy": 0.0, "revenue_growth_accel": 0.0,
        "research_to_revenue": 0.08,
    }
    path = tmp_path / "AMD" / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([legacy_row]).write_parquet(str(path))

    # Fallback available_date = 2023-12-31 + 45d = 2024-02-14
    price_df = pl.DataFrame({
        "ticker": ["AMD", "AMD"],
        "date": pl.Series([datetime.date(2024, 2, 10),   # PRE-fallback → null
                            datetime.date(2024, 2, 20)],   # POST-fallback → matched
                          dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path).sort("date")
    pre  = result.filter(pl.col("date") == datetime.date(2024, 2, 10))
    post = result.filter(pl.col("date") == datetime.date(2024, 2, 20))
    assert pre["gross_margin"][0] is None
    assert post["gross_margin"][0] == pytest.approx(0.50)
