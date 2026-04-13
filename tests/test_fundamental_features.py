import datetime
import pytest
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path


def _write_fundamentals_fixture(fund_dir: Path, ticker: str, quarters: list[dict]) -> None:
    """Write fundamental fixture Parquet for one ticker."""
    path = fund_dir / ticker / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(quarters).write_parquet(str(path))


def test_join_fundamentals_picks_most_recent_quarter_before_date(tmp_path):
    """join_fundamentals selects period_end <= date (backward asof join)."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   [datetime.date(2024, 5, 1), datetime.date(2024, 8, 1)],
        "close_price": [900.0, 950.0],
    })

    quarters = [
        {"ticker": "NVDA", "period_end": datetime.date(2024, 3, 31),
         "pe_ratio_trailing": 30.0, "price_to_sales": 10.0, "price_to_book": 5.0,
         "revenue_growth_yoy": 0.2, "gross_margin": 0.70, "operating_margin": 0.45,
         "capex_to_revenue": 0.10, "debt_to_equity": 0.30, "current_ratio": 2.0},
        {"ticker": "NVDA", "period_end": datetime.date(2024, 6, 30),
         "pe_ratio_trailing": 35.0, "price_to_sales": 12.0, "price_to_book": 6.0,
         "revenue_growth_yoy": 0.3, "gross_margin": 0.72, "operating_margin": 0.50,
         "capex_to_revenue": 0.12, "debt_to_equity": 0.25, "current_ratio": 2.1},
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
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA", "AMZN"],
        "date":   [datetime.date(2024, 1, 1), datetime.date(2024, 6, 1), datetime.date(2024, 6, 1)],
        "close_price": [500.0, 900.0, 180.0],
    })

    quarters_nvda = [
        {"ticker": "NVDA", "period_end": datetime.date(2023, 12, 31),
         "pe_ratio_trailing": None, "price_to_sales": None, "price_to_book": None,
         "revenue_growth_yoy": 0.1, "gross_margin": 0.65, "operating_margin": 0.4,
         "capex_to_revenue": 0.08, "debt_to_equity": 0.4, "current_ratio": 1.8},
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters_nvda)
    # No fundamentals for AMZN → AMZN rows get all-null fundamental cols

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    assert len(result) == 3
    # NVDA 2024-01-01 gets 2023-12-31 quarter
    nvda_jan = result.filter((pl.col("ticker") == "NVDA") & (pl.col("date") == datetime.date(2024, 1, 1)))
    assert nvda_jan["gross_margin"][0] == pytest.approx(0.65)
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
        {"ticker": "NVDA", "period_end": datetime.date(2024, 3, 31),
         "pe_ratio_trailing": 30.0, "price_to_sales": 10.0, "price_to_book": 5.0,
         "revenue_growth_yoy": 0.2, "gross_margin": 0.70, "operating_margin": 0.45,
         "capex_to_revenue": 0.10, "debt_to_equity": 0.30, "current_ratio": 2.0},
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters)

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    assert len(result) == 1
    assert result["gross_margin"][0] is None  # no quarter available before 2023-12-31
