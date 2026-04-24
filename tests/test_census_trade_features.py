import datetime
import polars as pl
import pytest
from pathlib import Path

from ingestion.census_trade_ingestion import _SCHEMA as _TRADE_SCHEMA


def _write_trade(census_trade_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = census_trade_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_TRADE_SCHEMA).write_parquet(out / "trade.parquet")


def _query_df(date: datetime.date) -> pl.DataFrame:
    return pl.DataFrame(
        {"ticker": ["NVDA"], "date": [date]},
        schema={"ticker": pl.Utf8, "date": pl.Date},
    )


_QUERY_DATE = datetime.date(2024, 4, 15)
_RUN_DATE = datetime.date(2024, 4, 1)


def _imp(hs: str, partner: str, year: int, month: int, value: float) -> dict:
    return {
        "date": _RUN_DATE, "direction": "import", "hs_code": hs,
        "partner_code": partner, "year": year, "month": month, "value_usd": value,
    }


def _exp(hs: str, partner: str, year: int, month: int, value: float) -> dict:
    return {
        "date": _RUN_DATE, "direction": "export", "hs_code": hs,
        "partner_code": partner, "year": year, "month": month, "value_usd": value,
    }


def test_semicon_import_value_most_recent_month(tmp_path):
    """semicon_import_value uses most recent month <= query date, not future months."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _imp("8541", "ALL", 2024, 2, 1_000_000_000.0),  # Feb HS8541: $1B
        _imp("8542", "ALL", 2024, 2, 2_000_000_000.0),  # Feb HS8542: $2B → Feb total $3B
        _imp("8541", "ALL", 2024, 3, 1_500_000_000.0),  # Mar HS8541: $1.5B (most recent <= Apr 15)
        _imp("8542", "ALL", 2024, 3, 2_500_000_000.0),  # Mar HS8542: $2.5B → Mar total $4B
        _imp("8541", "ALL", 2024, 5, 9_000_000_000.0),  # May: future, excluded
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    # Most recent month <= Apr 15 is March: $4B = 4000M
    assert df["semicon_import_value"][0] == pytest.approx(4000.0)


def test_semicon_import_momentum(tmp_path):
    """semicon_import_momentum = current month minus previous month value."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _imp("8541", "ALL", 2024, 2, 3_000_000_000.0),  # Feb: $3B = 3000M
        _imp("8541", "ALL", 2024, 3, 4_000_000_000.0),  # Mar: $4B = 4000M
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    # momentum = 4000M - 3000M = 1000M
    assert df["semicon_import_momentum"][0] == pytest.approx(1000.0)


def test_semicon_import_momentum_single_row_zero(tmp_path):
    """semicon_import_momentum = 0.0 when only one month available (no prior month)."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _imp("8541", "ALL", 2024, 3, 4_000_000_000.0),
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    assert df["semicon_import_momentum"][0] == pytest.approx(0.0)


def test_dc_equipment_import_value_uses_correct_hs_codes(tmp_path):
    """dc_equipment_import_value sums HS 8471+8473, not semiconductor codes 8541/8542."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _imp("8471", "ALL", 2024, 3, 500_000_000.0),    # $500M — should be included
        _imp("8473", "ALL", 2024, 3, 200_000_000.0),    # $200M — should be included → $700M total
        _imp("8541", "ALL", 2024, 3, 9_999_000_000.0),  # semiconductor — must NOT count here
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    assert df["dc_equipment_import_value"][0] == pytest.approx(700.0)  # $700M


def test_dc_equipment_import_momentum(tmp_path):
    """dc_equipment_import_momentum = current minus previous month value."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _imp("8471", "ALL", 2024, 2, 600_000_000.0),  # Feb: $600M
        _imp("8471", "ALL", 2024, 3, 700_000_000.0),  # Mar: $700M
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    # momentum = 700M - 600M = 100M
    assert df["dc_equipment_import_momentum"][0] == pytest.approx(100.0)


def test_china_semicon_export_share(tmp_path):
    """china_semicon_export_share = China exports / total exports, value in [0, 1]."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _exp("8541", "ALL", 2024, 3, 10_000_000_000.0),  # total: $10B
        _exp("8541", "5700", 2024, 3, 2_000_000_000.0),  # China: $2B → share = 0.2
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    assert df["china_semicon_export_share"][0] == pytest.approx(0.2)
    assert 0.0 <= df["china_semicon_export_share"][0] <= 1.0


def test_china_semicon_export_share_zero_when_no_data(tmp_path):
    """china_semicon_export_share = 0.0 when no export data exists."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    # Only import data — no exports in the parquet
    _write_trade(census_dir, [
        _imp("8541", "ALL", 2024, 3, 1_000_000_000.0),
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    assert df["china_semicon_export_share"][0] == pytest.approx(0.0)


def test_taiwan_semicon_import_share(tmp_path):
    """taiwan_semicon_import_share = Taiwan imports / total imports."""
    from processing.census_trade_features import join_census_trade_features

    census_dir = tmp_path / "census_trade"
    _write_trade(census_dir, [
        _imp("8541", "ALL", 2024, 3, 8_000_000_000.0),   # total: $8B
        _imp("8541", "5830", 2024, 3, 3_000_000_000.0),  # Taiwan: $3B → share = 3/8 = 0.375
    ])

    df = join_census_trade_features(_query_df(_QUERY_DATE), census_dir)
    assert df["taiwan_semicon_import_share"][0] == pytest.approx(0.375)


def test_join_adds_exactly_6_columns_and_missing_dir_zero_fill(tmp_path):
    """join adds exactly 6 columns; missing census_trade_dir zero-fills all 6."""
    from processing.census_trade_features import join_census_trade_features, CENSUS_TRADE_FEATURE_COLS

    census_dir = tmp_path / "does_not_exist"
    input_df = _query_df(_QUERY_DATE)
    result = join_census_trade_features(input_df, census_dir)

    added = [c for c in result.columns if c not in input_df.columns]
    assert sorted(added) == sorted(CENSUS_TRADE_FEATURE_COLS)
    assert len(added) == 6
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert result[col][0] == 0.0
