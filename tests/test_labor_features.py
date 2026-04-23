import datetime
import polars as pl
import pytest
from pathlib import Path

from ingestion.usajobs_ingestion import _SCHEMA as _POSTING_SCHEMA
from ingestion.bls_jolts_ingestion import _SCHEMA as _JOLTS_SCHEMA


def _write_postings(usajobs_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = usajobs_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_POSTING_SCHEMA).write_parquet(out / "postings.parquet")


def _write_jolts(jolts_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = jolts_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_JOLTS_SCHEMA).write_parquet(out / "openings.parquet")


def _query_df(date: datetime.date) -> pl.DataFrame:
    return pl.DataFrame(
        {"ticker": ["NVDA"], "date": [date]},
        schema={"ticker": pl.Utf8, "date": pl.Date},
    )


_QUERY_DATE = datetime.date(2024, 4, 1)
_RECENT = datetime.date(2024, 3, 20)    # 12 days ago — inside 30d window
_PRIOR = datetime.date(2024, 2, 20)     # 40 days ago — inside prior 30d window (31-60d)
_OLD = datetime.date(2024, 1, 1)        # 91 days ago — outside 60d window


def test_gov_ai_hiring_30d_window(tmp_path):
    """gov_ai_hiring_30d counts only postings within 30d, not older ones."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    _write_postings(usajobs_dir, [
        {"date": _QUERY_DATE, "posting_id": "P1", "title": "AI Engineer",
         "posted_date": _RECENT, "keyword": "artificial intelligence"},   # in window
        {"date": _QUERY_DATE, "posting_id": "P2", "title": "ML Scientist",
         "posted_date": _OLD, "keyword": "machine learning"},             # outside 60d
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["gov_ai_hiring_30d"][0] == 1.0


def test_gov_ai_hiring_momentum_positive(tmp_path):
    """gov_ai_hiring_momentum is positive when recent 30d count > prior 30d count."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # 2 recent, 1 prior
    _write_postings(usajobs_dir, [
        {"date": _QUERY_DATE, "posting_id": "P1", "title": "AI A",
         "posted_date": _RECENT, "keyword": "artificial intelligence"},
        {"date": _QUERY_DATE, "posting_id": "P2", "title": "AI B",
         "posted_date": _RECENT, "keyword": "machine learning"},
        {"date": _QUERY_DATE, "posting_id": "P3", "title": "AI C",
         "posted_date": _PRIOR, "keyword": "deep learning"},
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["gov_ai_hiring_momentum"][0] == pytest.approx(1.0)


def test_tech_job_openings_index_most_recent_month(tmp_path):
    """tech_job_openings_index uses the most recent JOLTS month <= query date."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # M02=100, M03=120 (both <= 2024-03-15), M04=150 (2024-04-01 > 2024-03-15, excluded)
    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M02", "value": 100.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M03", "value": 120.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M04", "value": 150.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M05", "value": 200.0},
    ])

    # Use query date 2024-03-15 so M04 (2024-04-01) is excluded
    df = join_labor_features(_query_df(datetime.date(2024, 3, 15)), usajobs_dir, jolts_dir)
    assert df["tech_job_openings_index"][0] == pytest.approx(120.0)


def test_tech_job_openings_momentum(tmp_path):
    """tech_job_openings_momentum equals current minus previous month value."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M01", "value": 90.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M02", "value": 100.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M03", "value": 120.0},
    ])

    # Query date 2024-04-01: most recent <= date is M03=120, previous is M02=100
    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["tech_job_openings_momentum"][0] == pytest.approx(20.0)


def test_tech_job_openings_momentum_single_row_zero(tmp_path):
    """tech_job_openings_momentum is 0.0 when only one JOLTS month is available."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # Only one month available — no prior month to compute momentum
    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M03", "value": 120.0},
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["tech_job_openings_index"][0] == pytest.approx(120.0)
    assert df["tech_job_openings_momentum"][0] == pytest.approx(0.0)


def test_join_adds_exactly_4_columns(tmp_path):
    """join_labor_features adds exactly 4 new columns."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"
    input_df = _query_df(_QUERY_DATE)
    result = join_labor_features(input_df, usajobs_dir, jolts_dir)

    added = [c for c in result.columns if c not in input_df.columns]
    assert sorted(added) == sorted(LABOR_FEATURE_COLS)
    assert len(added) == 4


def test_missing_usajobs_dir_zero_fill(tmp_path):
    """Missing usajobs_dir returns zero-filled USAJOBS features."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "does_not_exist" / "usajobs"
    jolts_dir = tmp_path / "does_not_exist" / "bls_jolts"

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    for col in LABOR_FEATURE_COLS:
        assert df[col][0] == 0.0


def test_missing_jolts_dir_zero_fill(tmp_path):
    """Missing jolts_dir returns zero-filled JOLTS features."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "does_not_exist" / "usajobs"
    jolts_dir = tmp_path / "does_not_exist" / "bls_jolts"

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    for col in LABOR_FEATURE_COLS:
        assert df[col][0] == 0.0


def test_no_data_in_window_zero_fill(tmp_path):
    """All 4 features zero-filled when postings and JOLTS data are outside windows."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # Posting is too old (outside 60d window)
    _write_postings(usajobs_dir, [
        {"date": _QUERY_DATE, "posting_id": "P1", "title": "Old Job",
         "posted_date": _OLD, "keyword": "semiconductor"},
    ])
    # JOLTS data is in the future (after query date)
    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M12", "value": 999.0},  # 2024-12-01 > query date
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["gov_ai_hiring_30d"][0] == 0.0
    assert df["gov_ai_hiring_momentum"][0] == 0.0
    assert df["tech_job_openings_index"][0] == 0.0
    assert df["tech_job_openings_momentum"][0] == 0.0
