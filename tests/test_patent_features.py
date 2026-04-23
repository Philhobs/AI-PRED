import datetime
import polars as pl
import pytest
from pathlib import Path


_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}


def _write_apps(apps_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = apps_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_APP_SCHEMA).write_parquet(out / "apps.parquet")


def _write_grants(grants_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = grants_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_GRANT_SCHEMA).write_parquet(out / "grants.parquet")


def _query_df(ticker: str, date: datetime.date) -> pl.DataFrame:
    return pl.DataFrame({"ticker": [ticker], "date": [date]},
                        schema={"ticker": pl.Utf8, "date": pl.Date})


# Query date used throughout tests
_QUERY_DATE = datetime.date(2024, 4, 1)
# 70 days before query → inside 90d window
_RECENT = datetime.date(2024, 1, 22)
# 100 days before query → inside prior 90d window (91–180d ago)
_PRIOR = datetime.date(2023, 12, 22)
# 400 days before query → outside 365d window
_OLD = datetime.date(2023, 2, 25)


def test_patent_app_count_90d_window(tmp_path):
    """patent_app_count_90d counts only apps filed within 90d, not older ones."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},   # in window
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A2", "cpc_group": "G06N", "filing_date": _OLD},      # outside 365d
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_app_count_90d"][0] == 1.0


def test_patent_app_momentum_positive(tmp_path):
    """patent_app_momentum is positive when recent 90d count > prior 90d count."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    # 2 apps in recent 90d, 1 app in prior 90d (91–180d window)
    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A2", "cpc_group": "H01L", "filing_date": _RECENT},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A3", "cpc_group": "G06N", "filing_date": _PRIOR},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_app_momentum"][0] > 0


def test_patent_grant_count_365d(tmp_path):
    """patent_grant_count_365d sums grants over full 365d window."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    grant_date_in = datetime.date(2023, 4, 15)   # 351 days ago — inside 365d
    grant_date_out = datetime.date(2023, 3, 15)  # 382 days ago — outside 365d

    _write_grants(grants_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P1", "cpc_group": "G06N", "grant_date": grant_date_in,
         "forward_citation_count": 0},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P2", "cpc_group": "G06N", "grant_date": grant_date_out,
         "forward_citation_count": 0},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_grant_count_365d"][0] == 1.0


def test_patent_grant_rate_zero_safe(tmp_path):
    """patent_grant_rate_365d uses GREATEST(apps, 1) denominator — no division by zero."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    # 1 grant, 0 apps → rate = 1/1 = 1.0 (not Inf)
    _write_grants(grants_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P1", "cpc_group": "G06N",
         "grant_date": datetime.date(2024, 1, 1), "forward_citation_count": 0},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_grant_rate_365d"][0] == pytest.approx(1.0)


def test_patent_ai_cpc_share_isolates_g06n(tmp_path):
    """patent_ai_cpc_share_90d counts only G06N, excludes H01L/G06F/G11C."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},   # AI
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A2", "cpc_group": "H01L", "filing_date": _RECENT},   # not AI
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A3", "cpc_group": "G06F", "filing_date": _RECENT},   # not AI
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_ai_cpc_share_90d"][0] == pytest.approx(1.0 / 3.0)


def test_patent_citation_count_365d(tmp_path):
    """patent_citation_count_365d sums forward citations within 365d grant window."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    _write_grants(grants_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P1", "cpc_group": "G06N",
         "grant_date": datetime.date(2023, 12, 1), "forward_citation_count": 5},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P2", "cpc_group": "H01L",
         "grant_date": datetime.date(2024, 1, 15), "forward_citation_count": 3},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_citation_count_365d"][0] == 8.0


def test_join_adds_exactly_6_columns(tmp_path):
    """join_patent_features adds exactly 6 new columns to df."""
    from processing.patent_features import join_patent_features, USPTO_PATENT_FEATURE_COLS

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"
    input_df = _query_df("NVDA", _QUERY_DATE)
    result = join_patent_features(input_df, apps_dir, grants_dir)

    added = [c for c in result.columns if c not in input_df.columns]
    assert sorted(added) == sorted(USPTO_PATENT_FEATURE_COLS)
    assert len(added) == 6


def test_ticker_no_patents_zero_fill(tmp_path):
    """Ticker with no matching patents gets zero-filled for all 6 features."""
    from processing.patent_features import join_patent_features, USPTO_PATENT_FEATURE_COLS

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    # Write data for a different company — NVDA gets no match
    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "Some Unrelated Corp",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    for col in USPTO_PATENT_FEATURE_COLS:
        assert df[col][0] == 0.0, f"{col} should be 0.0 when no patents match"


def test_missing_apps_dir_zero_fill(tmp_path):
    """Missing apps_dir returns zero-filled features."""
    from processing.patent_features import join_patent_features, USPTO_PATENT_FEATURE_COLS

    apps_dir = tmp_path / "does_not_exist" / "applications"
    grants_dir = tmp_path / "does_not_exist" / "grants"

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    for col in USPTO_PATENT_FEATURE_COLS:
        assert df[col][0] == 0.0


def test_uspto_col_names_correct():
    """USPTO_PATENT_FEATURE_COLS contains exactly the 6 expected names."""
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    expected = {
        "patent_app_count_90d",
        "patent_app_momentum",
        "patent_grant_count_365d",
        "patent_grant_rate_365d",
        "patent_ai_cpc_share_90d",
        "patent_citation_count_365d",
    }
    assert set(USPTO_PATENT_FEATURE_COLS) == expected
