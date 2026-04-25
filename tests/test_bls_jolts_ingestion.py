import datetime
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.bls_jolts_ingestion import _SCHEMA, _SERIES_IDS

_SERIES_ID = _SERIES_IDS[0]  # JTS510000000000000JOL — used by existing test helpers


def _make_jolts_response(data_rows: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [{"seriesID": _SERIES_ID, "data": data_rows}]
        },
    }
    return mock


_ONE_ROW = {"year": "2024", "period": "M03", "periodName": "March", "value": "123.4", "footnotes": [{}]}


def test_schema_correct():
    """fetch_jolts returns a DataFrame matching _SCHEMA."""
    from ingestion.bls_jolts_ingestion import fetch_jolts

    resp = _make_jolts_response([_ONE_ROW])
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp):
        df = fetch_jolts("2024-04-01")

    assert df.schema == _SCHEMA
    assert len(df) == 1
    assert df["year"][0] == 2024
    assert df["period"][0] == "M03"
    assert df["value"][0] == pytest.approx(123.4)


def test_period_stored_as_string():
    """period field is stored as a string 'M01'–'M12', not converted to a date."""
    from ingestion.bls_jolts_ingestion import fetch_jolts

    rows = [
        {"year": "2024", "period": "M01", "value": "100.0", "footnotes": []},
        {"year": "2024", "period": "M12", "value": "200.0", "footnotes": []},
    ]
    resp = _make_jolts_response(rows)
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp):
        df = fetch_jolts("2024-04-01")

    assert df["period"].dtype == pl.Utf8
    assert set(df["period"].to_list()) == {"M01", "M12"}


def test_same_month_snapshot_skipped(tmp_path):
    """ingest_bls_jolts skips re-download when existing snapshot is from the same calendar month."""
    from ingestion.bls_jolts_ingestion import ingest_bls_jolts

    # Both 2024-04-01 and 2024-04-15 are in April 2024
    existing = tmp_path / "date=2024-04-01"
    existing.mkdir()
    pl.DataFrame(schema=_SCHEMA).write_parquet(existing / "openings.parquet")

    with patch("ingestion.bls_jolts_ingestion.requests.post") as mock_post:
        ingest_bls_jolts("2024-04-15", tmp_path)

    mock_post.assert_not_called()


def test_different_month_snapshot_not_skipped(tmp_path):
    """ingest_bls_jolts re-downloads when existing snapshot is from a different calendar month."""
    from ingestion.bls_jolts_ingestion import ingest_bls_jolts

    # Existing snapshot from March 2024 — April 2024 run should NOT skip
    existing = tmp_path / "date=2024-03-15"
    existing.mkdir()
    pl.DataFrame(schema=_SCHEMA).write_parquet(existing / "openings.parquet")

    resp = _make_jolts_response([_ONE_ROW])
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp) as mock_post:
        ingest_bls_jolts("2024-04-01", tmp_path)

    mock_post.assert_called_once()


def test_empty_series_no_file_written(tmp_path):
    """When API returns no data rows, no parquet file is written."""
    from ingestion.bls_jolts_ingestion import ingest_bls_jolts

    resp = _make_jolts_response([])
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp):
        ingest_bls_jolts("2024-04-01", tmp_path)

    assert not (tmp_path / "date=2024-04-01").exists()


def test_series_ids_include_naics_333():
    """_SERIES_IDS must contain both NAICS 51 (Information) and NAICS 333 (Machinery)."""
    from ingestion.bls_jolts_ingestion import _SERIES_IDS
    assert set(_SERIES_IDS) == {
        "JTS510000000000000JOL",
        "JTS333000000000000JOL",
    }


def test_fetch_jolts_returns_rows_for_both_series():
    """fetch_jolts returns rows for both NAICS 51 and 333 when both come back from BLS."""
    from unittest.mock import patch
    from ingestion.bls_jolts_ingestion import fetch_jolts

    fake_response = {
        "Results": {
            "series": [
                {
                    "seriesID": "JTS510000000000000JOL",
                    "data": [
                        {"year": "2025", "period": "M01", "value": "100.0"},
                    ],
                },
                {
                    "seriesID": "JTS333000000000000JOL",
                    "data": [
                        {"year": "2025", "period": "M01", "value": "50.0"},
                    ],
                },
            ]
        }
    }
    fake_resp = type("R", (), {
        "json": lambda self: fake_response,
        "raise_for_status": lambda self: None,
    })()

    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=fake_resp):
        df = fetch_jolts("2025-02-15")

    assert df["series_id"].to_list().count("JTS510000000000000JOL") == 1
    assert df["series_id"].to_list().count("JTS333000000000000JOL") == 1
