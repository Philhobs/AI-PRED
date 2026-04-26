import datetime
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.census_trade_ingestion import _SCHEMA, _QUERIES


def _make_census_response(
    rows: list[list],
    value_field: str = "GEN_VAL_MO",
    commodity_field: str = "I_COMMODITY",
) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = [
        [value_field, commodity_field, "time"],
        *rows,
    ]
    return mock


_IMPORT_ROW = ["12345678.0", "8541", "2024-03"]
_EXPORT_ROW = ["9876543.0", "8541", "2024-03"]


def test_schema_correct():
    """fetch_trade returns a DataFrame matching _SCHEMA."""
    from ingestion.census_trade_ingestion import fetch_trade

    import_resp = _make_census_response([_IMPORT_ROW], "GEN_VAL_MO", "I_COMMODITY")
    export_resp = _make_census_response([_EXPORT_ROW], "ALL_VAL_MO", "E_COMMODITY")
    # 10 queries: 6 imports (indices 0-5), 4 exports (indices 6-9)
    responses = [import_resp] * 6 + [export_resp] * 4

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=responses):
        with patch("ingestion.census_trade_ingestion.time.sleep"):
            df = fetch_trade("2024-04-01")

    assert df.schema == _SCHEMA
    assert len(df) > 0


def test_direction_hs_partner_stored_correctly():
    """direction, hs_code, and partner_code are stored correctly."""
    from ingestion.census_trade_ingestion import fetch_trade

    import_resp = _make_census_response([["5000000.0", "8541", "2024-03"]], "GEN_VAL_MO", "I_COMMODITY")
    export_resp = _make_census_response([["3000000.0", "8541", "2024-03"]], "ALL_VAL_MO", "E_COMMODITY")
    responses = [import_resp] * 6 + [export_resp] * 4

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=responses):
        with patch("ingestion.census_trade_ingestion.time.sleep"):
            df = fetch_trade("2024-04-01")

    assert set(df["direction"].to_list()) >= {"import", "export"}
    # Taiwan import rows (queries 4 and 5 in _QUERIES are partner_code="5830")
    assert "5830" in df["partner_code"].to_list()
    # China export rows (queries 8 and 9 in _QUERIES are partner_code="5700")
    assert "5700" in df["partner_code"].to_list()


def test_same_month_snapshot_skipped(tmp_path):
    """ingest_census_trade skips re-download when existing snapshot is from the same calendar month."""
    from ingestion.census_trade_ingestion import ingest_census_trade

    existing = tmp_path / "date=2024-04-01"
    existing.mkdir()
    pl.DataFrame(schema=_SCHEMA).write_parquet(existing / "trade.parquet")

    with patch("ingestion.census_trade_ingestion.requests.get") as mock_get:
        ingest_census_trade("2024-04-15", tmp_path)

    mock_get.assert_not_called()


def test_empty_response_no_file_written(tmp_path):
    """When API returns no data rows, no parquet file is written."""
    from ingestion.census_trade_ingestion import ingest_census_trade

    import_empty = MagicMock()
    import_empty.raise_for_status.return_value = None
    import_empty.json.return_value = [["GEN_VAL_MO", "I_COMMODITY", "time"]]  # header only

    export_empty = MagicMock()
    export_empty.raise_for_status.return_value = None
    export_empty.json.return_value = [["ALL_VAL_MO", "E_COMMODITY", "time"]]  # header only

    responses = [import_empty] * 6 + [export_empty] * 4

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=responses):
        with patch("ingestion.census_trade_ingestion.time.sleep"):
            ingest_census_trade("2024-04-01", tmp_path)

    assert not (tmp_path / "date=2024-04-01").exists()


def test_sleep_between_queries():
    """time.sleep(1.0) is called between queries — not after the last one."""
    from ingestion.census_trade_ingestion import fetch_trade, _QUERIES

    import_resp = _make_census_response([_IMPORT_ROW], "GEN_VAL_MO", "I_COMMODITY")
    export_resp = _make_census_response([_EXPORT_ROW], "ALL_VAL_MO", "E_COMMODITY")
    responses = [import_resp] * 6 + [export_resp] * 4

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=responses):
        with patch("ingestion.census_trade_ingestion.time.sleep") as mock_sleep:
            fetch_trade("2024-04-01")

    assert mock_sleep.call_count == len(_QUERIES) - 1
    mock_sleep.assert_called_with(1.0)


def test_import_query_uses_i_commodity_field():
    """Import queries must use I_COMMODITY (not E_COMMODITY) — Census API requires direction-correct field."""
    from ingestion.census_trade_ingestion import _fetch_query

    captured_url = {"url": ""}

    def _capture(url, timeout):
        captured_url["url"] = url
        m = MagicMock()
        m.raise_for_status.return_value = None
        m.json.return_value = [["GEN_VAL_MO", "I_COMMODITY", "time"]]  # header only
        return m

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=_capture):
        _fetch_query("import", "8541", "ALL", datetime.date(2024, 4, 1), api_key="")

    assert "I_COMMODITY=8541" in captured_url["url"]
    assert "E_COMMODITY" not in captured_url["url"]


def test_export_query_uses_e_commodity_field():
    """Export queries continue to use E_COMMODITY."""
    from ingestion.census_trade_ingestion import _fetch_query

    captured_url = {"url": ""}

    def _capture(url, timeout):
        captured_url["url"] = url
        m = MagicMock()
        m.raise_for_status.return_value = None
        m.json.return_value = [["ALL_VAL_MO", "E_COMMODITY", "time"]]  # header only
        return m

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=_capture):
        _fetch_query("export", "8541", "ALL", datetime.date(2024, 4, 1), api_key="")

    assert "E_COMMODITY=8541" in captured_url["url"]
    assert "I_COMMODITY" not in captured_url["url"]


def test_fetch_query_fail_soft_on_http_error(caplog):
    """An exception on requests.get / json parse returns [] and logs a warning, no raise."""
    import logging
    from ingestion.census_trade_ingestion import _fetch_query

    with patch("ingestion.census_trade_ingestion.requests.get") as mock_get:
        mock_get.side_effect = Exception("503 Service Unavailable")
        with caplog.at_level(logging.WARNING, logger="ingestion.census_trade_ingestion"):
            result = _fetch_query("import", "8541", "ALL", datetime.date(2024, 4, 1), api_key="")

    assert result == []
    assert any("fetch failed" in rec.message for rec in caplog.records)


def test_fetch_query_fail_soft_on_dict_response(caplog):
    """A dict-shaped response (Census error shape) returns [] without crashing."""
    import logging
    from ingestion.census_trade_ingestion import _fetch_query

    err_resp = MagicMock()
    err_resp.raise_for_status.return_value = None
    err_resp.json.return_value = {"error": "invalid query"}  # dict, not list
    with patch("ingestion.census_trade_ingestion.requests.get", return_value=err_resp):
        with caplog.at_level(logging.WARNING, logger="ingestion.census_trade_ingestion"):
            result = _fetch_query("import", "8541", "ALL", datetime.date(2024, 4, 1), api_key="")

    assert result == []
    assert any("unexpected response shape" in rec.message for rec in caplog.records)
