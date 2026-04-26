"""Tests for FRED-based robotics macro signals ingestion."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl


def _mock_fred_response(observations: list[dict]) -> MagicMock:
    """Build a fake FRED API response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"observations": observations}
    resp.raise_for_status.return_value = None
    return resp


def test_fred_series_constant_has_four_entries():
    from ingestion.robotics_signals_ingestion import _FRED_SERIES
    assert set(_FRED_SERIES) == {"NEWORDER", "CFNAI", "IPG3331S", "WPU114"}


def test_fetch_fred_series_schema():
    """fetch_fred_series returns DataFrame with date (Date) and value (Float64)."""
    from ingestion.robotics_signals_ingestion import fetch_fred_series

    obs = [
        {"date": "2025-01-01", "value": "100.0"},
        {"date": "2025-02-01", "value": "101.5"},
    ]
    with patch("ingestion.robotics_signals_ingestion.requests.get",
               return_value=_mock_fred_response(obs)):
        df = fetch_fred_series("NEWORDER")

    assert df.columns == ["date", "value"]
    assert df["date"].dtype == pl.Date
    assert df["value"].dtype == pl.Float64
    assert len(df) == 2
    assert df["date"][0] == date(2025, 1, 1)
    assert df["value"][1] == 101.5


def test_fetch_fred_series_handles_missing_dot():
    """FRED uses '.' for missing observations — must convert to null."""
    from ingestion.robotics_signals_ingestion import fetch_fred_series

    obs = [
        {"date": "2025-01-01", "value": "100.0"},
        {"date": "2025-02-01", "value": "."},
        {"date": "2025-03-01", "value": "102.0"},
    ]
    with patch("ingestion.robotics_signals_ingestion.requests.get",
               return_value=_mock_fred_response(obs)):
        df = fetch_fred_series("NEWORDER")

    assert df["value"][1] is None
    assert df["value"][0] == 100.0
    assert df["value"][2] == 102.0


def test_fetch_fred_series_failure_returns_empty():
    """A 5xx or network exception returns an empty DataFrame, no exception raised."""
    from ingestion.robotics_signals_ingestion import fetch_fred_series

    with patch("ingestion.robotics_signals_ingestion.requests.get",
               side_effect=Exception("boom")):
        df = fetch_fred_series("NEWORDER")

    assert df.is_empty()
    assert df.columns == ["date", "value"]
    assert df["date"].dtype == pl.Date
    assert df["value"].dtype == pl.Float64


def test_save_robotics_signals_writes_parquet(tmp_path: Path):
    """save_robotics_signals writes one parquet per series with snappy compression."""
    from ingestion.robotics_signals_ingestion import save_robotics_signals

    series_dfs = {
        "NEWORDER": pl.DataFrame({
            "date": [date(2025, 1, 1)],
            "value": [100.0],
        }, schema={"date": pl.Date, "value": pl.Float64}),
        "CFNAI": pl.DataFrame({
            "date": [date(2025, 1, 1)],
            "value": [-0.2],
        }, schema={"date": pl.Date, "value": pl.Float64}),
    }

    save_robotics_signals(tmp_path, series_dfs)

    for series_id in series_dfs:
        path = tmp_path / f"{series_id}.parquet"
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert loaded["date"].dtype == pl.Date
        assert loaded["value"].dtype == pl.Float64


def test_save_robotics_signals_skips_empty(tmp_path: Path):
    """save_robotics_signals does not write a parquet for an empty DataFrame."""
    from ingestion.robotics_signals_ingestion import save_robotics_signals

    save_robotics_signals(tmp_path, {
        "NEWORDER": pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64}),
    })
    assert not (tmp_path / "NEWORDER.parquet").exists()


def test_fetch_all_invokes_each_series_once():
    """fetch_all calls fetch_fred_series exactly once per id in _FRED_SERIES."""
    from ingestion.robotics_signals_ingestion import fetch_all, _FRED_SERIES

    populated = pl.DataFrame(
        {"date": [date(2025, 1, 1)], "value": [1.0]},
        schema={"date": pl.Date, "value": pl.Float64},
    )

    calls: list[str] = []

    def fake_fetch(series_id, observation_start="2010-01-01"):
        calls.append(series_id)
        return populated

    with patch("ingestion.robotics_signals_ingestion.fetch_fred_series",
               side_effect=fake_fetch), \
         patch("ingestion.robotics_signals_ingestion.time.sleep"):
        out = fetch_all()

    assert sorted(calls) == sorted(_FRED_SERIES)
    assert set(out) == set(_FRED_SERIES)
