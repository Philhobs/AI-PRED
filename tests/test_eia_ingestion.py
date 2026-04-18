"""Tests for EIA capacity + PJM queue ingestion."""
from __future__ import annotations
import io
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


def _mock_eia_response() -> dict:
    """Minimal valid EIA API v2 response."""
    return {
        "response": {
            "data": [
                {"period": "2025-01", "fueltypeid": "NUC", "fueltypeDescription": "Nuclear",
                 "capacity": 95.4, "capacity-units": "gigawatts"},
                {"period": "2025-01", "fueltypeid": "NG",  "fueltypeDescription": "Natural Gas",
                 "capacity": 618.5, "capacity-units": "gigawatts"},
                {"period": "2024-12", "fueltypeid": "NUC", "fueltypeDescription": "Nuclear",
                 "capacity": 95.2, "capacity-units": "gigawatts"},
                {"period": "2024-12", "fueltypeid": "NG",  "fueltypeDescription": "Natural Gas",
                 "capacity": 617.1, "capacity-units": "gigawatts"},
            ]
        }
    }


def _mock_pjm_excel() -> bytes:
    """Minimal PJM queue Excel file as bytes."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Queue Number", "Name", "Zone", "MW", "Status"])
    ws.append(["Q001", "Solar Farm A", "MAAC",  500,  "Active"])
    ws.append(["Q002", "Gas Plant B",  "AECO",  800,  "Active"])
    ws.append(["Q003", "Wind Farm C",  "SWVA",  300,  "Withdrawn"])
    ws.append(["Q004", "Nuclear D",    "ComEd", 1000, "Active"])  # not Virginia — should be excluded
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_eia_capacity_schema(tmp_path):
    """EIA ingestion writes parquet with correct columns."""
    from ingestion.eia_ingestion import fetch_eia_capacity, save_eia_capacity

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: _mock_eia_response(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_eia_capacity(api_key="test_key")

    assert set(df.columns) == {"date", "fuel_type", "capacity_gw"}
    assert df["capacity_gw"].dtype == pl.Float64
    assert len(df) == 4  # 2 periods × 2 fuel types


def test_eia_capacity_fuel_types(tmp_path):
    """EIA ingestion includes nuclear and natural_gas rows."""
    from ingestion.eia_ingestion import fetch_eia_capacity

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: _mock_eia_response(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_eia_capacity(api_key="test_key")

    assert "nuclear" in df["fuel_type"].to_list()
    assert "natural_gas" in df["fuel_type"].to_list()


def test_pjm_queue_schema(tmp_path):
    """PJM ingestion writes parquet with correct columns."""
    from ingestion.eia_ingestion import fetch_pjm_queue

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=_mock_pjm_excel(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_pjm_queue()

    assert set(df.columns) == {"date", "zone", "queue_backlog_gw", "project_count"}
    assert df["queue_backlog_gw"].dtype == pl.Float64
    assert df["project_count"].dtype == pl.Int32


def test_pjm_filters_virginia_zones_only(tmp_path):
    """PJM ingestion excludes non-Virginia zones and withdrawn projects."""
    from ingestion.eia_ingestion import fetch_pjm_queue

    with patch("ingestion.eia_ingestion.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=_mock_pjm_excel(),
        )
        mock_get.return_value.raise_for_status = lambda: None
        df = fetch_pjm_queue()

    # ComEd zone should be excluded; Withdrawn Q003 should be excluded
    # Active: Q001 (MAAC, 500MW) + Q002 (AECO, 800MW) = 1300 MW = 1.3 GW
    zones = df["zone"].to_list()
    assert "ComEd" not in zones
    assert "ALL_VIRGINIA" in zones
    virginia_row = df.filter(pl.col("zone") == "ALL_VIRGINIA")
    assert abs(float(virginia_row["queue_backlog_gw"][0]) - 1.3) < 0.01
