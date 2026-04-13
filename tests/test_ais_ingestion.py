import pytest
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime, timezone


def test_identify_corridor_taiwan_strait():
    """Vessel at 23°N, 120.5°E is inside Taiwan Strait bbox."""
    from ingestion.ais_ingestion import _identify_corridor
    assert _identify_corridor(23.0, 120.5) == "taiwan_strait"


def test_identify_corridor_anchorage():
    """Vessel at 61°N, -149°W is inside Anchorage bbox."""
    from ingestion.ais_ingestion import _identify_corridor
    assert _identify_corridor(61.0, -149.0) == "anchorage"


def test_identify_corridor_none_for_open_ocean():
    """Vessel at 0°N, 0°E is outside all defined corridors."""
    from ingestion.ais_ingestion import _identify_corridor
    assert _identify_corridor(0.0, 0.0) is None


def test_is_cargo_true_for_type_70():
    """AIS type code 70 (general cargo) → _is_cargo returns True."""
    from ingestion.ais_ingestion import _is_cargo
    msg = {"Message": {"ShipStaticData": {"Type": 70}}}
    assert _is_cargo(msg) is True


def test_is_cargo_true_for_tanker_type_80():
    """AIS type code 80 (tanker) → _is_cargo returns True."""
    from ingestion.ais_ingestion import _is_cargo
    msg = {"Message": {"ShipStaticData": {"Type": 80}}}
    assert _is_cargo(msg) is True


def test_is_cargo_false_for_passenger_type_60():
    """AIS type code 60 (passenger) with no PositionReport → _is_cargo False."""
    from ingestion.ais_ingestion import _is_cargo
    msg = {"Message": {"ShipStaticData": {"Type": 60}}}
    assert _is_cargo(msg) is False


def test_parse_message_valid_vessel_in_taiwan_strait():
    """Parse well-formed AIS message → corridor=taiwan_strait, all fields correct."""
    from ingestion.ais_ingestion import _parse_message
    msg = {
        "MetaData": {
            "MMSI": "123456789",
            "ShipName": "EVER GIVEN",
            "latitude": 23.0,
            "longitude": 120.5,
        },
        "Message": {
            "PositionReport": {"Sog": 10.5, "Cog": 180.0},
            "ShipStaticData": {
                "Type": 71,
                "ImoNumber": "9811000",
                "Destination": "KLAX",
                "Draught": 14.5,
            },
        },
    }
    result = _parse_message(msg)

    assert result is not None
    assert result["mmsi"] == "123456789"
    assert result["vessel_name"] == "EVER GIVEN"
    assert result["corridor"] == "taiwan_strait"
    assert result["speed_knots"] == 10.5
    assert result["latitude"] == 23.0
    assert result["draught"] == 14.5


def test_parse_message_returns_none_when_no_position():
    """Messages without lat/lon → None (not written to buffer)."""
    from ingestion.ais_ingestion import _parse_message
    msg = {"MetaData": {}, "Message": {}}
    assert _parse_message(msg) is None


def test_write_parquet_creates_file(tmp_path):
    """_write_parquet creates dated Parquet file with correct schema."""
    from ingestion.ais_ingestion import _write_parquet

    records = [
        {
            "timestamp": datetime.now(timezone.utc),
            "mmsi": "123456789",
            "imo": "9811000",
            "vessel_name": "TEST VESSEL",
            "vessel_type": 70,
            "latitude": 23.0,
            "longitude": 120.5,
            "speed_knots": 10.5,
            "course": 180.0,
            "destination": "KLAX",
            "draught": 14.5,
            "corridor": "taiwan_strait",
        }
    ]

    _write_parquet(records, tmp_path)

    parquet_files = list(tmp_path.glob("ais/date=*/data.parquet"))
    assert len(parquet_files) == 1
    table = pq.read_table(parquet_files[0])
    assert table.num_rows == 1
    assert table.column("vessel_name")[0].as_py() == "TEST VESSEL"
    assert table.column("corridor")[0].as_py() == "taiwan_strait"


def test_parse_message_equator_vessel_not_dropped():
    """Vessel at lat=0.0 must NOT be dropped (falsy zero is still valid position)."""
    from ingestion.ais_ingestion import _parse_message
    msg = {
        "MetaData": {"MMSI": "999", "ShipName": "GULF", "latitude": 0.0, "longitude": 5.5},
        "Message": {"PositionReport": {"Sog": 3.0, "Cog": 0.0}, "ShipStaticData": {"Type": 70}},
    }
    result = _parse_message(msg)
    assert result is not None
    assert result["latitude"] == 0.0


def test_parse_message_prime_meridian_vessel_not_dropped():
    """Vessel at lon=0.0 must NOT be dropped."""
    from ingestion.ais_ingestion import _parse_message
    msg = {
        "MetaData": {"MMSI": "888", "ShipName": "ATLANTIC", "latitude": 10.0, "longitude": 0.0},
        "Message": {"PositionReport": {"Sog": 5.0, "Cog": 90.0}, "ShipStaticData": {"Type": 70}},
    }
    result = _parse_message(msg)
    assert result is not None
    assert result["longitude"] == 0.0
