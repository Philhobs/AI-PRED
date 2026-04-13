import pytest
import duckdb
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone


def _write_minimal_ais_parquet(base_dir: Path):
    """Write a minimal AIS parquet for view testing."""
    path = base_dir / "ais" / "date=2024-01-15" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("timestamp", pa.timestamp("s", tz="UTC")),
        pa.field("mmsi", pa.string()),
        pa.field("corridor", pa.string()),
        pa.field("vessel_type", pa.int32()),
        pa.field("speed_knots", pa.float32()),
    ])
    table = pa.Table.from_pylist(
        [{"timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
          "mmsi": "123456789", "corridor": "taiwan_strait",
          "vessel_type": 70, "speed_knots": 10.5}],
        schema=schema,
    )
    pq.write_table(table, path, compression="snappy")


def test_get_db_connection_executes_query(tmp_path, monkeypatch):
    """get_db_connection returns a live DuckDB connection."""
    import storage
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test.duckdb"))

    from storage import get_db_connection
    con = get_db_connection()
    result = con.execute("SELECT 42 AS answer").fetchone()
    assert result[0] == 42
    con.close()


def test_create_views_allows_querying_ais_parquet(tmp_path):
    """create_views registers v_ais queryable against existing Parquet files."""
    _write_minimal_ais_parquet(tmp_path)
    con = duckdb.connect()

    from storage import create_views
    create_views(con, data_dir=tmp_path)

    result = con.execute("SELECT COUNT(*) FROM v_ais").fetchone()
    assert result[0] == 1
    con.close()


def test_create_views_succeeds_with_no_parquet_files(tmp_path):
    """create_views handles missing Parquet files gracefully (pipeline hasn't run yet)."""
    con = duckdb.connect()
    from storage import create_views
    create_views(con, data_dir=tmp_path)  # Should not raise
    con.close()
