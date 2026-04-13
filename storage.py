import duckdb
from pathlib import Path

DB_PATH = str(Path(__file__).parent / "data" / "analytics.duckdb")


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Return DuckDB connection. No server required — in-process."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(DB_PATH)


def create_views(con: duckdb.DuckDBPyConnection, data_dir: Path = Path("data/raw")):
    """
    Register DuckDB views over Hive-partitioned Parquet files.
    Views are virtual — no data is copied.
    Skips views where no Parquet files exist yet.
    """
    views = {
        "v_ais": data_dir / "ais" / "date=*" / "data.parquet",
        "v_flights": data_dir / "flights" / "date=*" / "cargo.parquet",
        "v_news": data_dir / "news" / "scored" / "date=*" / "data.parquet",
        "v_news_rss": data_dir / "news" / "rss" / "date=*" / "data.parquet",
        "v_satellite": data_dir / "satellite" / "date=*" / "data.parquet",
        "v_financials_capex": data_dir / "financials" / "capex_history.parquet",
    }

    for view_name, path_glob in views.items():
        # Use data_dir.glob() with the relative pattern for reliable matching
        try:
            glob_pattern = str(path_glob.relative_to(data_dir))
        except ValueError:
            glob_pattern = str(path_glob)

        matches = list(data_dir.glob(glob_pattern))

        if not matches:
            print(f"[DB] Skipped view {view_name} (no data yet)")
            continue
        try:
            safe_path = str(path_glob).replace("'", "''")
            con.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT *
                FROM read_parquet('{safe_path}', hive_partitioning=True, union_by_name=True)
            """)
            print(f"[DB] Created view: {view_name}")
        except Exception as e:
            print(f"[DB] ERROR creating view {view_name}: {e}")


if __name__ == "__main__":
    with get_db_connection() as con:
        create_views(con)
        for view in ["v_ais", "v_flights", "v_news_rss", "v_financials_capex"]:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
                print(f"  {view}: {n} rows")
            except Exception:
                print(f"  {view}: not available")
