# Census Trade Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add US Census international trade data (semiconductor + DC equipment imports/exports) as 6 market-wide features, growing FEATURE_COLS from 77 → 83.

**Architecture:** One ingestion module makes 10 targeted Census API queries per run (4 HS codes × import/export directions + 2 country-filtered queries), writes a normalized Hive-partitioned parquet, and one feature module derives 6 features via DuckDB window functions. All 6 features are market-wide (date-only join), medium + long tiers only.

**Tech Stack:** Python 3.11+, Polars, DuckDB, requests, pytest with unittest.mock.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ingestion/census_trade_ingestion.py` | Create | Census API client, 10 queries, staleness guard, parquet output |
| `processing/census_trade_features.py` | Create | 6 `CENSUS_TRADE_FEATURE_COLS` + `join_census_trade_features()` |
| `tests/test_census_trade_ingestion.py` | Create | 5 ingestion tests |
| `tests/test_census_trade_features.py` | Create | 9 feature tests |
| `models/train.py` | Modify | Import + FEATURE_COLS 77→83 + tier update + join call |
| `models/inference.py` | Modify | Import + join call |
| `tests/test_train.py` | Modify | Update count 77→83 + 5 new CENSUS tests |
| `tools/run_refresh.sh` | Modify | Renumber X/14 → X/15, add step 15/15 |

---

## Task 1: Census Trade Ingestion

**Files:**
- Create: `ingestion/census_trade_ingestion.py`
- Create: `tests/test_census_trade_ingestion.py`

- [ ] **Step 1: Write the 5 failing tests**

Create `tests/test_census_trade_ingestion.py`:

```python
import datetime
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.census_trade_ingestion import _SCHEMA, _QUERIES


def _make_census_response(rows: list[list], value_field: str = "GEN_VAL_MO") -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = [
        [value_field, "E_COMMODITY", "time"],
        *rows,
    ]
    return mock


_IMPORT_ROW = ["12345678.0", "8541", "2024-03"]
_EXPORT_ROW = ["9876543.0", "8541", "2024-03"]


def test_schema_correct():
    """fetch_trade returns a DataFrame matching _SCHEMA."""
    from ingestion.census_trade_ingestion import fetch_trade

    import_resp = _make_census_response([_IMPORT_ROW], "GEN_VAL_MO")
    export_resp = _make_census_response([_EXPORT_ROW], "ALL_VAL_MO")
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

    import_resp = _make_census_response([["5000000.0", "8541", "2024-03"]], "GEN_VAL_MO")
    export_resp = _make_census_response([["3000000.0", "8541", "2024-03"]], "ALL_VAL_MO")
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

    empty_resp = MagicMock()
    empty_resp.raise_for_status.return_value = None
    empty_resp.json.return_value = [["GEN_VAL_MO", "E_COMMODITY", "time"]]  # header only

    with patch("ingestion.census_trade_ingestion.requests.get", return_value=empty_resp):
        with patch("ingestion.census_trade_ingestion.time.sleep"):
            ingest_census_trade("2024-04-01", tmp_path)

    assert not (tmp_path / "date=2024-04-01").exists()


def test_sleep_between_queries():
    """time.sleep(0.5) is called between queries — not after the last one."""
    from ingestion.census_trade_ingestion import fetch_trade, _QUERIES

    import_resp = _make_census_response([_IMPORT_ROW], "GEN_VAL_MO")
    export_resp = _make_census_response([_EXPORT_ROW], "ALL_VAL_MO")
    responses = [import_resp] * 6 + [export_resp] * 4

    with patch("ingestion.census_trade_ingestion.requests.get", side_effect=responses):
        with patch("ingestion.census_trade_ingestion.time.sleep") as mock_sleep:
            fetch_trade("2024-04-01")

    assert mock_sleep.call_count == len(_QUERIES) - 1
    mock_sleep.assert_called_with(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_census_trade_ingestion.py -v 2>&1 | head -20
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'ingestion.census_trade_ingestion'`

- [ ] **Step 3: Implement `ingestion/census_trade_ingestion.py`**

Create `ingestion/census_trade_ingestion.py`:

```python
"""Census international trade data ingestion.

Fetches US semiconductor and data center equipment import/export values
from the Census International Trade API (timeseries/intltrade).

Queries (10 per run):
  Imports, all partners: HS 8541, 8542, 8471, 8473
  Imports, Taiwan (CTY=5830): HS 8541, 8542
  Exports, all partners: HS 8541, 8542
  Exports, China (CTY=5700): HS 8541, 8542

Output: data/raw/census_trade/date=YYYY-MM-DD/trade.parquet

Staleness guard: skips re-download if existing snapshot is from the same calendar month.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_IMPORTS_URL = "https://api.census.gov/data/timeseries/intltrade/imports"
_EXPORTS_URL = "https://api.census.gov/data/timeseries/intltrade/exports"

_SCHEMA = {
    "date": pl.Date,
    "direction": pl.Utf8,
    "hs_code": pl.Utf8,
    "partner_code": pl.Utf8,
    "year": pl.Int32,
    "month": pl.Int32,
    "value_usd": pl.Float64,
}

_SLEEP_BETWEEN_QUERIES = 0.5

# (direction, hs_code, partner_code) — "ALL" means omit CTY_CODE parameter
_QUERIES: list[tuple[str, str, str]] = [
    ("import", "8541", "ALL"),
    ("import", "8542", "ALL"),
    ("import", "8471", "ALL"),
    ("import", "8473", "ALL"),
    ("import", "8541", "5830"),  # Taiwan semiconductor imports
    ("import", "8542", "5830"),  # Taiwan integrated circuit imports
    ("export", "8541", "ALL"),
    ("export", "8542", "ALL"),
    ("export", "8541", "5700"),  # China semiconductor exports
    ("export", "8542", "5700"),  # China integrated circuit exports
]


def _same_month(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent parquet snapshot was taken in the same calendar month as today."""
    files = sorted(existing_dir.glob("date=*/trade.parquet"))
    if not files:
        return False
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        return last_date.year == today.year and last_date.month == today.month
    except ValueError:
        return False


def _fetch_query(
    direction: str,
    hs_code: str,
    partner_code: str,
    run_date: datetime.date,
    api_key: str,
) -> list[dict]:
    """Fetch 12-month lookback for one (direction, hs_code, partner_code) combination."""
    url = _IMPORTS_URL if direction == "import" else _EXPORTS_URL
    value_field = "GEN_VAL_MO" if direction == "import" else "ALL_VAL_MO"

    from_str = f"{run_date.year - 1}-{run_date.month:02d}"
    to_str = f"{run_date.year}-{run_date.month:02d}"

    params: dict = {
        "get": f"{value_field},E_COMMODITY",
        "COMM_LVL": "HS4",
        "E_COMMODITY": hs_code,
        "time": f"from+{from_str}+to+{to_str}",
    }
    if partner_code != "ALL":
        params["CTY_CODE"] = partner_code
    if api_key:
        params["key"] = api_key

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data or len(data) < 2:
        return []

    headers = data[0]
    try:
        val_idx = headers.index(value_field)
        time_idx = headers.index("time")
    except ValueError:
        _LOG.warning(
            "Census: unexpected headers %s for %s %s %s", headers, direction, hs_code, partner_code
        )
        return []

    rows = []
    for row in data[1:]:
        try:
            time_str = row[time_idx]   # "YYYY-MM"
            year = int(time_str[:4])
            month = int(time_str[5:7])
            value = float(row[val_idx])
        except (ValueError, TypeError, IndexError):
            continue
        rows.append({
            "date": run_date,
            "direction": direction,
            "hs_code": hs_code,
            "partner_code": partner_code,
            "year": year,
            "month": month,
            "value_usd": value,
        })

    return rows


def fetch_trade(date_str: str) -> pl.DataFrame:
    """Fetch all Census trade queries for date_str.

    Returns DataFrame with _SCHEMA. Empty DataFrame if no results.
    """
    run_date = datetime.date.fromisoformat(date_str)
    api_key = os.environ.get("CENSUS_API_KEY", "")

    all_rows: list[dict] = []
    for i, (direction, hs_code, partner_code) in enumerate(_QUERIES):
        all_rows.extend(_fetch_query(direction, hs_code, partner_code, run_date, api_key))
        if i < len(_QUERIES) - 1:
            time.sleep(_SLEEP_BETWEEN_QUERIES)

    if not all_rows:
        return pl.DataFrame(schema=_SCHEMA)

    return pl.DataFrame(all_rows, schema=_SCHEMA)


def ingest_census_trade(date_str: str, output_dir: Path) -> None:
    """Fetch and persist Census trade data for date_str.

    Skips download if same-month snapshot exists. Writes nothing when results are empty.
    """
    if _same_month(output_dir, date_str):
        _LOG.info("Census trade: same month snapshot exists — skipping for %s", date_str)
        return

    _LOG.info("Census trade: fetching semiconductor + DC equipment trade data for %s", date_str)
    df = fetch_trade(date_str)
    if df.is_empty():
        _LOG.info("Census trade: API returned no data rows for %s — nothing written", date_str)
    else:
        out = output_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "trade.parquet", compression="snappy")
        _LOG.info("Census trade: wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ingest_census_trade(
        datetime.date.today().isoformat(),
        Path("data/raw/census_trade"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_census_trade_ingestion.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add ingestion/census_trade_ingestion.py tests/test_census_trade_ingestion.py
git commit -m "$(cat <<'EOF'
feat: add Census international trade ingestion (semiconductors + DC equipment)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Census Trade Features + Model Integration

**Files:**
- Create: `processing/census_trade_features.py`
- Create: `tests/test_census_trade_features.py`
- Modify: `models/train.py`
- Modify: `models/inference.py`
- Modify: `tests/test_train.py`
- Modify: `tools/run_refresh.sh`

- [ ] **Step 1: Write the 9 failing feature tests**

Create `tests/test_census_trade_features.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_census_trade_features.py -v 2>&1 | head -20
```

Expected: 9 failures with `ModuleNotFoundError: No module named 'processing.census_trade_features'`

- [ ] **Step 3: Implement `processing/census_trade_features.py`**

Create `processing/census_trade_features.py`:

```python
"""Census international trade signal features.

Features (CENSUS_TRADE_FEATURE_COLS):
    semicon_import_value         — US semiconductor import value, most recent month ≤ query date (USD millions)
    semicon_import_momentum      — MoM change in semiconductor import value (USD millions)
    dc_equipment_import_value    — Data center equipment import value, most recent month (USD millions)
    dc_equipment_import_momentum — MoM change in DC equipment import value (USD millions)
    china_semicon_export_share   — US semiconductor exports to China / total (0–1 ratio)
    taiwan_semicon_import_share  — Taiwan's share of US semiconductor imports (0–1 ratio)

All 6 features are market-wide (joined on date only).
All features zero-filled when data is absent.

Tier routing: medium + long only (monthly data too slow for 5d/20d horizons).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

from ingestion.census_trade_ingestion import _SCHEMA as _TRADE_SCHEMA

_LOG = logging.getLogger(__name__)

CENSUS_TRADE_FEATURE_COLS: list[str] = [
    "semicon_import_value",
    "semicon_import_momentum",
    "dc_equipment_import_value",
    "dc_equipment_import_momentum",
    "china_semicon_export_share",
    "taiwan_semicon_import_share",
]


def _load_trade(census_trade_dir: Path) -> pl.DataFrame:
    files = sorted(census_trade_dir.glob("date=*/trade.parquet")) if census_trade_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_TRADE_SCHEMA)
    # Dedup: keep most recent snapshot's value for each (direction, hs_code, partner_code, year, month)
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .sort("date")
        .unique(subset=["direction", "hs_code", "partner_code", "year", "month"], keep="last")
    )


def join_census_trade_features(
    df: pl.DataFrame,
    census_trade_dir: Path,
) -> pl.DataFrame:
    """Left-join Census trade features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'date' (Date) column.
        census_trade_dir: Root of data/raw/census_trade/ Hive tree.

    Returns:
        df with CENSUS_TRADE_FEATURE_COLS appended (Float64). Zero-filled.
    """
    trade = _load_trade(census_trade_dir)
    query_dates = df.select(["date"]).unique()

    with duckdb.connect() as con:
        con.register("query_dates", query_dates.to_arrow())

        if not trade.is_empty():
            con.register("trade", trade.to_arrow())

            # Semiconductor import value + momentum (HS 8541+8542, all partners)
            semicon_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month, SUM(value_usd) / 1e6 AS value_m
                    FROM trade
                    WHERE direction = 'import'
                      AND hs_code IN ('8541', '8542')
                      AND partner_code = 'ALL'
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.value_m,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date <= q.date
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value_m END), 0.0)
                        AS semicon_import_value,
                    CASE
                        WHEN MAX(CASE WHEN rn = 2 THEN value_m END) IS NULL THEN 0.0
                        ELSE MAX(CASE WHEN rn = 1 THEN value_m END)
                           - MAX(CASE WHEN rn = 2 THEN value_m END)
                    END AS semicon_import_momentum
                FROM ranked
                GROUP BY date
            """).pl()

            # DC equipment import value + momentum (HS 8471+8473, all partners)
            dc_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month, SUM(value_usd) / 1e6 AS value_m
                    FROM trade
                    WHERE direction = 'import'
                      AND hs_code IN ('8471', '8473')
                      AND partner_code = 'ALL'
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.value_m,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date <= q.date
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value_m END), 0.0)
                        AS dc_equipment_import_value,
                    CASE
                        WHEN MAX(CASE WHEN rn = 2 THEN value_m END) IS NULL THEN 0.0
                        ELSE MAX(CASE WHEN rn = 1 THEN value_m END)
                           - MAX(CASE WHEN rn = 2 THEN value_m END)
                    END AS dc_equipment_import_momentum
                FROM ranked
                GROUP BY date
            """).pl()

            # China semiconductor export share
            china_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month,
                        SUM(CASE WHEN partner_code = '5700' THEN value_usd ELSE 0.0 END) AS china_val,
                        SUM(value_usd) AS total_val
                    FROM trade
                    WHERE direction = 'export'
                      AND hs_code IN ('8541', '8542')
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.china_val, d.total_val,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date <= q.date
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN china_val END), 0.0)
                    / GREATEST(COALESCE(MAX(CASE WHEN rn = 1 THEN total_val END), 1.0), 1.0)
                        AS china_semicon_export_share
                FROM ranked
                GROUP BY date
            """).pl()

            # Taiwan semiconductor import share
            taiwan_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month,
                        SUM(CASE WHEN partner_code = '5830' THEN value_usd ELSE 0.0 END) AS taiwan_val,
                        SUM(value_usd) AS total_val
                    FROM trade
                    WHERE direction = 'import'
                      AND hs_code IN ('8541', '8542')
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.taiwan_val, d.total_val,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date <= q.date
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN taiwan_val END), 0.0)
                    / GREATEST(COALESCE(MAX(CASE WHEN rn = 1 THEN total_val END), 1.0), 1.0)
                        AS taiwan_semicon_import_share
                FROM ranked
                GROUP BY date
            """).pl()

        else:
            semicon_result = pl.DataFrame(schema={
                "date": pl.Date,
                "semicon_import_value": pl.Float64,
                "semicon_import_momentum": pl.Float64,
            })
            dc_result = pl.DataFrame(schema={
                "date": pl.Date,
                "dc_equipment_import_value": pl.Float64,
                "dc_equipment_import_momentum": pl.Float64,
            })
            china_result = pl.DataFrame(schema={
                "date": pl.Date,
                "china_semicon_export_share": pl.Float64,
            })
            taiwan_result = pl.DataFrame(schema={
                "date": pl.Date,
                "taiwan_semicon_import_share": pl.Float64,
            })

    # Join all feature sets directly to original df
    df = df.join(semicon_result, on="date", how="left")
    df = df.join(dc_result, on="date", how="left")
    df = df.join(china_result, on="date", how="left")
    df = df.join(taiwan_result, on="date", how="left")

    # Zero-fill backstop
    for col in CENSUS_TRADE_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
```

- [ ] **Step 4: Run feature tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_census_trade_features.py -v
```

Expected: 9 passed

- [ ] **Step 5: Write 5 failing train tests**

Read `tests/test_train.py` to find where to append. Add these 5 tests at the end of the file:

```python
def test_feature_cols_includes_census():
    """FEATURE_COLS must contain all 6 CENSUS_TRADE_FEATURE_COLS and total must be 83."""
    from models.train import FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    assert len(CENSUS_TRADE_FEATURE_COLS) == 6
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 83, f"Expected 83 features, got {len(FEATURE_COLS)}"


def test_census_cols_absent_from_short_tier():
    """CENSUS cols must not appear in short tier — monthly data too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_census_cols_in_medium_tier():
    """CENSUS cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_census_cols_in_long_tier():
    """CENSUS cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_census_col_names_correct():
    """CENSUS_TRADE_FEATURE_COLS must contain exactly the 6 expected column names."""
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    expected = {
        "semicon_import_value",
        "semicon_import_momentum",
        "dc_equipment_import_value",
        "dc_equipment_import_momentum",
        "china_semicon_export_share",
        "taiwan_semicon_import_share",
    }
    assert set(CENSUS_TRADE_FEATURE_COLS) == expected
```

Also find `test_feature_cols_includes_labor` in `tests/test_train.py`. It has `assert len(FEATURE_COLS) == 77`. Change that to `assert len(FEATURE_COLS) == 83`.

Also find `test_feature_cols_includes_uspto_patent`. It has `assert len(FEATURE_COLS) == 77`. Change that to `assert len(FEATURE_COLS) == 83`.

- [ ] **Step 6: Run train tests to verify new tests fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_train.py -k "test_feature_cols_includes_census or test_census or test_feature_cols_includes_labor or test_feature_cols_includes_uspto" -v 2>&1 | head -30
```

Expected: the 5 new CENSUS tests fail + count assertions in labor/uspto tests fail

- [ ] **Step 7: Update `models/train.py`**

Read the file first. Make these 4 changes:

**Change A** — Add import after `from processing.labor_features import LABOR_FEATURE_COLS, join_labor_features` (line 45):
```python
from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS, join_census_trade_features
```

**Change B** — Append to FEATURE_COLS after `+ LABOR_FEATURE_COLS           # 73 → 77 features total`:
```python
    + CENSUS_TRADE_FEATURE_COLS    # 77 → 83 features total
```

**Change C** — In TIER_FEATURE_COLS:
- Change medium tier comment from `# all 77 features` to `# all 83 features`
- Add to long tier after `+ LABOR_FEATURE_COLS           # labor market cycles relevant at year+ horizons`:
```python
        + CENSUS_TRADE_FEATURE_COLS    # semiconductor trade cycles relevant at year+ horizons
```

**Change D** — In `build_training_dataset`, after the labor features join block (after `df = join_labor_features(df, usajobs_dir, jolts_dir)`):
```python
    # Join Census trade features (semiconductor + DC equipment import/export)
    census_trade_dir = fundamentals_dir.parent.parent / "census_trade"
    df = join_census_trade_features(df, census_trade_dir)
```

- [ ] **Step 8: Update `models/inference.py`**

Read the file first. Make these 2 changes:

**Change A** — Add import after `from processing.labor_features import join_labor_features` (line 48):
```python
from processing.census_trade_features import join_census_trade_features
```

**Change B** — In `_build_feature_df`, after the labor features block (after `df = join_labor_features(df, usajobs_dir, jolts_dir)`):
```python
    census_trade_dir = data_dir / "census_trade"
    df = join_census_trade_features(df, census_trade_dir)
```

- [ ] **Step 9: Update `tools/run_refresh.sh`**

Read the file first. Then:

1. Renumber all `X/14` steps to `X/15`
2. After the `14/15 BLS JOLTS` step and before `Refresh complete`, add:

```bash
echo ""
echo "=== 15/15  Census international trade (semiconductors + DC equipment) ==="
python ingestion/census_trade_ingestion.py
```

- [ ] **Step 10: Run the full test suite**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/ -m "not integration" -q --tb=short 2>&1 | tail -10
```

Expected: all tests pass including the 5 new CENSUS tests in test_train.py

- [ ] **Step 11: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add processing/census_trade_features.py tests/test_census_trade_features.py \
    models/train.py models/inference.py tests/test_train.py tools/run_refresh.sh
git commit -m "$(cat <<'EOF'
feat: wire Census trade features into model pipeline (FEATURE_COLS 77→83)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
