"""
Pipeline health check — prints data freshness for every source.

Usage:
    python tools/pipeline_health.py

Exit code 0 = all healthy, 1 = one or more sources stale or missing.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

_PROJECT_ROOT = Path(__file__).parent.parent
_TODAY = date.today()

# Each entry: (display_name, data_path_or_glob, date_column, cadence)
# cadence: "daily" → warn if >3 days, "weekly" → warn if >10 days, "quarterly" → never warn
_SOURCES: list[tuple[str, str, str, str]] = [
    (
        "OHLCV",
        str(_PROJECT_ROOT / "data/raw/financials/ohlcv/**/*.parquet"),
        "date",
        "daily",
    ),
    (
        "Short interest",
        str(_PROJECT_ROOT / "data/raw/financials/short_interest/short_interest_daily.parquet"),
        "date",
        "daily",
    ),
    (
        "Earnings",
        str(_PROJECT_ROOT / "data/raw/financials/earnings/earnings_surprises.parquet"),
        "quarter_end",
        "weekly",
    ),
    (
        "Sentiment (scored)",
        str(_PROJECT_ROOT / "data/raw/news/scored/**/*.parquet"),
        "article_date",
        "daily",
    ),
    (
        "Sentiment (features)",
        str(_PROJECT_ROOT / "data/raw/news/sentiment_features/*/daily.parquet"),
        "date",
        "daily",
    ),
    (
        "Graph features",
        str(_PROJECT_ROOT / "data/raw/graph/features/*/graph_daily.parquet"),
        "date",
        "weekly",
    ),
    (
        "13F holdings (raw)",
        str(_PROJECT_ROOT / "data/raw/financials/13f_holdings/raw/**/*.parquet"),
        "period_end",
        "quarterly",
    ),
    (
        "13F features",
        str(_PROJECT_ROOT / "data/raw/financials/13f_holdings/features/*/quarterly.parquet"),
        "period_end",
        "quarterly",
    ),
]

_STALE_DAYS = {"daily": 3, "weekly": 10, "quarterly": 999}


def _latest_date(glob_or_path: str, date_col: str) -> date | None:
    try:
        with duckdb.connect() as con:
            result = con.execute(
                f"SELECT MAX({date_col}) FROM read_parquet(?)", [glob_or_path]
            ).fetchone()
        val = result[0] if result else None
        if val is None:
            return None
        return val if isinstance(val, date) else date.fromisoformat(str(val)[:10])
    except Exception:
        return None


def main() -> int:
    print(f"\nAI Infra Predictor — Pipeline Health Check ({_TODAY})")
    print("═" * 60)
    print(f"{'Source':<28} {'Latest':>12}  {'Age':>8}  Status")
    print("─" * 60)

    any_problem = False
    for name, glob_path, date_col, cadence in _SOURCES:
        latest = _latest_date(glob_path, date_col)
        if latest is None:
            print(f"{name:<28} {'MISSING':>12}  {'—':>8}  ✗ MISSING")
            any_problem = True
            continue

        age_days = (_TODAY - latest).days
        threshold = _STALE_DAYS[cadence]
        status = "✓ OK" if age_days <= threshold else f"✗ STALE ({cadence} threshold: {threshold}d)"
        if age_days > threshold:
            any_problem = True

        print(f"{name:<28} {str(latest):>12}  {age_days:>6}d  {status}")

    print("═" * 60)
    if any_problem:
        print("Issues found. Run: bash tools/run_refresh.sh\n")
        return 1
    print("All sources healthy.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
