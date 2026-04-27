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
# cadence: "daily" → warn if >3 days, "weekly" → warn if >10 days,
#          "monthly" → warn if >60 days (covers FRED's 4-6 week pub lag),
#          "quarterly" → never warn
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
        "monthly",
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
    (
        "FX rates (9 pairs)",
        str(_PROJECT_ROOT / "data/raw/financials/fx/*.parquet"),
        "date",
        "daily",
    ),
    (
        "Insider trades (Form 4)",
        str(_PROJECT_ROOT / "data/raw/financials/insider_trades/**/*.parquet"),
        "filed_date",
        "weekly",
    ),
    (
        "Deal graph (8-K)",
        str(_PROJECT_ROOT / "data/raw/graph/deals.parquet"),
        "date",
        "monthly",   # 8-K material agreements are naturally sporadic
    ),
    (
        "Robotics signals (FRED 4 series)",
        str(_PROJECT_ROOT / "data/raw/robotics_signals/*.parquet"),
        "date",
        "monthly",
    ),
    (
        "BLS JOLTS",
        str(_PROJECT_ROOT / "data/raw/bls_jolts/date=*/openings.parquet"),
        "date",
        "monthly",
    ),
    (
        "Census trade",
        str(_PROJECT_ROOT / "data/raw/census_trade/date=*/trade.parquet"),
        "date",
        "monthly",
    ),
    (
        "USPTO physical-AI",
        str(_PROJECT_ROOT / "data/raw/uspto/physical_ai/cpc_class=*/filings.parquet"),
        "quarter_end",
        "quarterly",
    ),
    (
        "FERC interconnection queue",
        str(_PROJECT_ROOT / "data/raw/ferc_queue/date=*/queue.parquet"),
        "snapshot_date",
        "quarterly",
    ),
    (
        "USAJOBS",
        str(_PROJECT_ROOT / "data/raw/usajobs/date=*/postings.parquet"),
        "posted_date",
        "weekly",
    ),
    (
        "SAM.gov contracts",
        str(_PROJECT_ROOT / "data/raw/sam_gov/date=*/awards.parquet"),
        "award_date",
        "weekly",
    ),
    (
        "Cyber threat (NVD)",
        str(_PROJECT_ROOT / "data/raw/cyber_threat/date=*/threats.parquet"),
        "date",
        "weekly",
    ),
    (
        "OWID energy geography",
        str(_PROJECT_ROOT / "data/raw/energy_geo/country_energy.parquet"),
        "year",
        "quarterly",
    ),
    # NOTE: data/raw/financials/market_caps.parquet is an *inference-side* cache
    # written by processing/portfolio_metrics.py during prediction enrichment, not
    # by the daily refresh pipeline. Excluded from health check intentionally.
]

_STALE_DAYS = {"daily": 3, "weekly": 10, "monthly": 60, "quarterly": 999}


def _latest_date(glob_or_path: str, date_col: str) -> date | None:
    try:
        with duckdb.connect() as con:
            result = con.execute(
                f"SELECT MAX({date_col}) FROM read_parquet(?)", [glob_or_path]
            ).fetchone()
        val = result[0] if result else None
        if val is None:
            return None
        if isinstance(val, date):
            return val
        if isinstance(val, int):
            # Annual data — interpret year as Jan 1 of that year
            return date(val, 1, 1)
        return date.fromisoformat(str(val)[:10])
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
