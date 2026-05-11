"""Phase E2: feature-data history-depth audit.

For each on-disk feature source under data/raw/, reports:
  - first/last observed date in the data
  - row count
  - first-non-null date per representative column (the "effective signal start")
  - missing-data directories (features that train as always-null)

Helps decide which horizon/cutoff combinations are feasible for backtests.
Long horizons (252d / 756d+) require both:
  (a) enough forward time after cutoff for the holdout to mature
  (b) enough deep-history features that aren't just imputed nulls

Run:
  python -m tools.feature_history_audit
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb


_PROJECT_ROOT = Path(__file__).parent.parent
_DR = _PROJECT_ROOT / "data" / "raw"


_SOURCES: list[tuple[str, str]] = [
    ("price_features (OHLCV)",        str(_DR / "financials" / "ohlcv" / "*" / "*.parquet")),
    ("fundamentals",                  str(_DR / "financials" / "fundamentals" / "*" / "*.parquet")),
    ("insider_features",              str(_DR / "financials" / "insider_features" / "*" / "daily.parquet")),
    ("sentiment_features",            str(_DR / "news" / "sentiment_features" / "*" / "daily.parquet")),
    ("short_interest_features",       str(_DR / "financials" / "short_interest_features" / "*" / "si_daily.parquet")),
    ("earnings_features",             str(_DR / "financials" / "earnings_features" / "*" / "earnings_daily.parquet")),
    ("13f_holdings.raw",              str(_DR / "financials" / "13f_holdings" / "raw" / "*" / "*.parquet")),
    ("13f_holdings.features",         str(_DR / "financials" / "13f_holdings" / "features" / "*" / "*.parquet")),
    ("graph.edges",                   str(_DR / "graph" / "edges.parquet")),
    ("graph.deals",                   str(_DR / "graph" / "deals.parquet")),
    ("energy_geo.country_energy",     str(_DR / "energy_geo" / "country_energy.parquet")),
    ("fx",                            str(_DR / "financials" / "fx" / "*.parquet")),
    ("cyber_threat",                  str(_DR / "cyber_threat" / "date=*" / "*.parquet")),
    ("robotics_signals (FRED)",       str(_DR / "robotics_signals" / "*.parquet")),
    ("bls_jolts",                     str(_DR / "bls_jolts" / "date=*" / "*.parquet")),
    ("census_trade",                  str(_DR / "census_trade" / "date=*" / "*.parquet")),
    ("ferc_queue",                    str(_DR / "ferc_queue" / "date=*" / "*.parquet")),
    ("ai_economics",                  str(_DR / "financials" / "ai_economics" / "*.parquet")),
]

_MISSING_DIRS: list[tuple[str, str]] = [
    ("options_features",         "data/raw/options/"),
    ("gov_behavioral.contracts", "data/raw/gov_contracts/"),
    ("patent_features.apps",     "data/raw/patents/applications/"),
    ("patent_features.grants",   "data/raw/patents/grants/"),
    ("labor.usajobs",            "data/raw/usajobs/"),
    ("physical_ai.uspto",        "data/raw/uspto/physical_ai/"),
]

# Representative columns per module for the "first non-null" probe.
_NONNULL_PROBES: list[tuple[str, str, list[str]]] = [
    ("insider", str(_DR / "financials" / "insider_features" / "*" / "daily.parquet"),
        ["cluster_buy", "insider_buy_count_30d", "insider_sell_count_30d"]),
    ("sentiment", str(_DR / "news" / "sentiment_features" / "*" / "daily.parquet"),
        ["sentiment_mean_7d", "article_count_7d"]),
    ("short_interest", str(_DR / "financials" / "short_interest_features" / "*" / "si_daily.parquet"),
        ["short_vol_ratio_10d", "short_interest_pct_float"]),
    ("earnings", str(_DR / "financials" / "earnings_features" / "*" / "earnings_daily.parquet"),
        ["eps_surprise_last", "eps_beat_streak", "days_since_earnings"]),
    ("13f_features", str(_DR / "financials" / "13f_holdings" / "features" / "*" / "*.parquet"),
        ["inst_ownership_pct", "inst_holder_count", "top10_concentration"]),
]


def _fmt(d) -> str:
    return str(d)[:10] if d is not None else "—"


def _audit_source(con, label: str, glob: str) -> dict:
    """Return (label, earliest, latest, n_rows, date_col, status)."""
    try:
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()
        cols = [r[0] for r in desc]
        types = {r[0]: r[1] for r in desc}
    except Exception:
        return dict(label=label, earliest=None, latest=None, n=0, col="—", status="NO MATCH")

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]

    date_cols: list[str] = []
    for c in cols:
        if "DATE" in str(types[c]).upper() or "TIMESTAMP" in str(types[c]).upper():
            date_cols.append(c)
    for c in ("date", "available_date", "report_date", "period_date", "period_end",
              "filing_date", "fiscal_period_end", "deal_date", "publication_date",
              "as_of_date", "last_deal_date", "snapshot_date", "filed_at", "year"):
        if c in cols and c not in date_cols:
            date_cols.append(c)
    if not date_cols:
        return dict(label=label, earliest=None, latest=None, n=n, col="—", status="STATIC")

    col = date_cols[0]
    mn, mx = con.execute(
        f"SELECT MIN({col}), MAX({col}) FROM read_parquet('{glob}')"
    ).fetchone()
    return dict(label=label, earliest=mn, latest=mx, n=n, col=col, status="OK")


def _probe_nonnull(con, module: str, glob: str, cols: list[str]) -> list[dict]:
    rows = []
    for c in cols:
        for dcol in ("date", "available_date", "period_end", "as_of_date"):
            try:
                first_nn, total, nn = con.execute(
                    f"SELECT MIN({dcol}) FILTER (WHERE {c} IS NOT NULL), "
                    f"COUNT(*), COUNT({c}) FROM read_parquet('{glob}')"
                ).fetchone()
                pct = (nn / total * 100) if total else 0.0
                rows.append(dict(module=module, col=c, first_nonnull=first_nn, coverage_pct=pct))
                break
            except Exception:
                continue
    return rows


def main() -> int:
    con = duckdb.connect()

    print(f"\n{'─' * 100}")
    print("  Phase E2: Feature data history depth")
    print(f"{'─' * 100}")
    print(f"{'source':<32s}  {'earliest':<11s}  {'latest':<11s}  {'rows':>10s}  {'date col':<18s}  status")
    print("─" * 100)
    for label, glob in _SOURCES:
        r = _audit_source(con, label, glob)
        print(f"  {r['label']:<30s}  {_fmt(r['earliest']):<11s}  {_fmt(r['latest']):<11s}  "
              f"{r['n']:>10}  {r['col']:<18s}  {r['status']}")

    print(f"\n{'─' * 100}")
    print("  No data directory — features train + infer as always-null")
    print(f"{'─' * 100}")
    for label, path in _MISSING_DIRS:
        full = _PROJECT_ROOT / path.rstrip("/")
        state = "EMPTY DIR" if full.exists() else "NO DIR"
        print(f"  {label:<28s}  {path:<40s}  {state}")

    print(f"\n{'─' * 100}")
    print("  Effective signal start (first date column is NOT NULL)")
    print(f"{'─' * 100}")
    print(f"{'module':<14s}  {'column':<32s}  {'first non-null':<14s}  {'coverage':>10s}")
    print("─" * 100)
    for module, glob, cols in _NONNULL_PROBES:
        for r in _probe_nonnull(con, module, glob, cols):
            print(f"  {r['module']:<12s}  {r['col']:<32s}  "
                  f"{_fmt(r['first_nonnull']):<14s}  {r['coverage_pct']:>9.1f}%")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
