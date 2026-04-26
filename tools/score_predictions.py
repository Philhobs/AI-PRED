"""
Score saved predictions against realized OHLCV returns.

Reads `data/predictions/date=YYYY-MM-DD/horizon=Nd/predictions.parquet` and
joins to OHLCV at date+N to compute the actual N-day forward return per
ticker. Reports IC, hit-rate, top-decile return per (date, horizon).

Usage:
    python tools/score_predictions.py                    # score everything
    python tools/score_predictions.py --date 2026-04-24  # one date
    python tools/score_predictions.py --horizon 5d       # one horizon

Exit code 0 = at least one (date, horizon) was scoreable.
Exit code 1 = no scoreable predictions found (likely future hasn't arrived yet).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

_PROJECT_ROOT = Path(__file__).parent.parent
_OHLCV_GLOB = str(_PROJECT_ROOT / "data/raw/financials/ohlcv/*/*.parquet")
_PREDICTIONS_DIR = _PROJECT_ROOT / "data/predictions"

# Map horizon tag → forward shift in trading days
_HORIZON_DAYS: dict[str, int] = {
    "5d": 5, "20d": 20, "65d": 65, "252d": 252,
    "756d": 756, "1260d": 1260, "2520d": 2520, "5040d": 5040,
}


def _trading_day_close(ticker: str, after_date: date, n_days: int) -> tuple[date, float] | None:
    """Return the (close_date, close_price) of the Nth trading day on/after after_date.

    'Trading day' = any date in the OHLCV parquet (which is already trading-day-only).
    Returns None if not enough future data exists.
    """
    sql = """
        SELECT date, close_price
        FROM read_parquet(?)
        WHERE ticker = ? AND date >= ?
        ORDER BY date
        LIMIT ?
    """
    con = duckdb.connect()
    try:
        rows = con.execute(sql, [_OHLCV_GLOB, ticker, after_date.isoformat(), n_days + 1]).fetchall()
    finally:
        con.close()
    if len(rows) < n_days + 1:
        return None
    last = rows[n_days]
    last_date = last[0] if isinstance(last[0], date) else date.fromisoformat(str(last[0])[:10])
    return last_date, float(last[1])


def _load_close_at(ticker: str, on_date: date) -> float | None:
    """Return the close_price for a single ticker on a single date, or None."""
    sql = "SELECT close_price FROM read_parquet(?) WHERE ticker=? AND date=? LIMIT 1"
    con = duckdb.connect()
    try:
        row = con.execute(sql, [_OHLCV_GLOB, ticker, on_date.isoformat()]).fetchone()
    finally:
        con.close()
    return float(row[0]) if row else None


def _score_one(predictions_path: Path, as_of: date, horizon_days: int) -> dict | None:
    """Compute IC, hit-rate, top-decile return for one (date, horizon) prediction set.

    Returns None if too few tickers had realized returns (insufficient future data).
    """
    preds = pl.read_parquet(predictions_path)
    if "ticker" not in preds.columns or "expected_annual_return" not in preds.columns:
        return None

    rows: list[dict] = []
    for ticker, predicted in zip(preds["ticker"].to_list(), preds["expected_annual_return"].to_list()):
        if predicted is None:
            continue
        start_close = _load_close_at(ticker, as_of)
        if start_close is None or start_close <= 0:
            continue
        future = _trading_day_close(ticker, as_of, horizon_days)
        if future is None:
            continue
        future_date, future_close = future
        realized = (future_close / start_close) - 1.0
        rows.append({
            "ticker": ticker,
            "predicted": float(predicted),
            "realized": realized,
            "future_date": future_date,
        })

    if len(rows) < 10:
        return None

    df = pl.DataFrame(rows)
    # Information Coefficient (Pearson correlation between predicted and realized)
    ic = df.select(pl.corr("predicted", "realized")).item()
    # Hit rate: sign agreement
    hit = df.filter(pl.col("predicted").sign() == pl.col("realized").sign()).height / len(df)
    # Top decile mean realized return
    n_top = max(1, len(df) // 10)
    top_decile = df.sort("predicted", descending=True).head(n_top)["realized"].mean()

    return {
        "as_of": as_of.isoformat(),
        "horizon_days": horizon_days,
        "n_tickers": len(df),
        "ic": round(float(ic), 4) if ic is not None else None,
        "hit_rate": round(hit, 4),
        "top_decile_return": round(float(top_decile), 4),
        "future_date_range": (df["future_date"].min().isoformat(), df["future_date"].max().isoformat()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score saved predictions vs realized OHLCV.")
    parser.add_argument("--date", help="Score only this date (YYYY-MM-DD).")
    parser.add_argument("--horizon", help="Score only this horizon tag (e.g. '5d').")
    args = parser.parse_args()

    if not _PREDICTIONS_DIR.exists():
        print(f"No predictions directory at {_PREDICTIONS_DIR}. Run models/inference.py first.")
        return 1

    pred_files = sorted(_PREDICTIONS_DIR.glob("date=*/horizon=*/predictions.parquet"))
    if args.date:
        pred_files = [p for p in pred_files if f"date={args.date}" in p.parts[-3]]
    if args.horizon:
        pred_files = [p for p in pred_files if f"horizon={args.horizon}" in p.parts[-2]]

    if not pred_files:
        print("No matching prediction files.")
        return 1

    print(f"\n{'Date':<12} {'Horizon':>8} {'N':>5} {'IC':>8} {'Hit':>6} {'TopDec':>8}  Future")
    print("─" * 70)

    any_scored = False
    for path in pred_files:
        date_str = path.parts[-3].replace("date=", "")
        horizon_tag = path.parts[-2].replace("horizon=", "")
        if horizon_tag not in _HORIZON_DAYS:
            continue
        as_of = date.fromisoformat(date_str)
        horizon_days = _HORIZON_DAYS[horizon_tag]
        result = _score_one(path, as_of, horizon_days)
        if result is None:
            print(f"{date_str:<12} {horizon_tag:>8}     —    PENDING (future not yet realized)")
            continue
        any_scored = True
        ic_str = f"{result['ic']:+.4f}" if result['ic'] is not None else "  N/A"
        print(f"{date_str:<12} {horizon_tag:>8} {result['n_tickers']:>5} "
              f"{ic_str:>8} {result['hit_rate']:>6.3f} {result['top_decile_return']:>+8.4f}  "
              f"{result['future_date_range'][1]}")

    print("─" * 70)
    return 0 if any_scored else 1


if __name__ == "__main__":
    sys.exit(main())
