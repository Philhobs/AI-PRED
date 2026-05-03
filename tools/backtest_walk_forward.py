"""Walk-forward backtest harness — Phase C.

For every (target, horizon, as_of_date) prediction parquet on disk, compute:

  - **IC** (Pearson correlation of predicted vs realized return)
  - **Long-short portfolio return** = top decile − bottom decile, optionally
    sector-neutral within layer (top decile of each layer minus bottom decile
    of each layer, then averaged), after transaction costs.
  - **Hit rate** (sign agreement)
  - **Top-decile mean realized return**

Aggregates across matured dates produce per-(target, horizon) summary tables
for A/B comparison.

Limitations:
  - Uses *existing* model artifacts (one global cutoff). True per-fold
    walk-forward (retrain at each cutoff) is not implemented in v1; the
    in-sample bias affects all targets equally, so the comparative result
    is still meaningful but absolute IC is overstated.
  - Excess-target predictions are read from horizon=<H>_excess partitions.

Usage:
  python -m tools.backtest_walk_forward                     # all targets, all horizons
  python -m tools.backtest_walk_forward --horizon 5d         # one horizon
  python -m tools.backtest_walk_forward --target raw         # one target
  python -m tools.backtest_walk_forward --no-sector-neutral  # global decile

Default transaction cost:
  - 15 bps round-trip per name (entry + exit)
  - 50 bps/yr borrow charged on the short leg
Override with --tc-bps and --borrow-bps-per-year.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_PROJECT_ROOT = Path(__file__).parent.parent
_OHLCV_GLOB = str(_PROJECT_ROOT / "data/raw/financials/ohlcv/*/*.parquet")
_PREDICTIONS_DIR = _PROJECT_ROOT / "data/predictions"

# Map base horizon tag → forward shift in trading days
_HORIZON_DAYS: dict[str, int] = {
    "5d": 5, "20d": 20, "65d": 65, "252d": 252,
    "756d": 756, "1260d": 1260, "2520d": 2520, "5040d": 5040,
}

_HORIZON_PARTITION_RE = re.compile(r"^horizon=(\d+d)(_excess)?$")


def _parse_partition(path: Path) -> tuple[date, str, str] | None:
    """Extract (as_of_date, horizon_tag, target) from a predictions parquet path.

    Returns None if the path doesn't match the expected partition layout.
    """
    parts = path.parts
    try:
        date_part = next(p for p in parts if p.startswith("date="))
        horizon_part = next(p for p in parts if p.startswith("horizon="))
    except StopIteration:
        return None
    m = _HORIZON_PARTITION_RE.match(horizon_part)
    if not m:
        return None
    horizon_tag = m.group(1)
    if horizon_tag not in _HORIZON_DAYS:
        return None
    target = "excess" if m.group(2) else "raw"
    try:
        as_of = date.fromisoformat(date_part.replace("date=", ""))
    except ValueError:
        return None
    return as_of, horizon_tag, target


def _load_realized(as_of: date, horizon_days: int, tickers: list[str]) -> dict[str, float]:
    """Bulk-load realized N-trading-day forward return per ticker via DuckDB.

    Returns ticker → realized_return. Tickers without enough forward data are
    omitted from the result.
    """
    if not tickers:
        return {}
    con = duckdb.connect()
    try:
        # For each ticker: rank trading days from as_of forward; row 0 is the
        # entry close, row horizon_days is the exit close.
        sql = f"""
        WITH ranked AS (
            SELECT ticker, date, close_price,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) - 1 AS day_idx
            FROM read_parquet(?)
            WHERE ticker IN ({",".join("?" * len(tickers))})
              AND date >= ?
        )
        SELECT entry.ticker,
               (exit.close_price / entry.close_price) - 1.0 AS realized
        FROM ranked entry
        JOIN ranked exit
          ON exit.ticker = entry.ticker AND exit.day_idx = ?
        WHERE entry.day_idx = 0
          AND entry.close_price > 0
        """
        params = [_OHLCV_GLOB, *tickers, as_of.isoformat(), horizon_days]
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return {r[0]: float(r[1]) for r in rows if r[1] is not None}


def _decile_split(df: pl.DataFrame, sector_neutral: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (top_decile_df, bottom_decile_df) given a DataFrame with columns
    [ticker, layer, predicted, realized]. If sector_neutral, decile is taken
    within each layer; otherwise globally."""
    if sector_neutral:
        # Take top 10% / bottom 10% within each layer
        df_sorted = df.sort(["layer", "predicted"], descending=[False, True])
        per_layer_n = df.group_by("layer").len()
        # Assign rank within layer
        ranked = df_sorted.with_columns(
            pl.col("predicted").rank(method="ordinal", descending=True).over("layer").alias("rank_in_layer")
        ).join(per_layer_n.rename({"len": "n_in_layer"}), on="layer")
        top = ranked.filter(pl.col("rank_in_layer") <= pl.max_horizontal(pl.col("n_in_layer") // 10, pl.lit(1)))
        bot = ranked.filter(pl.col("rank_in_layer") > pl.col("n_in_layer") - pl.max_horizontal(pl.col("n_in_layer") // 10, pl.lit(1)))
        return top.drop(["rank_in_layer", "n_in_layer"]), bot.drop(["rank_in_layer", "n_in_layer"])
    n = max(1, len(df) // 10)
    sorted_df = df.sort("predicted", descending=True)
    return sorted_df.head(n), sorted_df.tail(n)


def _score_one(
    predictions_path: Path,
    as_of: date,
    horizon_days: int,
    sector_neutral: bool,
    tc_bps: float,
    borrow_bps_per_year: float,
) -> dict | None:
    """Score one prediction parquet against realized returns.

    Returns a metrics dict, or None if too few tickers had matured returns.
    """
    preds = pl.read_parquet(predictions_path)
    needed = {"ticker", "layer", "expected_annual_return"}
    if not needed.issubset(set(preds.columns)):
        return None

    preds = preds.filter(pl.col("expected_annual_return").is_not_null())
    if preds.is_empty():
        return None

    tickers = preds["ticker"].to_list()
    realized = _load_realized(as_of, horizon_days, tickers)
    if len(realized) < 10:  # not enough matured tickers to score
        return None

    rows = []
    for t, p, layer in zip(preds["ticker"].to_list(),
                            preds["expected_annual_return"].to_list(),
                            preds["layer"].to_list()):
        if t in realized:
            rows.append({"ticker": t, "layer": layer, "predicted": float(p), "realized": realized[t]})
    df = pl.DataFrame(rows)

    # IC (Pearson) and hit rate
    ic = df.select(pl.corr("predicted", "realized")).item()
    hit = df.filter(pl.col("predicted").sign() == pl.col("realized").sign()).height / df.height

    # Long-short portfolio
    top_df, bot_df = _decile_split(df, sector_neutral=sector_neutral)
    top_realized = float(top_df["realized"].mean())
    bot_realized = float(bot_df["realized"].mean())
    raw_ls_return = (top_realized - bot_realized) / 2.0  # half capital each leg

    # Transaction costs:
    #   - round-trip = enter + exit = 2 trades per name. We charge once per
    #     name in BOTH legs (long + short). Approximation: tc_per_dollar =
    #     tc_bps / 10000. The portfolio holds n_long + n_short names; we
    #     amortize across the deciles (which have ~equal capital each).
    n_long = top_df.height
    n_short = bot_df.height
    tc_drag = (tc_bps / 10000.0) * 2.0  # round-trip applied once over the holding window
    # Borrow charge on short leg, prorated to horizon
    horizon_years = horizon_days / 252.0
    borrow_drag = (borrow_bps_per_year / 10000.0) * horizon_years
    # Net: long leg pays tc_drag; short leg pays tc_drag + borrow_drag.
    # Each leg holds half capital → per-dollar cost = (tc_drag + (tc_drag + borrow_drag)) / 2.
    cost_per_dollar = tc_drag + borrow_drag / 2.0
    net_ls_return = raw_ls_return - cost_per_dollar

    return {
        "as_of": as_of.isoformat(),
        "horizon_days": horizon_days,
        "n_tickers": df.height,
        "n_long": n_long,
        "n_short": n_short,
        "ic": float(ic) if ic is not None else None,
        "hit_rate": hit,
        "top_decile_realized": top_realized,
        "bot_decile_realized": bot_realized,
        "ls_return_gross": raw_ls_return,
        "ls_return_net": net_ls_return,
        "cost_drag": cost_per_dollar,
    }


def _summarize(rows: list[dict]) -> dict:
    """Aggregate per-fold metrics into headline summary stats."""
    if not rows:
        return {}
    ic_vals = [r["ic"] for r in rows if r["ic"] is not None]
    hit_vals = [r["hit_rate"] for r in rows]
    ls_net_vals = [r["ls_return_net"] for r in rows]
    return {
        "n_folds": len(rows),
        "mean_ic":          sum(ic_vals) / len(ic_vals) if ic_vals else None,
        "mean_hit_rate":    sum(hit_vals) / len(hit_vals),
        "mean_ls_net":      sum(ls_net_vals) / len(ls_net_vals),
        "best_fold_ls_net": max(ls_net_vals),
        "worst_fold_ls_net": min(ls_net_vals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", choices=["raw", "excess", "both"], default="both")
    parser.add_argument("--horizon", default=None,
                        help="Filter to one horizon tag (e.g. 5d). Default: all.")
    parser.add_argument("--no-sector-neutral", action="store_true",
                        help="Take global decile instead of per-layer decile.")
    parser.add_argument("--tc-bps", type=float, default=15.0,
                        help="Round-trip transaction cost in bps (default 15).")
    parser.add_argument("--borrow-bps-per-year", type=float, default=50.0,
                        help="Annualized borrow cost on shorts (default 50 bps).")
    args = parser.parse_args()
    sector_neutral = not args.no_sector_neutral

    pred_files = sorted(_PREDICTIONS_DIR.glob("date=*/horizon=*/predictions.parquet"))
    if not pred_files:
        print(f"No predictions found under {_PREDICTIONS_DIR}.")
        return 1

    print(f"\nWalk-forward backtest — sector_neutral={sector_neutral}, "
          f"tc={args.tc_bps}bps round-trip, borrow={args.borrow_bps_per_year}bps/yr\n")

    targets_to_run = ["raw", "excess"] if args.target == "both" else [args.target]
    by_bucket: dict[tuple[str, str], list[dict]] = {}  # (target, horizon_tag) → rows
    pending: dict[tuple[str, str], list[date]] = {}    # for reporting unscoreable folds

    for path in pred_files:
        parsed = _parse_partition(path)
        if parsed is None:
            continue
        as_of, horizon_tag, target = parsed
        if target not in targets_to_run:
            continue
        if args.horizon and horizon_tag != args.horizon:
            continue

        result = _score_one(
            path, as_of, _HORIZON_DAYS[horizon_tag],
            sector_neutral=sector_neutral,
            tc_bps=args.tc_bps,
            borrow_bps_per_year=args.borrow_bps_per_year,
        )
        key = (target, horizon_tag)
        if result is None:
            pending.setdefault(key, []).append(as_of)
        else:
            by_bucket.setdefault(key, []).append(result)

    # Per-bucket summary
    if not by_bucket and not pending:
        print("Nothing matched.")
        return 1

    print(f"{'Target':<7} {'Horiz':>6} {'Folds':>6} {'IC':>9} "
          f"{'Hit':>6} {'TopDec':>9} {'BotDec':>9} {'LS Gross':>9} {'LS Net':>9}")
    print("─" * 90)

    any_scored = False
    for (target, horizon_tag), rows in sorted(by_bucket.items()):
        any_scored = True
        s = _summarize(rows)
        ic = f"{s['mean_ic']:+.4f}" if s['mean_ic'] is not None else "  N/A"
        # Show fold-by-fold detail
        for r in sorted(rows, key=lambda x: x["as_of"]):
            print(f"{target:<7} {horizon_tag:>6} {r['as_of']:>11}  "
                  f"n={r['n_tickers']:3d}  IC={(r['ic'] or 0):+.4f}  "
                  f"hit={r['hit_rate']:.3f}  topD={r['top_decile_realized']:+.4f}  "
                  f"botD={r['bot_decile_realized']:+.4f}  LSnet={r['ls_return_net']:+.4f}")
        print(f"  ── {target}/{horizon_tag} mean: IC={ic}  hit={s['mean_hit_rate']:.3f}  "
              f"LSnet={s['mean_ls_net']:+.4f}  (across {s['n_folds']} folds)")
        print()

    if pending:
        print("\nPending (not yet matured):")
        for (target, horizon_tag), dates in sorted(pending.items()):
            for d in dates:
                print(f"  {target:<7} {horizon_tag:>6}  {d}")

    # A/B comparison if both targets scored
    if args.target == "both" and any_scored:
        print("\n=== A/B comparison ===")
        horizons_compared = {h for (t, h) in by_bucket}
        print(f"{'Horiz':>6} {'RAW IC':>9} {'EXC IC':>9} {'RAW LSnet':>10} {'EXC LSnet':>10} {'Δ LSnet (exc-raw)':>20}")
        for h in sorted(horizons_compared):
            raw_rows = by_bucket.get(("raw", h), [])
            exc_rows = by_bucket.get(("excess", h), [])
            if not raw_rows or not exc_rows:
                continue
            raw_s = _summarize(raw_rows)
            exc_s = _summarize(exc_rows)
            d_ls = exc_s["mean_ls_net"] - raw_s["mean_ls_net"]
            print(f"{h:>6} {raw_s['mean_ic'] or 0:+.4f}  {exc_s['mean_ic'] or 0:+.4f}  "
                  f"{raw_s['mean_ls_net']:+.4f}  {exc_s['mean_ls_net']:+.4f}  "
                  f"{d_ls:+.4f}  ({'EXCESS wins' if d_ls > 0 else 'RAW wins'})")

    return 0 if any_scored else 1


if __name__ == "__main__":
    sys.exit(main())
