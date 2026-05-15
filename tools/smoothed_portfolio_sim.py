"""Phase 2.7: smoothed-rebalance portfolio simulator.

The Phase 2.6 trade-by-trade max drawdown is an UPPER BOUND. In real
deployment with daily walk-forward, each trading day's L/S portfolio is
the AVERAGE position across the past `horizon_days` selections — adjacent
overlapping folds cancel each other's idiosyncratic noise, so the
realized portfolio P&L curve is much smoother than the trade-by-trade
simulation suggests.

This script does the smoothed simulation correctly:

  For each trading day t in the holdout:
    active = { fold_start dates s : t - horizon_days < s ≤ t and we have
               a prediction for s }
    For each s in active, compute its top/bot decile members.
    Aggregate weights per ticker:
      long_weight  = +(0.5 / n_top(s)) / |active|
      short_weight = -(0.5 / n_bot(s)) / |active|
    Portfolio weights are summed across all active folds.
    daily_return(t) = Σ_t  w(t, ticker) × daily_close-to-close return(t, ticker)

  Turnover cost: tc_bps × Σ_ticker |w(t) - w(t-1)| / 2
  Short-leg borrow: borrow_bps_yr × (sum of |short weights|) / 252

  Equity_t = Equity_{t-1} × (1 + daily_return_net(t))

Reports: annualized return, daily Sharpe, max drawdown (on real
smoothed equity curve), longest drawdown duration. Compares against the
mean and trade-by-trade max drawdown from Phase 2.6.

Run:
  python -m tools.smoothed_portfolio_sim                  # both horizons
  python -m tools.smoothed_portfolio_sim --horizon 65d
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from tools.backtest_walk_forward import _HORIZON_DAYS
from tools.daily_inference import PRODUCTION_CUTOFFS, _resolve_default_ablation
from tools.robustness_audit import _decile_split_pct

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent
_OHLCV_GLOB = str(_PROJECT_ROOT / "data" / "raw" / "financials" / "ohlcv" / "*" / "*.parquet")


def _load_daily_returns(start: date, end: date, tickers: list[str]) -> pl.DataFrame:
    """Return long DataFrame of (ticker, date, daily_return) for the given
    tickers across [start, end]. daily_return = close[t]/close[t-1] - 1."""
    if not tickers:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date,
                                    "daily_return": pl.Float64})
    con = duckdb.connect()
    try:
        ph = ",".join("?" * len(tickers))
        sql = f"""
        WITH ranked AS (
          SELECT ticker, date, close_price,
                 LAG(close_price) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
          FROM read_parquet(?)
          WHERE ticker IN ({ph})
            AND date >= ? AND date <= ?
        )
        SELECT ticker, date,
               (close_price / NULLIF(prev_close, 0)) - 1.0 AS daily_return
        FROM ranked
        WHERE prev_close IS NOT NULL
        """
        params = [_OHLCV_GLOB, *tickers, start.isoformat(), end.isoformat()]
        df = con.execute(sql, params).pl()
    finally:
        con.close()
    return df.with_columns(pl.col("date").cast(pl.Date))


def _trading_days(start: date, end: date) -> list[date]:
    """All trading days in [start, end] derived from NVDA OHLCV."""
    nvda = _PROJECT_ROOT / "data" / "raw" / "financials" / "ohlcv" / "NVDA"
    files = sorted(nvda.glob("*.parquet"))
    if not files:
        return []
    df = pl.concat([pl.read_parquet(f).select("date") for f in files]).unique()
    days = [d for d in df["date"].to_list() if start <= d <= end]
    days.sort()
    return days


def _prediction_dates(base: Path, horizon: str) -> dict[date, Path]:
    """Map as_of_date → path to predictions parquet for `horizon`."""
    out: dict[date, Path] = {}
    for date_dir in sorted(base.glob("date=*")):
        f = date_dir / f"horizon={horizon}" / "predictions.parquet"
        if f.exists():
            as_of = date.fromisoformat(date_dir.name.replace("date=", ""))
            out[as_of] = f
    return out


def _per_fold_picks(
    pred_files: dict[date, Path],
    decile_pct: float,
    sector_neutral: bool,
) -> dict[date, tuple[list[str], list[str]]]:
    """For each fold, return (long_names, short_names) from the prediction file."""
    out: dict[date, tuple[list[str], list[str]]] = {}
    for as_of, path in pred_files.items():
        preds = pl.read_parquet(path).filter(pl.col("expected_annual_return").is_not_null())
        if preds.is_empty():
            continue
        # Rename to match _decile_split_pct's expected schema
        df = preds.rename({"expected_annual_return": "predicted"})
        top, bot = _decile_split_pct(df, sector_neutral, decile_pct)
        out[as_of] = (top["ticker"].to_list(), bot["ticker"].to_list())
    return out


def _simulate(
    horizon: str,
    decile_pct: float,
    sector_neutral: bool,
    tc_bps: float,
    borrow_bps_yr: float,
) -> dict:
    """Run the smoothed-rebalance simulation. Returns summary stats."""
    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}")
    if ablation != "none":
        base = base / f"ablation={ablation}"
    horizon_td = _HORIZON_DAYS[horizon]

    # Pre-compute per-fold picks
    pred_files = _prediction_dates(base, horizon)
    if not pred_files:
        return {"n_days": 0}
    picks = _per_fold_picks(pred_files, decile_pct, sector_neutral)
    if not picks:
        return {"n_days": 0}

    # Universe = union of all selected names
    universe: set[str] = set()
    for longs, shorts in picks.values():
        universe.update(longs)
        universe.update(shorts)
    tickers = sorted(universe)

    # Simulation window: first as_of date through the latest OHLCV
    sim_start = min(picks)
    nvda_dir = _PROJECT_ROOT / "data" / "raw" / "financials" / "ohlcv" / "NVDA"
    nvda_files = sorted(nvda_dir.glob("*.parquet"))
    last_ohlcv = max(
        pl.read_parquet(f).select("date").to_series().max()
        for f in nvda_files
    )

    sim_days = _trading_days(sim_start, last_ohlcv)
    if not sim_days:
        return {"n_days": 0}

    # Load daily returns once
    daily = _load_daily_returns(sim_start, last_ohlcv, tickers)
    # Index by (date, ticker) for fast lookup
    daily_lookup: dict[tuple[date, str], float] = {
        (d, t): r for d, t, r in zip(
            daily["date"].to_list(),
            daily["ticker"].to_list(),
            daily["daily_return"].to_list(),
        )
    }

    # Sorted list of fold start dates for window slicing
    fold_dates = sorted(picks)

    # Walk trading days, compute daily portfolio weights and P&L
    prev_weights: dict[str, float] = {}
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    daily_returns: list[float] = []
    days_since_peak = 0
    longest_dd_duration = 0

    horizon_years = horizon_td / 252.0
    daily_borrow_rate = (borrow_bps_yr / 10000.0) / 252.0

    for t in sim_days:
        # Active folds: those with start ∈ (t - horizon_td trading days, t]
        # Approximate using fold_dates index (assumes ≈ trading-day spacing)
        # Take fold_dates s.t. s ≤ t and count back by trading days using sim_days.
        active = [s for s in fold_dates if s <= t]
        if not active:
            prev_weights = {}
            continue
        # Restrict to last horizon_td fold dates (≈ horizon_days trading days back)
        active = active[-horizon_td:]
        n_active = len(active)

        # Aggregate weights
        weights: dict[str, float] = {}
        for s in active:
            longs, shorts = picks.get(s, ([], []))
            if longs:
                lw = 0.5 / len(longs) / n_active
                for tk in longs:
                    weights[tk] = weights.get(tk, 0.0) + lw
            if shorts:
                sw = -0.5 / len(shorts) / n_active
                for tk in shorts:
                    weights[tk] = weights.get(tk, 0.0) + sw

        # Daily portfolio return (gross)
        gross = 0.0
        for tk, w in weights.items():
            r = daily_lookup.get((t, tk))
            if r is not None:
                gross += w * r

        # Turnover and tc drag
        all_tk = set(weights) | set(prev_weights)
        turnover = sum(abs(weights.get(tk, 0.0) - prev_weights.get(tk, 0.0))
                       for tk in all_tk) / 2.0
        tc_drag = turnover * tc_bps / 10000.0

        # Short-leg borrow drag (daily)
        short_notional = sum(-w for w in weights.values() if w < 0)
        borrow_drag = short_notional * daily_borrow_rate

        net = gross - tc_drag - borrow_drag
        daily_returns.append(net)

        equity *= (1.0 + net)
        if equity > peak:
            peak = equity
            days_since_peak = 0
        else:
            days_since_peak += 1
            longest_dd_duration = max(longest_dd_duration, days_since_peak)
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

        prev_weights = weights

    n = len(daily_returns)
    if n == 0:
        return {"n_days": 0}

    mean_daily = sum(daily_returns) / n
    var = sum((r - mean_daily) ** 2 for r in daily_returns) / max(1, n - 1)
    std_daily = math.sqrt(var)
    sharpe = (mean_daily / std_daily * math.sqrt(252)) if std_daily > 0 else float("nan")
    ann_return = (equity ** (252.0 / n) - 1.0) if n > 0 else float("nan")
    pos_days = sum(1 for r in daily_returns if r > 0)

    return {
        "n_days":      n,
        "ann_return":  ann_return,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "max_dd_days": longest_dd_duration,
        "pos_day_pct": pos_days / n,
        "final_eq":    equity,
        "mean_daily":  mean_daily,
        "std_daily":   std_daily,
    }


def _simulate_horizon(horizon: str) -> None:
    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)
    print()
    print("=" * 96)
    print(f"  SMOOTHED-REBALANCE SIM — horizon={horizon}, cutoff={cutoff}, ablation={ablation}")
    print("=" * 96)
    print()
    print(f"  {'variant':<36s} {'days':>5s} {'ann_ret':>9s} {'sharpe':>7s} "
          f"{'max_dd':>9s} {'dd_days':>8s} {'pos%':>5s} {'final_eq':>9s}")
    print("  " + "-" * 92)

    VARIANTS = [
        (10.0, True),   # baseline
        (10.0, False),
        (5.0,  True),
        (5.0,  False),
        (20.0, False),
    ]

    for pct, sn in VARIANTS:
        s = _simulate(horizon, decile_pct=pct, sector_neutral=sn,
                      tc_bps=15.0, borrow_bps_yr=50.0)
        if s.get("n_days", 0) == 0:
            continue
        sn_str = "sector" if sn else "global"
        label = f"pct={pct:>4.0f}%  {sn_str}"
        marker = "  ← baseline" if (pct, sn) == (10.0, True) else ""
        print(f"  {label:<36s} {s['n_days']:>5d} {s['ann_return']*100:>+8.1f}% "
              f"{s['sharpe']:>+7.2f} {s['max_dd']*100:>+8.1f}% "
              f"{s['max_dd_days']:>8d} {s['pos_day_pct']*100:>4.0f}% "
              f"{s['final_eq']:>+9.2f}x{marker}")

    print()
    print("  Legend:")
    print("    ann_ret   = annualized total return over simulation window")
    print("    sharpe    = daily Sharpe ratio (mean / std × sqrt(252))")
    print("    max_dd    = max drawdown of smoothed equity curve (real, not")
    print("                trade-by-trade upper bound)")
    print("    dd_days   = longest stretch of days below the peak (drawdown duration)")
    print("    pos%      = fraction of days with positive net return")
    print("    final_eq  = cumulative equity multiplier at end of sim")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--horizon", choices=list(PRODUCTION_CUTOFFS) + ["all"], default="all",
    )
    args = parser.parse_args()
    horizons = list(PRODUCTION_CUTOFFS) if args.horizon == "all" else [args.horizon]
    for h in horizons:
        _simulate_horizon(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
