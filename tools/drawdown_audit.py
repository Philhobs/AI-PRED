"""Phase 2.6: distribution + drawdown audit per variant.

Phase 2.5 reported MEAN LSnet per variant — telling us the average outcome.
This audit reports the DISTRIBUTION of outcomes:
  - mean (recap from 2.5)
  - worst single fold (worst-case one-shot loss)
  - 5th percentile (typical bad fold)
  - fraction of folds with positive LSnet
  - worst losing streak (longest run of consecutive negative folds)
  - trade-by-trade max drawdown (cumulative equity drawdown if folds
    were independent, non-overlapping trades — an upper bound on
    deployment drawdown since real positions overlap heavily)

Used to spot variants whose high mean is driven by a few heroic folds
and a long tail of losers — which a mean-only view would hide.

Caveats baked into the output:
  - Daily walk-forward folds OVERLAP heavily (a 65d fold[t] holds 64 of
    the same 65 days as fold[t+1]). The trade-by-trade max drawdown is
    therefore an UPPER BOUND on real-position drawdown; a smoothly-
    rebalanced portfolio sees much lower drawdown because adjacent
    overlapping folds cancel one another's noise.
  - 82 folds (65d) and 753 folds (252d) are different sample sizes —
    tail percentiles at 65d are noisier than at 252d.

Run:
  python -m tools.drawdown_audit              # both horizons
  python -m tools.drawdown_audit --horizon 65d
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from datetime import date
from itertools import product
from pathlib import Path

import polars as pl

from tools.backtest_walk_forward import _load_realized, _HORIZON_DAYS
from tools.daily_inference import PRODUCTION_CUTOFFS, _resolve_default_ablation
from tools.robustness_audit import _decile_split_pct, _preload_folds

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent


def _per_fold_lsnet(
    folds: list[dict],
    horizon_days: int,
    decile_pct: float,
    sector_neutral: bool,
    tc_bps: float,
    borrow_bps_yr: float,
) -> list[tuple[date, float]]:
    """Return [(as_of_date, lsnet), ...] for each fold under this variant."""
    tc_drag = (tc_bps / 10000.0) * 2.0
    horizon_years = horizon_days / 252.0
    borrow_drag = (borrow_bps_yr / 10000.0) * horizon_years
    cost = tc_drag + borrow_drag / 2.0

    out: list[tuple[date, float]] = []
    for fold in folds:
        df = fold["df"]
        top, bot = _decile_split_pct(df, sector_neutral, decile_pct)
        if top.height < 1 or bot.height < 1:
            continue
        gross = (float(top["realized"].mean()) - float(bot["realized"].mean())) / 2.0
        # as_of date is not stored on the fold dict; we need to preload it.
        # _preload_folds doesn't track as_of, so we reconstruct from disk order.
        out.append((fold.get("as_of"), gross - cost))
    return out


def _max_consecutive_negative(values: list[float]) -> int:
    """Longest run of consecutive negative values."""
    longest = 0
    current = 0
    for v in values:
        if v < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _trade_by_trade_max_drawdown(values: list[float]) -> float:
    """Max drawdown of the cumulative-equity curve treating each fold as
    an independent multiplicative return. Returns a negative number (the
    biggest peak-to-trough fractional loss; 0.0 means no drawdown)."""
    if not values:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for v in values:
        equity *= (1.0 + v)
        peak = max(peak, equity)
        dd = (equity - peak) / peak  # ≤ 0
        max_dd = min(max_dd, dd)
    return max_dd


def _preload_folds_with_date(pred_files: list[Path], horizon_days: int) -> list[dict]:
    """Wrap _preload_folds to attach as_of date to each fold so we can keep
    chronological order when computing streaks / drawdowns."""
    folds_dated: list[dict] = []
    for f in pred_files:
        date_part = next(p for p in f.parts if p.startswith("date="))
        as_of = date.fromisoformat(date_part.replace("date=", ""))
        preds = pl.read_parquet(f).filter(pl.col("expected_annual_return").is_not_null())
        if preds.is_empty():
            continue
        tickers = preds["ticker"].to_list()
        realized = _load_realized(as_of, horizon_days, tickers)
        if len(realized) < 10:
            continue
        rows = []
        for t, p, lyr in zip(preds["ticker"].to_list(),
                              preds["expected_annual_return"].to_list(),
                              preds["layer"].to_list()):
            if t in realized:
                rows.append({"ticker": t, "layer": lyr,
                             "predicted": float(p),
                             "realized": realized[t]})
        df = pl.DataFrame(rows)
        if df.height < 10:
            continue
        ic = df.select(pl.corr("predicted", "realized")).item()
        folds_dated.append({"as_of": as_of, "df": df,
                            "ic": float(ic) if ic is not None else None})
    folds_dated.sort(key=lambda f: f["as_of"])
    return folds_dated


def _stats_for_variant(values: list[float]) -> dict:
    """Distribution stats for a per-fold LSnet list."""
    if not values:
        return {"n": 0}
    arr = sorted(values)
    n = len(arr)
    def pct(p: float) -> float:
        if n == 1:
            return arr[0]
        i = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return arr[i]
    return {
        "n":          n,
        "mean":       sum(values) / n,
        "p5":         pct(5),
        "p25":        pct(25),
        "p50":        pct(50),
        "p75":        pct(75),
        "p95":        pct(95),
        "min":        arr[0],
        "max":        arr[-1],
        "pos_frac":   sum(1 for v in values if v > 0) / n,
        "wls":        _max_consecutive_negative(values),
        "max_dd":     _trade_by_trade_max_drawdown(values),
    }


def _audit_horizon(horizon: str) -> None:
    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}")
    if ablation != "none":
        base = base / f"ablation={ablation}"
    pred_files = sorted(base.glob(f"date=*/horizon={horizon}/predictions.parquet"))
    if not pred_files:
        print(f"\n[no predictions for horizon={horizon}]\n")
        return
    horizon_days = _HORIZON_DAYS[horizon]

    print()
    print("=" * 100)
    print(f"  DRAWDOWN AUDIT — horizon={horizon}, cutoff={cutoff}, ablation={ablation}")
    print(f"  files on disk: {len(pred_files)}")
    print("=" * 100)

    print(f"  preloading {len(pred_files)} folds (realized-return queries)…", flush=True)
    folds = _preload_folds_with_date(pred_files, horizon_days)
    print(f"  {len(folds)} matured folds preloaded (chronological order).\n")
    if not folds:
        return

    # Compact header
    print(f"  {'variant':<46s} {'n':>4s} {'mean':>8s} {'p5':>8s} "
          f"{'min':>8s} {'pos%':>5s} {'WLS':>4s} {'maxDD':>8s}")
    print("  " + "-" * 95)

    DECILE_PCTS = [5.0, 10.0, 20.0]
    SECTOR_VALS = [True, False]
    TC_BPS_VALS = [15.0]      # production cost only (Phase 2.5 already showed mild cost sensitivity)
    BORROW_VALS = [50.0]

    for pct, sn, tc, br in product(DECILE_PCTS, SECTOR_VALS, TC_BPS_VALS, BORROW_VALS):
        per_fold = _per_fold_lsnet(folds, horizon_days, pct, sn, tc, br)
        values = [v for _, v in per_fold]
        s = _stats_for_variant(values)
        if s["n"] == 0:
            continue
        sn_str = "sector" if sn else "global"
        label = f"pct={pct:>4.0f}%  {sn_str:<6s}  tc={tc:>4.0f}bps  br={br:>3.0f}bps/y"
        marker = "  ← baseline" if (pct, sn, tc, br) == (10.0, True, 15.0, 50.0) else ""
        print(f"  {label:<46s} {s['n']:>4d} {s['mean']:>+8.4f} {s['p5']:>+8.4f} "
              f"{s['min']:>+8.4f} {s['pos_frac']*100:>4.0f}% {s['wls']:>4d} "
              f"{s['max_dd']:>+8.4f}{marker}")

    print()
    print("  Legend:")
    print("    p5      = 5th percentile of per-fold LSnet (typical-bad fold)")
    print("    min     = worst single fold")
    print("    pos%    = fraction of folds with positive LSnet")
    print("    WLS     = worst losing streak (longest consecutive-negative run)")
    print("    maxDD   = trade-by-trade max drawdown of the compounded equity curve")
    print("              (UPPER BOUND on real-position drawdown — overlapping folds")
    print("              in deployment cancel adjacent-fold noise; smoothed portfolio")
    print("              drawdown is materially lower)")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--horizon", choices=list(PRODUCTION_CUTOFFS) + ["all"], default="all",
    )
    args = parser.parse_args()
    horizons = list(PRODUCTION_CUTOFFS) if args.horizon == "all" else [args.horizon]
    for h in horizons:
        _audit_horizon(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
