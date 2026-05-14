"""Phase 2.5: robustness audit for the production-default alpha signals.

For each validated horizon (65d / no_ai_infra, 252d / deep_only), re-score
all matured predictions under variants of harness assumptions:

  decile size           : 5%, 10% (baseline), 20%
  sector-neutral        : True (baseline) / False (global decile)
  round-trip tc (bps)   : 5, 15 (baseline), 30
  short-leg borrow/yr   : 0, 50 (baseline), 200 bps

Reports mean IC + mean LSnet per variant. IC is invariant to these knobs
(Pearson corr depends only on (predicted, realized) pairs) — kept in the
table as a sanity check that each variant uses the same fold set. LSnet is
what should move; the question is whether the baseline +428 / +444 bps
survives reasonable perturbations.

Used to test the +428 / +444 bps headline numbers before committing real
capital. If a variant collapses the signal, that's a harness assumption
not to lean on.

Run:
  python -m tools.robustness_audit              # both horizons
  python -m tools.robustness_audit --horizon 65d
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from itertools import product
from pathlib import Path

import polars as pl

from tools.backtest_walk_forward import _load_realized, _HORIZON_DAYS
from tools.daily_inference import PRODUCTION_CUTOFFS, _resolve_default_ablation

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent


def _decile_split_pct(
    df: pl.DataFrame,
    sector_neutral: bool,
    pct: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Like tools.backtest_walk_forward._decile_split but takes a percentage
    (5.0 / 10.0 / 20.0) instead of being hard-coded to a decile."""
    if sector_neutral:
        ranked = df.with_columns(
            pl.col("predicted").rank(method="ordinal", descending=True)
              .over("layer").alias("rank_in_layer")
        ).join(
            df.group_by("layer").len().rename({"len": "n_in_layer"}),
            on="layer",
        )
        cutoff = pl.max_horizontal(
            (pl.col("n_in_layer") * pct / 100).cast(pl.Int64),
            pl.lit(1),
        )
        top = ranked.filter(pl.col("rank_in_layer") <= cutoff)
        bot = ranked.filter(pl.col("rank_in_layer") > pl.col("n_in_layer") - cutoff)
        return (top.drop(["rank_in_layer", "n_in_layer"]),
                bot.drop(["rank_in_layer", "n_in_layer"]))
    n = max(1, int(len(df) * pct / 100))
    sorted_df = df.sort("predicted", descending=True)
    return sorted_df.head(n), sorted_df.tail(n)


def _preload_folds(pred_files: list[Path], horizon_days: int) -> list[dict]:
    """Pre-load matured fold dataframes once so variant scoring doesn't
    re-query DuckDB 54 times per fold."""
    folds: list[dict] = []
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
        folds.append({"df": df, "ic": float(ic) if ic is not None else None})

    return folds


def _score_variant(
    folds: list[dict],
    horizon_days: int,
    decile_pct: float,
    sector_neutral: bool,
    tc_bps: float,
    borrow_bps_yr: float,
) -> dict:
    """Run the harness math under one (decile_pct, sn, tc, borrow) variant
    against pre-loaded fold dataframes."""
    ic_vals: list[float] = []
    lsnet_vals: list[float] = []

    tc_drag = (tc_bps / 10000.0) * 2.0
    horizon_years = horizon_days / 252.0
    borrow_drag = (borrow_bps_yr / 10000.0) * horizon_years
    cost = tc_drag + borrow_drag / 2.0

    for fold in folds:
        df = fold["df"]
        top, bot = _decile_split_pct(df, sector_neutral, decile_pct)
        if top.height < 1 or bot.height < 1:
            continue
        gross = (float(top["realized"].mean()) - float(bot["realized"].mean())) / 2.0
        lsnet_vals.append(gross - cost)
        if fold["ic"] is not None:
            ic_vals.append(fold["ic"])

    n = len(lsnet_vals)
    return {
        "n_folds":    n,
        "mean_ic":    sum(ic_vals) / len(ic_vals) if ic_vals else None,
        "mean_lsnet": sum(lsnet_vals) / n if n else None,
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
    print("=" * 90)
    print(f"  ROBUSTNESS AUDIT — horizon={horizon}, cutoff={cutoff}, ablation={ablation}")
    print(f"  files on disk: {len(pred_files)}")
    print("=" * 90)

    print(f"  preloading {len(pred_files)} folds (realized-return queries)…",
          flush=True)
    folds = _preload_folds(pred_files, horizon_days)
    print(f"  {len(folds)} matured folds preloaded.\n")
    if not folds:
        print("  [no matured folds — nothing to audit]\n")
        return

    baseline = _score_variant(folds, horizon_days, 10.0, True, 15.0, 50.0)
    print("  Production baseline  "
          "(10% decile, sector-neutral, 15 bps tc, 50 bps/yr borrow):")
    print(f"    folds={baseline['n_folds']}  IC={baseline['mean_ic']:+.4f}  "
          f"LSnet={baseline['mean_lsnet']:+.4f}")
    print()
    print(f"  {'variant':<54s}  {'folds':>5s}  {'IC':>8s}  {'LSnet':>9s}  "
          f"{'Δ LSnet':>9s}")
    print("  " + "-" * 88)

    DECILE_PCTS  = [5.0, 10.0, 20.0]
    SECTOR_VALS  = [True, False]
    TC_BPS_VALS  = [5.0, 15.0, 30.0]
    BORROW_VALS  = [0.0, 50.0, 200.0]

    for pct, sn, tc, br in product(DECILE_PCTS, SECTOR_VALS, TC_BPS_VALS, BORROW_VALS):
        r = _score_variant(folds, horizon_days, pct, sn, tc, br)
        if r["mean_lsnet"] is None:
            continue
        sn_str = "sector" if sn else "global"
        label = (f"pct={pct:>4.0f}%  {sn_str:<6s}  "
                 f"tc={tc:>4.0f}bps  borrow={br:>4.0f}bps/yr")
        delta = r["mean_lsnet"] - baseline["mean_lsnet"]
        marker = "  ← baseline" if (pct, sn, tc, br) == (10.0, True, 15.0, 50.0) else ""
        print(f"  {label:<54s}  {r['n_folds']:>5d}  "
              f"{r['mean_ic']:+.4f}  {r['mean_lsnet']:+.4f}  "
              f"{delta:+.4f}{marker}")


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
