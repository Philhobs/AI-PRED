"""Daily report: today's top long/short picks plus rolling realized metrics.

Reads, for each production horizon (65d, 252d):
  - The latest prediction parquet under the production ablation subtree
    (per HORIZON_ABLATION_DEFAULTS via tools.daily_inference).
  - The top-N longs (highest expected_annual_return) and shorts (lowest),
    with E5 sign convention already applied at inference time.

Then reads data/scoring_log.parquet (from tools.rolling_score) and prints
the rolling all-time / last-60 / last-20 mean IC + hit + LSnet per horizon.

Suitable for a daily cron tail of the operational pipeline:
  ingest refresh → daily_inference → rolling_score → daily_report

Run:
  python -m tools.daily_report            # default 10 longs / 10 shorts
  python -m tools.daily_report --top-n 20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from tools.daily_inference import PRODUCTION_CUTOFFS, _resolve_default_ablation
from tools.rolling_score import _LOG_PATH as _SCORING_LOG_PATH
from tools.macro_overlay import latest as _macro_latest

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent


def _latest_prediction_path(horizon: str) -> tuple[Path, str] | None:
    """Return (path, as_of_date_str) of the latest production prediction
    for `horizon`, or None if no predictions exist."""
    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}")
    if ablation != "none":
        base = base / f"ablation={ablation}"
    if not base.exists():
        return None
    candidates: list[tuple[Path, str]] = []
    for date_dir in base.glob("date=*"):
        f = date_dir / f"horizon={horizon}" / "predictions.parquet"
        if f.exists():
            candidates.append((f, date_dir.name.replace("date=", "")))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1])
    return candidates[-1]


def _print_picks(horizon: str, top_n: int) -> None:
    info = _latest_prediction_path(horizon)
    if info is None:
        print(f"  {horizon}: no predictions on disk")
        return
    path, as_of = info
    df = pl.read_parquet(path).filter(pl.col("expected_annual_return").is_not_null())
    if df.is_empty():
        print(f"  {horizon}: latest prediction parquet is empty")
        return

    cutoff = PRODUCTION_CUTOFFS[horizon]
    abl = _resolve_default_ablation(horizon)
    print(f"\n── {horizon}  as_of={as_of}  cutoff={cutoff}  ablation={abl}  "
          f"n={df.height} ──")

    df = df.sort("expected_annual_return", descending=True)
    longs = df.head(top_n).select(
        ["rank", "ticker", "layer", "expected_annual_return"]
    )
    shorts = df.tail(top_n).reverse().select(
        ["rank", "ticker", "layer", "expected_annual_return"]
    )

    print(f"\n  TOP {top_n} LONG  (buy — highest expected return):")
    print(f"    {'rank':<5s} {'ticker':<10s} {'layer':<24s} {'pred':>9s}")
    for r in longs.iter_rows(named=True):
        print(f"    {r['rank']:<5d} {r['ticker']:<10s} {r['layer']:<24s} "
              f"{r['expected_annual_return']:+.4f}")

    print(f"\n  TOP {top_n} SHORT  (short — lowest expected return):")
    print(f"    {'rank':<5s} {'ticker':<10s} {'layer':<24s} {'pred':>9s}")
    for r in shorts.iter_rows(named=True):
        print(f"    {r['rank']:<5d} {r['ticker']:<10s} {r['layer']:<24s} "
              f"{r['expected_annual_return']:+.4f}")


def _print_macro_overlay() -> None:
    row = _macro_latest()
    if row is None:
        print("\n[Macro overlay not initialized. Run: python -m tools.macro_overlay]")
        return
    regime = ("risk-OFF" if row['macro_risk_score'] > 0.6 else
              "risk-on"  if row['macro_risk_score'] < 0.4 else "neutral")
    print(f"\n## Macro overlay  (as_of {row['date']})\n")
    print(f"  VIX={row['vix']:.1f}  yield_curve={row['yield_curve_bps']:+.0f}bps  "
          f"DXY={row['dxy']:.1f}")
    print(f"  macro_risk_score = {row['macro_risk_score']:.2f}  ({regime})")
    print(f"  GROSS SCALE      = {row['gross_scale']:.2f}  "
          f"(production sizes positions at {row['gross_scale']*100:.0f}% of nominal)")


def _print_rolling_metrics() -> None:
    if not _SCORING_LOG_PATH.exists():
        print("\n[No scoring log yet. Run: python -m tools.rolling_score]")
        return
    log = pl.read_parquet(_SCORING_LOG_PATH)
    if log.is_empty():
        print("\n[Scoring log empty.]")
        return

    print("\n── Rolling realized performance ──\n")
    print(f"  {'horizon':<7s} {'window':<10s} {'folds':>5s}  {'IC':>8s}  "
          f"{'hit':>5s}  {'LSnet':>9s}")
    print("  " + "-" * 56)
    for h in sorted(log["horizon"].unique().to_list()):
        sub = log.filter(pl.col("horizon") == h).sort("as_of_date")
        n_total = sub.height
        for window_name, window_size in (("all-time", n_total),
                                         ("last 60", 60),
                                         ("last 20", 20)):
            slc = sub.tail(window_size) if window_size < n_total else sub
            if slc.height < 5:
                continue
            ic_mean = slc["ic"].drop_nulls().mean()
            ls_mean = slc["ls_net"].mean()
            hit_mean = slc["hit_rate"].mean()
            print(f"  {h:<7s} {window_name:<10s} {slc.height:>5d}  "
                  f"{ic_mean:+.4f}  {hit_mean:.3f}  {ls_mean:+.4f}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of long / short picks to print per horizon (default 10).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("AI-PRED daily report")
    print("=" * 78)

    _print_macro_overlay()

    print("\n## Latest production picks")
    for h in PRODUCTION_CUTOFFS:
        _print_picks(h, args.top_n)

    _print_rolling_metrics()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
