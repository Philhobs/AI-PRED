"""Multi-date walk-forward inference loop.

Runs `models.inference.run_inference` for every trading day in
[start_date, end_date], for one or both targets, against models trained with
a fixed cutoff. Predictions land under
data/predictions/walkforward/cutoff=<CUTOFF>/date=<D>/horizon=<H>[_excess]/.

Used for the Phase C v2 backtest: train ONCE with cutoff < start_date, predict
for every spine date in the holdout, then score with
tools.backtest_walk_forward.

Usage:
  python -m tools.walkforward_inference --cutoff 2025-09-30 \\
      --start 2025-10-01 --end 2026-03-31 --horizon 5d --target both
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from models.inference import run_inference

_PROJECT_ROOT = Path(__file__).parent.parent
_OHLCV_NVDA = _PROJECT_ROOT / "data" / "raw" / "financials" / "ohlcv" / "NVDA"


def _trading_days_between(start: date, end: date) -> list[date]:
    """Return all trading days in [start, end] using NVDA OHLCV as the calendar."""
    files = sorted(_OHLCV_NVDA.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No NVDA OHLCV under {_OHLCV_NVDA}; can't derive trading calendar")
    df = pl.concat([pl.read_parquet(f).select(["date"]) for f in files]).unique().sort("date")
    return [
        d for d in df["date"].to_list()
        if start <= d <= end
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cutoff",  required=True, help="Walk-forward training cutoff YYYY-MM-DD.")
    parser.add_argument("--start",   required=True, help="First spine date YYYY-MM-DD (must be > cutoff).")
    parser.add_argument("--end",     required=True, help="Last spine date YYYY-MM-DD (typically today - horizon).")
    parser.add_argument("--horizon", default="5d",  help="Single horizon tag, e.g. '5d'.")
    parser.add_argument("--target",  choices=["raw", "excess", "both"], default="both")
    args = parser.parse_args()

    cutoff_date = date.fromisoformat(args.cutoff)
    start_date  = date.fromisoformat(args.start)
    end_date    = date.fromisoformat(args.end)
    if start_date <= cutoff_date:
        raise SystemExit(f"--start ({args.start}) must be AFTER --cutoff ({args.cutoff})")

    trading_days = _trading_days_between(start_date, end_date)
    if not trading_days:
        raise SystemExit(f"No trading days in [{args.start}, {args.end}]")

    targets = ["raw", "excess"] if args.target == "both" else [args.target]

    print(f"Walk-forward inference: cutoff={args.cutoff}, "
          f"horizon={args.horizon}, targets={targets}, "
          f"{len(trading_days)} trading days from {trading_days[0]} to {trading_days[-1]}",
          flush=True)

    n_done = 0
    n_skipped = 0
    n_error = 0
    for d in trading_days:
        for tgt in targets:
            try:
                run_inference(
                    date_str=d.isoformat(),
                    horizon_tag=args.horizon,
                    target=tgt,
                    cutoff=args.cutoff,
                )
                n_done += 1
            except ValueError as exc:
                # weekend or no price data for date
                n_skipped += 1
            except Exception as exc:  # noqa: BLE001 — fail-soft per project convention
                print(f"[walkforward] {d} {tgt}: ERROR {exc}", flush=True)
                n_error += 1
        if (n_done + n_skipped + n_error) % 20 == 0:
            print(f"[walkforward] progress: {n_done} done / {n_skipped} skipped / "
                  f"{n_error} errored", flush=True)

    print(f"\n[walkforward] complete: {n_done} predictions, {n_skipped} skipped, {n_error} errors")
    return 0 if n_done > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
