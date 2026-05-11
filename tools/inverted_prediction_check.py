"""Phase E4: inverted-prediction sanity check.

For a given (cutoff, horizon, ablation) prediction subtree, computes IC and
sector-neutral L/S net return with the ORIGINAL predictions and again with
the SIGN-FLIPPED predictions. Useful when a model produces negative IC at
some horizon — the test reveals whether the signal is real but
sign-inverted (e.g., cross-sectional momentum reversal at long horizons)
or just noise.

If NEGATED IC > +ic_threshold and NEGATED LSnet > 0: model has learned a
real cross-sectional pattern but its sign convention is wrong for this
horizon. Productionize by inverting `expected_annual_return` at inference
time for the affected horizon.

Run:
  python -m tools.inverted_prediction_check --cutoff 2022-04-30 --horizon 252d
  python -m tools.inverted_prediction_check --cutoff 2022-04-30 --horizon 252d --ablation deep_only
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

from tools.backtest_walk_forward import _load_realized, _decile_split, _HORIZON_DAYS


_PROJECT_ROOT = Path(__file__).parent.parent
_TC_BPS = 15.0
_BORROW_BPS_YR = 50.0


def _score_with_sign(pred_files: list[Path], horizon_days: int, invert: bool) -> dict:
    """Compute mean IC + LSnet across prediction files, optionally negating
    expected_annual_return before ranking."""
    ic_vals: list[float] = []
    lsnet_vals: list[float] = []
    hits: list[float] = []

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
                signed_p = -float(p) if invert else float(p)
                rows.append({"ticker": t, "layer": lyr,
                             "predicted": signed_p, "realized": realized[t]})
        df = pl.DataFrame(rows)
        if df.height < 10:
            continue

        ic = df.select(pl.corr("predicted", "realized")).item()
        hit = df.filter(pl.col("predicted").sign() == pl.col("realized").sign()).height / df.height
        top_df, bot_df = _decile_split(df, sector_neutral=True)
        top_r = float(top_df["realized"].mean())
        bot_r = float(bot_df["realized"].mean())
        gross = (top_r - bot_r) / 2.0

        tc_drag = (_TC_BPS / 10000.0) * 2.0
        horizon_years = horizon_days / 252.0
        borrow_drag = (_BORROW_BPS_YR / 10000.0) * horizon_years
        cost = tc_drag + borrow_drag / 2.0
        lsnet = gross - cost

        if ic is not None:
            ic_vals.append(float(ic))
        lsnet_vals.append(lsnet)
        hits.append(hit)

    n = len(lsnet_vals)
    return {
        "folds":      n,
        "mean_ic":    sum(ic_vals) / len(ic_vals) if ic_vals else None,
        "mean_hit":   sum(hits) / n if n else None,
        "mean_lsnet": sum(lsnet_vals) / n if n else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cutoff", required=True, help="Walk-forward cutoff (YYYY-MM-DD)")
    p.add_argument("--horizon", required=True, help="Horizon tag e.g. '252d'")
    p.add_argument("--ablation", default=None,
                   help="Ablation subtree (e.g. 'deep_only', 'no_ai_infra'). "
                        "Default: scan both root and all ablation= subtrees.")
    p.add_argument("--ic-threshold", type=float, default=0.005,
                   help="Verdict threshold on |delta_IC|. Default: 0.005")
    args = p.parse_args()

    if args.horizon not in _HORIZON_DAYS:
        raise SystemExit(f"Unknown horizon {args.horizon}")
    horizon_days = _HORIZON_DAYS[args.horizon]
    root = _PROJECT_ROOT / "data" / "predictions" / "walkforward" / f"cutoff={args.cutoff}"

    if args.ablation:
        subtrees = [(args.ablation, f"ablation={args.ablation}/date=*/horizon={args.horizon}/predictions.parquet")]
    else:
        subtrees = [
            ("FULL (no ablation)", f"date=*/horizon={args.horizon}/predictions.parquet"),
        ]
        # Include any ablation= subtree present
        for sub in sorted(root.glob("ablation=*")):
            tag = sub.name.replace("ablation=", "")
            subtrees.append((f"ablation={tag}",
                             f"ablation={tag}/date=*/horizon={args.horizon}/predictions.parquet"))

    print("=" * 88)
    print(f"PHASE E4: inverted-prediction sanity check  "
          f"cutoff={args.cutoff} horizon={args.horizon}")
    print("=" * 88)

    for label, glob in subtrees:
        files = sorted(root.glob(glob))
        if not files:
            print(f"\n--- {label}: NO predictions found ---")
            continue
        print(f"\n--- {label}: {len(files)} files ---")
        for invert, name in [(False, "ORIGINAL"), (True, "NEGATED")]:
            s = _score_with_sign(files, horizon_days, invert)
            ic_str = f"{s['mean_ic']:+.4f}" if s['mean_ic'] is not None else "—"
            ls_str = f"{s['mean_lsnet']:+.4f}" if s['mean_lsnet'] is not None else "—"
            hit_str = f"{s['mean_hit']:.3f}" if s['mean_hit'] is not None else "—"
            print(f"  {name:8s}  folds={s['folds']:>4d}  IC={ic_str}  hit={hit_str}  LSnet={ls_str}")

    print("\n" + "=" * 88)
    print("Verdict criteria:")
    print(f"  NEGATED IC > +{args.ic_threshold} and NEGATED LSnet > 0 → real signal, wrong sign.")
    print(f"      Productionize: invert expected_annual_return for this horizon at inference.")
    print(f"  NEGATED IC near 0 → genuine noise/reversal at this horizon, no easy fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
