"""One-shot calibration sanity check for the LGBM quantile bands.

Reads matured production predictions per horizon, joins to realized
returns, and reports whether the q10 / q90 bands cover realized values
at the rates LightGBM was trained to target (10% and 90%).

This is the lightweight replacement for the full calibration-scoring
plan (no diagrams, no drift monitor, no markdown reports). The whole
point is one number per horizon: are the bands sane?

Important — sign convention:
  At horizons where HORIZON_SIGN_CONVENTION flips the point estimate
  (252d+), the saved confidence_low / confidence_high columns are in
  PRE-FLIP space, while expected_annual_return is post-flip. To compare
  apples-to-apples we re-flip the bands here:

    post-flip q10 = -original_q90
    post-flip q50 =  expected_annual_return  (already flipped at inference)
    post-flip q90 = -original_q10

Targets: q10 coverage ≈ 0.10, q90 coverage ≈ 0.90, q50 coverage ≈ 0.50.
Mean Absolute Calibration Error (MACE) under 0.03 = well-calibrated,
0.03-0.05 acceptable, > 0.05 worth investigating.

Run:
  python -m tools.calibration_check
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

from models.train import HORIZON_SIGN_CONVENTION
from tools.backtest_walk_forward import _load_realized, _HORIZON_DAYS
from tools.daily_inference import PRODUCTION_CUTOFFS, _resolve_default_ablation

_PROJECT_ROOT = Path(__file__).parent.parent


def _pinball(q: float, alpha: float, realized: float) -> float:
    """Standard quantile (pinball) loss at level alpha."""
    diff = realized - q
    return max(alpha * diff, (alpha - 1.0) * diff)


def _load_rows(horizon: str) -> list[dict]:
    """Pull (q10, q50, q90, realized) per ticker per matured as_of for one horizon.

    Bands are re-flipped to match the sign convention applied to q50 at
    inference, so all three quantiles live in the same (post-flip) space.
    """
    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)
    sign = HORIZON_SIGN_CONVENTION.get(horizon, 1)
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}")
    if ablation != "none":
        base = base / f"ablation={ablation}"
    horizon_days = _HORIZON_DAYS[horizon]

    rows: list[dict] = []
    for date_dir in sorted(base.glob("date=*")):
        f = date_dir / f"horizon={horizon}" / "predictions.parquet"
        if not f.exists():
            continue
        as_of = date.fromisoformat(date_dir.name.replace("date=", ""))
        df = pl.read_parquet(f).filter(pl.col("expected_annual_return").is_not_null())
        if df.is_empty() or "confidence_low" not in df.columns:
            continue
        realized = _load_realized(as_of, horizon_days, df["ticker"].to_list())
        if len(realized) < 10:
            continue
        for t, q50, lo, hi in zip(
            df["ticker"].to_list(),
            df["expected_annual_return"].to_list(),
            df["confidence_low"].to_list(),
            df["confidence_high"].to_list(),
        ):
            if t not in realized or lo is None or hi is None:
                continue
            # Re-flip bands so q10 < q50 < q90 in post-flip space.
            if sign == -1:
                q10 = -float(hi)
                q90 = -float(lo)
            else:
                q10 = float(lo)
                q90 = float(hi)
            rows.append({
                "q10": q10, "q50": float(q50), "q90": q90,
                "realized": float(realized[t]),
            })
    return rows


def _report(horizon: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n--- {horizon}: no matured rows on disk ---")
        return
    n = len(rows)
    q10_cov = sum(1 for r in rows if r["realized"] < r["q10"]) / n
    q50_cov = sum(1 for r in rows if r["realized"] < r["q50"]) / n
    q90_cov = sum(1 for r in rows if r["realized"] < r["q90"]) / n
    mace = (abs(q10_cov - 0.10) + abs(q50_cov - 0.50) + abs(q90_cov - 0.90)) / 3.0
    width = sum(r["q90"] - r["q10"] for r in rows) / n
    pb_q10 = sum(_pinball(r["q10"], 0.10, r["realized"]) for r in rows) / n
    pb_q50 = sum(_pinball(r["q50"], 0.50, r["realized"]) for r in rows) / n
    pb_q90 = sum(_pinball(r["q90"], 0.90, r["realized"]) for r in rows) / n

    verdict = ("WELL CALIBRATED" if mace < 0.03 else
               "acceptable"       if mace < 0.05 else
               "INVESTIGATE")

    print(f"\n--- horizon={horizon}, n={n} (ticker × matured as_of) ---")
    print(f"  q10 coverage = {q10_cov:.3f}  (target 0.10, Δ {q10_cov - 0.10:+.3f})")
    print(f"  q50 coverage = {q50_cov:.3f}  (target 0.50, Δ {q50_cov - 0.50:+.3f})")
    print(f"  q90 coverage = {q90_cov:.3f}  (target 0.90, Δ {q90_cov - 0.90:+.3f})")
    print(f"  MACE         = {mace:.3f}  → {verdict}")
    print(f"  mean width   = {width:.3f}  (q90 - q10, in return units)")
    print(f"  pinball loss = q10:{pb_q10:.4f}  q50:{pb_q50:.4f}  q90:{pb_q90:.4f}")


def main() -> int:
    print("=" * 70)
    print("  Quantile calibration sanity check")
    print("=" * 70)
    for h in PRODUCTION_CUTOFFS:
        _report(h, _load_rows(h))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
