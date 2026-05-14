"""Daily forward inference for the validated production horizons.

For each of the production-validated horizons (65d, 252d), finds the latest
already-predicted as_of date on disk and runs walk-forward inference for
[latest+1, today]. Uses ablation='auto' so the per-horizon production
default from `models.train.HORIZON_ABLATION_DEFAULTS` is applied:
  65d → no_ai_infra (Phase 2.2 retest winner, +428 bps LSnet)
  252d → deep_only  (Phase E4/2.1/2.2 winner, +444 bps LSnet, sign-flipped)

The script reads/writes under the existing walk-forward prediction tree:
  data/predictions/walkforward/cutoff=<CUTOFF>/ablation=<NAME>/date=*/horizon=<H>/

Idempotent: re-running on the same calendar day is a no-op when predictions
for today are already on disk. Weekends auto-skip to the prior Friday.

Suitable for a daily cron at market-close + ingestion-refresh time, e.g.:
  30 23 * * 1-5  cd /path/to/AI-PRED && python -m tools.daily_inference

Run:
  python -m tools.daily_inference              # both horizons
  python -m tools.daily_inference --horizon 65d
  python -m tools.daily_inference --horizon 252d
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent

# Production training cutoffs per horizon. These are the cutoffs whose trained
# artifacts have been backtested and serve as the production model. To roll the
# production model forward, retrain at a new cutoff, then update this dict.
#
#   65d → 2025-09-30  Phase E1/2.1/2.2 cutoff; ~5 months of matured holdout
#   252d → 2022-04-30  Phase E3/E4/2.1/2.2 cutoff; ~3 years of matured holdout
PRODUCTION_CUTOFFS: dict[str, str] = {
    "65d":  "2025-09-30",
    "252d": "2022-04-30",
}


def _resolve_default_ablation(horizon: str) -> str:
    """Look up the production ablation default for a horizon."""
    from models.train import HORIZON_ABLATION_DEFAULTS
    return HORIZON_ABLATION_DEFAULTS.get(horizon, "none")


def _latest_predicted_date(
    cutoff: str, ablation: str, horizon: str,
) -> date | None:
    """Return the most recent as_of date with a prediction parquet on disk
    for this (cutoff, ablation, horizon) triple, or None if no predictions
    exist yet."""
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}")
    if ablation != "none":
        base = base / f"ablation={ablation}"
    if not base.exists():
        return None
    dates: list[str] = []
    for date_dir in base.glob("date=*"):
        if (date_dir / f"horizon={horizon}" / "predictions.parquet").exists():
            dates.append(date_dir.name.replace("date=", ""))
    if not dates:
        return None
    return date.fromisoformat(max(dates))


def _previous_weekday(d: date) -> date:
    """Return d itself if a weekday, else the most recent Friday on or before d."""
    if d.weekday() <= 4:
        return d
    return d - timedelta(days=d.weekday() - 4)


def _latest_ohlcv_date() -> date | None:
    """Return the latest date present in the NVDA OHLCV calendar, or None.

    Inference can only run for dates with OHLCV data; if today's market data
    hasn't landed yet (or the daily refresh hasn't run), capping at the
    actual data frontier prevents 'No trading days in range' errors from
    walkforward_inference.
    """
    import polars as pl
    ohlcv_dir = _PROJECT_ROOT / "data" / "raw" / "financials" / "ohlcv" / "NVDA"
    files = sorted(ohlcv_dir.glob("*.parquet"))
    if not files:
        return None
    try:
        df = pl.read_parquet(files[-1]).select("date")
        return df["date"].max()
    except Exception:
        return None


def run_daily_for_horizon(horizon: str) -> int:
    """Catch up predictions for one horizon. Returns subprocess exit code."""
    if horizon not in PRODUCTION_CUTOFFS:
        raise ValueError(
            f"No PRODUCTION_CUTOFF entry for horizon={horizon!r}. "
            f"Valid: {list(PRODUCTION_CUTOFFS)}"
        )
    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)

    latest = _latest_predicted_date(cutoff, ablation, horizon)
    # Inference can only run for dates where we already have OHLCV. Cap at
    # the OHLCV frontier so a daily run before the refresh has landed
    # doesn't try to predict beyond the data.
    ohlcv_latest = _latest_ohlcv_date()
    today = _previous_weekday(date.today())
    if ohlcv_latest is not None and today > ohlcv_latest:
        today = _previous_weekday(ohlcv_latest)

    if latest is None:
        # No predictions yet — start from the day after cutoff
        start = date.fromisoformat(cutoff) + timedelta(days=1)
    else:
        start = latest + timedelta(days=1)

    if start > today:
        _LOG.info("[daily] horizon=%s ablation=%s already up to date "
                  "(latest=%s, today=%s)", horizon, ablation, latest, today)
        return 0

    _LOG.info("[daily] horizon=%s ablation=%s: predicting %s → %s",
              horizon, ablation, start, today)
    cmd = [
        sys.executable, "-m", "tools.walkforward_inference",
        "--cutoff",  cutoff,
        "--start",   start.isoformat(),
        "--end",     today.isoformat(),
        "--horizon", horizon,
        "--target",  "raw",
        "--ablation", "auto",
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--horizon",
        choices=list(PRODUCTION_CUTOFFS) + ["all"],
        default="all",
        help="Single horizon (65d / 252d) or 'all' (default).",
    )
    args = parser.parse_args()

    horizons = list(PRODUCTION_CUTOFFS) if args.horizon == "all" else [args.horizon]

    overall_rc = 0
    for h in horizons:
        rc = run_daily_for_horizon(h)
        if rc != 0:
            _LOG.warning("[daily] horizon=%s exited with code %d", h, rc)
            overall_rc = rc
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
