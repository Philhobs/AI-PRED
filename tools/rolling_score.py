"""Rolling realized-performance log for the production-default ablations.

For each prediction parquet under the production ablation subtree (65d /
no_ai_infra and 252d / deep_only per HORIZON_ABLATION_DEFAULTS), checks
whether realized OHLCV returns exist for (as_of + horizon trading days);
if so, scores the prediction using the same sector-neutral L/S harness as
tools.backtest_walk_forward and appends a row to data/scoring_log.parquet.

Idempotent — re-running only scores predictions not already in the log.
Prints rolling 20-fold / 60-fold / all-time mean IC + LSnet per horizon.

Run:
  python -m tools.rolling_score                # both horizons
  python -m tools.rolling_score --horizon 65d
  python -m tools.rolling_score --rescore      # wipe log and rebuild
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl

from tools.backtest_walk_forward import _score_one, _HORIZON_DAYS
from tools.daily_inference import PRODUCTION_CUTOFFS, _resolve_default_ablation

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent
_LOG_PATH = _PROJECT_ROOT / "data" / "scoring_log.parquet"

_SCHEMA = {
    "as_of_date":     pl.Date,
    "horizon":        pl.Utf8,
    "cutoff":         pl.Utf8,
    "ablation":       pl.Utf8,
    # Phase 2.8: production portfolio settings persisted alongside metrics.
    # Lets us detect when settings change (auto-rebuild) and lets the log
    # carry context for later analysis.
    "decile_pct":     pl.Float64,
    "sector_neutral": pl.Boolean,
    "n_tickers":      pl.Int32,
    "ic":             pl.Float64,
    "hit_rate":       pl.Float64,
    "ls_net":         pl.Float64,
    "top_decile":     pl.Float64,
    "bot_decile":     pl.Float64,
}


def _load_log() -> pl.DataFrame:
    if not _LOG_PATH.exists():
        return pl.DataFrame(schema=_SCHEMA)
    log = pl.read_parquet(_LOG_PATH)
    # Phase 2.8 schema migration: if the log was written before production
    # portfolio settings became part of the schema, rebuild from scratch
    # so every row carries (decile_pct, sector_neutral) context.
    if {"decile_pct", "sector_neutral"} - set(log.columns):
        _LOG.warning("[rolling_score] log predates Phase 2.8 schema "
                     "(missing decile_pct / sector_neutral) — rebuilding from disk")
        return pl.DataFrame(schema=_SCHEMA)
    return log


def _save_log(log: pl.DataFrame) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.write_parquet(_LOG_PATH, compression="snappy")


def _score_new_for_horizon(horizon: str, existing: pl.DataFrame) -> list[dict]:
    """Score every prediction not already in the log. _score_one returns
    None when realized data hasn't matured yet, so unmatured predictions
    are auto-skipped.

    Production portfolio settings (decile_pct, sector_neutral) come from
    models.train.HORIZON_PORTFOLIO_DEFAULTS — Phase 2.7 showed the harness
    baseline (sector / 10%) gives up most of the deployable alpha.
    """
    from models.train import resolve_portfolio

    cutoff = PRODUCTION_CUTOFFS[horizon]
    ablation = _resolve_default_ablation(horizon)
    decile_pct, sector_neutral = resolve_portfolio(horizon)
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}")
    if ablation != "none":
        base = base / f"ablation={ablation}"
    if not base.exists():
        _LOG.warning("[rolling_score] %s: no prediction tree at %s", horizon, base)
        return []

    # Already-scored under THESE production settings. If decile_pct or
    # sector_neutral changed since the row was written, re-score it.
    already_scored: set[str] = set()
    if not existing.is_empty():
        filt = existing.filter(
            (pl.col("horizon") == horizon)
            & (pl.col("cutoff") == cutoff)
            & (pl.col("ablation") == ablation)
            & (pl.col("decile_pct") == decile_pct)
            & (pl.col("sector_neutral") == sector_neutral)
        )
        already_scored = {d.isoformat() for d in filt["as_of_date"].to_list()}

    horizon_days = _HORIZON_DAYS[horizon]
    pred_files = sorted(base.glob(f"date=*/horizon={horizon}/predictions.parquet"))

    new_rows: list[dict] = []
    n_skipped_already = 0
    n_skipped_unmatured = 0
    for pred_file in pred_files:
        date_part = next(p for p in pred_file.parts if p.startswith("date="))
        as_of_str = date_part.replace("date=", "")
        if as_of_str in already_scored:
            n_skipped_already += 1
            continue

        as_of = date.fromisoformat(as_of_str)
        m = _score_one(pred_file, as_of, horizon_days,
                       sector_neutral=sector_neutral,
                       tc_bps=15.0, borrow_bps_per_year=50.0,
                       decile_pct=decile_pct)
        if m is None:
            n_skipped_unmatured += 1
            continue
        new_rows.append({
            "as_of_date":     as_of,
            "horizon":        horizon,
            "cutoff":         cutoff,
            "ablation":       ablation,
            "decile_pct":     decile_pct,
            "sector_neutral": sector_neutral,
            "n_tickers":      m["n_tickers"],
            "ic":             float(m["ic"]) if m["ic"] is not None else float("nan"),
            "hit_rate":       m["hit_rate"],
            "ls_net":         m["ls_return_net"],
            "top_decile":     m["top_decile_realized"],
            "bot_decile":     m["bot_decile_realized"],
        })

    _LOG.info("[rolling_score] %s: %d new scored (decile_pct=%.0f%%, "
              "sector_neutral=%s) / %d already in log / %d unmatured (skipped)",
              horizon, len(new_rows), decile_pct, sector_neutral,
              n_skipped_already, n_skipped_unmatured)
    return new_rows


def _rolling_summary(log: pl.DataFrame) -> None:
    if log.is_empty():
        print("\nLog empty — no matured predictions yet.")
        return
    print("\n" + "=" * 88)
    print("Rolling realized performance (PRODUCTION portfolio settings per horizon)")
    print("=" * 88)
    print(f"  {'horizon':<7s} {'pct':>5s} {'sn':>5s} {'window':<10s} "
          f"{'folds':>5s}  {'IC':>9s}  {'hit':>5s}  {'LSnet':>9s}")
    print("  " + "-" * 64)
    for h in sorted(log["horizon"].unique().to_list()):
        sub = log.filter(pl.col("horizon") == h).sort("as_of_date")
        pct = float(sub["decile_pct"].tail(1).item())
        sn = bool(sub["sector_neutral"].tail(1).item())
        sn_str = "sect" if sn else "glob"
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
            print(f"  {h:<7s} {pct:>4.0f}% {sn_str:>5s} {window_name:<10s} "
                  f"{slc.height:>5d}  {ic_mean:+.4f}  {hit_mean:.3f}  {ls_mean:+.4f}")
    print("=" * 88)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--horizon", choices=list(PRODUCTION_CUTOFFS) + ["all"], default="all",
    )
    parser.add_argument(
        "--rescore", action="store_true",
        help="Wipe the existing log and re-score everything from scratch.",
    )
    args = parser.parse_args()

    log = _load_log()
    if args.rescore:
        _LOG.info("[rolling_score] --rescore: wiping log (%d rows)", log.height)
        log = pl.DataFrame(schema=_SCHEMA)

    horizons = list(PRODUCTION_CUTOFFS) if args.horizon == "all" else [args.horizon]

    all_new: list[dict] = []
    for h in horizons:
        all_new.extend(_score_new_for_horizon(h, log))

    if all_new:
        new_df = pl.DataFrame(all_new, schema=_SCHEMA)
        log = pl.concat([log, new_df]) if not log.is_empty() else new_df
        log = log.sort(["horizon", "as_of_date"])
        _save_log(log)
        _LOG.info("[rolling_score] log now %d rows → %s", log.height, _LOG_PATH)

    _rolling_summary(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
