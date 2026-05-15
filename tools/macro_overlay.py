"""Phase 2.10: macro risk overlay (yfinance-sourced composite).

Originally designed around Polymarket / Kalshi prediction-market consensus.
Those APIs are geo-blocked from non-US IPs (CFTC restrictions), so this
implementation substitutes publicly observable macro indicators that are
free, accessible from any network, and well-established as regime signals.

Composite score combines three z-score components against a 5-year rolling
baseline:

  1. VIX z-score                  — risk appetite / equity stress
  2. Yield-curve slope (10Y-13W)  — recession proxy (inversion historically
                                    precedes recessions 6-18 months)
  3. DXY z-score                  — dollar / global liquidity tightness

Each component contributes equally to a "risk-on/risk-off" composite, mapped
through a logistic to macro_risk_score ∈ [0, 1]. Higher = more risk-off
regime.

gross_scale = clip(1.0 - macro_risk_score, 0.3, 1.0)
  When macro risk is high, production gross notional scales down. Floor at
  0.3 so the system never goes fully flat — there's always a position to
  measure realized signal against.

This overlay applies to ALL horizons. For the validated 65d/252d signals it
modulates position sizing. For the unvalidated 756d+ horizons it's the
primary risk control — the model produces ranking predictions but the
gross_scale ensures we hold less when macro regime is hostile.

Run:
  python -m tools.macro_overlay              # print current score + log
  python -m tools.macro_overlay --no-log     # just print, don't update log
  python -m tools.macro_overlay --history 30 # show last 30 days of log
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent
_LOG_PATH = _PROJECT_ROOT / "data" / "macro_overlay_log.parquet"

# Tickers used to derive the composite. ^IRX is 13-week T-bill yield × 10
# (yfinance convention), used as a short-rate proxy because daily 2Y data
# isn't reliably available via yfinance.
_TICKERS = {
    "VIX":  "^VIX",
    "T10Y": "^TNX",
    "T13W": "^IRX",
    "DXY":  "DX-Y.NYB",
}

# Lookback for z-score baselines (5 trading-year window)
_BASELINE_YEARS = 5

# Component weights in the composite (currently equal; tune if backtest data
# justifies it later).
_WEIGHTS = {"vix": 1.0, "yield_curve": 1.0, "dxy": 1.0}

# gross_scale floor: never go fully flat. Lets the system keep producing
# realized P&L even in extreme regimes — informative for forward audit.
_GROSS_SCALE_FLOOR = 0.3

_SCHEMA = {
    "date":              pl.Date,
    "vix":               pl.Float64,
    "vix_z":             pl.Float64,
    "t10y":              pl.Float64,
    "t13w":              pl.Float64,
    "yield_curve_bps":   pl.Float64,  # 10Y - 13W in bps (negative = inverted)
    "yield_curve_z":     pl.Float64,
    "dxy":               pl.Float64,
    "dxy_z":             pl.Float64,
    "macro_risk_score":  pl.Float64,  # [0, 1], higher = more risk-off
    "gross_scale":       pl.Float64,  # [0.3, 1.0]
}


def _fetch_series(ticker: str, years: int) -> pl.DataFrame:
    """Pull daily close prices for `ticker` over the last `years` years."""
    end = date.today()
    start = end - timedelta(days=int(years * 365.25) + 10)
    df = yf.download(
        ticker, start=start.isoformat(), end=end.isoformat(),
        progress=False, auto_adjust=False,
    )
    if df.empty:
        return pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64})
    # yfinance returns MultiIndex columns when multi-ticker; single-ticker
    # is usually flat but defensive-extract.
    close_col = df["Close"]
    if hasattr(close_col, "ndim") and close_col.ndim > 1:
        close_col = close_col.iloc[:, 0]
    return pl.DataFrame({
        "date":  [d.date() for d in df.index],
        "value": [float(v) for v in close_col.values],
    }).filter(pl.col("value").is_not_null())


def _zscore(value: float, history: list[float]) -> float | None:
    """Z-score `value` against the historical sample. None if history < 30."""
    if len(history) < 30:
        return None
    n = len(history)
    mean = sum(history) / n
    var = sum((h - mean) ** 2 for h in history) / (n - 1)
    std = math.sqrt(var)
    if std < 1e-9:
        return 0.0
    return (value - mean) / std


def _logistic(x: float) -> float:
    """Logistic mapping z-score sum → [0, 1]. Steepness 0.5 means a
    composite z of ±2 maps to ~0.27 / 0.73."""
    return 1.0 / (1.0 + math.exp(-0.5 * x))


def _z_to_scale(composite_z: float, floor: float = _GROSS_SCALE_FLOOR) -> float:
    """Map composite z-score → gross sizing factor.

    Key property: at z ≤ 0 (benign/normal regime) scale = 1.0 — we don't
    de-size when nothing is wrong. Above z = 0, sizing shrinks
    exponentially with stress. At z = +2 (genuinely stressful regime) the
    scale is ~0.55; at z = +4 (severe stress) scale is at the floor of 0.3.

    Asymmetric on purpose: the cost of under-sizing in a normal regime is
    permanent missed alpha; the cost of over-sizing in a stressed regime
    is a drawdown. The overlay should be a 'reduce when bad' overlay, not
    a 'bet bigger when good' overlay.
    """
    if composite_z <= 0:
        return 1.0
    return max(floor, math.exp(-0.35 * composite_z))


def compute_overlay(as_of: date | None = None) -> dict:
    """Pull the 4 macro tickers and compute the composite for `as_of`
    (default: today). Returns a dict matching _SCHEMA."""
    if as_of is None:
        as_of = date.today()

    series: dict[str, pl.DataFrame] = {}
    for name, ticker in _TICKERS.items():
        df = _fetch_series(ticker, _BASELINE_YEARS)
        if df.is_empty():
            raise RuntimeError(f"No data for {name} ({ticker}) — yfinance down?")
        series[name] = df.filter(pl.col("date") <= as_of).sort("date")

    # Most-recent value per series at or before as_of
    def last_value(name: str) -> tuple[date, float]:
        s = series[name]
        if s.is_empty():
            raise RuntimeError(f"No data for {name} on/before {as_of}")
        return (s["date"][-1], float(s["value"][-1]))

    vix_date, vix = last_value("VIX")
    t10y_date, t10y = last_value("T10Y")
    t13w_date, t13w = last_value("T13W")
    dxy_date, dxy = last_value("DXY")

    # Yield curve (10Y - 13W) in basis points. yfinance ^TNX/^IRX are yield
    # × 10 (e.g. 4.46% comes back as 4.46, so subtract directly and × 100
    # for bps).
    yield_curve_bps = (t10y - t13w) * 100.0

    # Historical baselines for z-scores (last 5 years up to as_of)
    cutoff = as_of - timedelta(days=int(_BASELINE_YEARS * 365.25))
    vix_hist = series["VIX"].filter(pl.col("date") >= cutoff)["value"].to_list()
    dxy_hist = series["DXY"].filter(pl.col("date") >= cutoff)["value"].to_list()
    # Yield curve history: align T10Y and T13W on dates within window
    t10y_h = series["T10Y"].filter(pl.col("date") >= cutoff).rename({"value": "t10y"})
    t13w_h = series["T13W"].filter(pl.col("date") >= cutoff).rename({"value": "t13w"})
    yc_hist_df = t10y_h.join(t13w_h, on="date").with_columns(
        ((pl.col("t10y") - pl.col("t13w")) * 100.0).alias("yc_bps")
    )
    yc_hist = yc_hist_df["yc_bps"].to_list()

    vix_z = _zscore(vix, vix_hist) or 0.0
    dxy_z = _zscore(dxy, dxy_hist) or 0.0
    # For yield curve we want INVERSION to push risk UP — so flip the sign
    # (lower yc means more recession-risk; we want higher z = more risk).
    yc_z_raw = _zscore(yield_curve_bps, yc_hist) or 0.0
    yc_z = -yc_z_raw

    # Composite z (weighted sum of the three component z-scores)
    composite_z = (
        _WEIGHTS["vix"] * vix_z
        + _WEIGHTS["yield_curve"] * yc_z
        + _WEIGHTS["dxy"] * dxy_z
    ) / sum(_WEIGHTS.values())

    macro_risk = _logistic(composite_z)
    gross_scale = _z_to_scale(composite_z)

    return {
        "date":              as_of,
        "vix":               vix,
        "vix_z":             vix_z,
        "t10y":              t10y,
        "t13w":              t13w,
        "yield_curve_bps":   yield_curve_bps,
        "yield_curve_z":     yc_z,
        "dxy":               dxy,
        "dxy_z":             dxy_z,
        "macro_risk_score":  macro_risk,
        "gross_scale":       gross_scale,
    }


def _load_log() -> pl.DataFrame:
    if not _LOG_PATH.exists():
        return pl.DataFrame(schema=_SCHEMA)
    return pl.read_parquet(_LOG_PATH)


def _save_log(log: pl.DataFrame) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.write_parquet(_LOG_PATH, compression="snappy")


def update_log() -> dict:
    """Compute today's overlay and append/replace today's row in the log."""
    today = date.today()
    row = compute_overlay(today)
    log = _load_log()
    # Drop any existing row for today (idempotent reruns)
    if not log.is_empty():
        log = log.filter(pl.col("date") != today)
    new_row = pl.DataFrame([row], schema=_SCHEMA)
    log = pl.concat([log, new_row]).sort("date") if not log.is_empty() else new_row
    _save_log(log)
    _LOG.info("[macro_overlay] log now %d rows → %s", log.height, _LOG_PATH)
    return row


def latest() -> dict | None:
    """Return the most-recent log row as a dict, or None if no log yet."""
    log = _load_log()
    if log.is_empty():
        return None
    last = log.sort("date").tail(1).row(0, named=True)
    return last


def _print_overlay(row: dict) -> None:
    print()
    print("=" * 78)
    print(f"  Macro overlay  ({row['date']})")
    print("=" * 78)
    print(f"  VIX           = {row['vix']:>6.2f}   z = {row['vix_z']:+.2f}")
    print(f"  10Y yield     = {row['t10y']:>6.2f}%")
    print(f"  13W yield     = {row['t13w']:>6.2f}%")
    print(f"  Yield curve   = {row['yield_curve_bps']:>+6.1f} bps  z = {row['yield_curve_z']:+.2f}   "
          f"({'INVERTED ⚠' if row['yield_curve_bps'] < 0 else 'normal'})")
    print(f"  DXY           = {row['dxy']:>6.2f}   z = {row['dxy_z']:+.2f}")
    print()
    print(f"  macro_risk_score = {row['macro_risk_score']:.3f}   "
          f"({'risk-OFF' if row['macro_risk_score'] > 0.6 else 'risk-on' if row['macro_risk_score'] < 0.4 else 'neutral'})")
    print(f"  gross_scale      = {row['gross_scale']:.3f}   "
          f"(production sizes positions at {row['gross_scale']*100:.0f}% of nominal)")
    print("=" * 78)


def _print_history(n_days: int) -> None:
    log = _load_log()
    if log.is_empty():
        print("[macro overlay log empty — run without --history to populate]")
        return
    sub = log.sort("date").tail(n_days)
    print()
    print(f"  Last {sub.height} days of macro overlay:")
    print(f"  {'date':<12s} {'VIX':>6s} {'yc_bps':>8s} {'DXY':>7s} "
          f"{'risk':>5s} {'scale':>6s}")
    print("  " + "-" * 50)
    for r in sub.iter_rows(named=True):
        print(f"  {str(r['date']):<12s} {r['vix']:>6.2f} {r['yield_curve_bps']:>+7.1f} "
              f"{r['dxy']:>7.2f} {r['macro_risk_score']:>5.2f} {r['gross_scale']:>6.3f}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-log", action="store_true",
        help="Print current overlay only; skip updating data/macro_overlay_log.parquet.",
    )
    parser.add_argument(
        "--history", type=int, default=0,
        help="Print last N days of the log instead of computing today's value.",
    )
    args = parser.parse_args()

    if args.history:
        _print_history(args.history)
        return 0

    if args.no_log:
        row = compute_overlay()
    else:
        row = update_log()
    _print_overlay(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
