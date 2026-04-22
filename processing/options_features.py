"""
Compute 6 ticker-specific options-derived features from raw options chain data.

OPTIONS_FEATURE_COLS:
    iv_rank_30d         — ATM IV percentile vs. 52-week high/low, scaled 0–100
    iv_hv_spread        — Near-term ATM IV minus 30-day realized historical vol (HV30)
    put_call_oi_ratio   — Put OI / Call OI for near-term expiry (≤45 DTE, closest to 30)
    put_call_vol_ratio  — Put volume / Call volume for near-term expiry
    skew_otm            — OTM put IV minus OTM call IV (~5–10% moneyness)
    iv_term_slope       — 30d ATM IV minus 90d ATM IV (positive = inverted = fear)
"""
from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

OPTIONS_FEATURE_COLS: list[str] = [
    "iv_rank_30d",
    "iv_hv_spread",
    "put_call_oi_ratio",
    "put_call_vol_ratio",
    "skew_otm",
    "iv_term_slope",
]

_RAW_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}

_FEATURE_SCHEMA = {"ticker": pl.Utf8, "date": pl.Date, **{c: pl.Float64 for c in OPTIONS_FEATURE_COLS}}


def _empty_features() -> pl.DataFrame:
    return pl.DataFrame(schema=_FEATURE_SCHEMA)


def _select_expiry(
    contracts: pl.DataFrame,
    as_of_date: datetime.date,
    target_dte: int,
    max_dte: int,
) -> pl.DataFrame:
    """Return contracts for the expiry closest to target_dte (DTE in (0, max_dte])."""
    expiries = sorted(contracts["expiry"].unique().to_list())
    candidates = [e for e in expiries if 0 < (e - as_of_date).days <= max_dte]
    if not candidates:
        return pl.DataFrame(schema=_RAW_SCHEMA)
    best = min(candidates, key=lambda e: abs((e - as_of_date).days - target_dte))
    return contracts.filter(pl.col("expiry") == best)


def _atm_strike(contracts: pl.DataFrame, spot: float) -> float | None:
    """Return the strike closest to spot price."""
    strikes = contracts["strike"].unique().to_list()
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s - spot))


def _otm_put_strike(contracts: pl.DataFrame, spot: float) -> float | None:
    """Return the OTM put strike closest to 92.5% of spot (searching 88–97% band)."""
    target = spot * 0.925
    strikes = contracts.filter(pl.col("option_type") == "put")["strike"].unique().to_list()
    candidates = [s for s in strikes if 0.88 * spot <= s <= 0.97 * spot]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s - target))


def _otm_call_strike(contracts: pl.DataFrame, spot: float) -> float | None:
    """Return the OTM call strike closest to 107.5% of spot (searching 103–112% band)."""
    target = spot * 1.075
    strikes = contracts.filter(pl.col("option_type") == "call")["strike"].unique().to_list()
    candidates = [s for s in strikes if 1.03 * spot <= s <= 1.12 * spot]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s - target))


def _get_atm_iv(contracts: pl.DataFrame, atm: float) -> float:
    """Return ATM IV (prefer call; fall back to any contract at that strike)."""
    atm_rows = contracts.filter(pl.col("strike") == atm)
    call_rows = atm_rows.filter(pl.col("option_type") == "call")
    iv_series = call_rows["iv"] if not call_rows.is_empty() else atm_rows["iv"]
    vals = [v for v in iv_series.to_list() if v and v > 0]
    return vals[0] if vals else 0.0


def _compute_hv30(ohlcv_dir: Path, ticker: str, as_of_date: datetime.date) -> float:
    """Compute 30-day realized historical volatility (annualized) from close_price."""
    ticker_dir = ohlcv_dir / ticker
    if not ticker_dir.exists():
        return float("nan")
    files = sorted(ticker_dir.glob("*.parquet"))
    if not files:
        return float("nan")

    df = (
        pl.concat([pl.read_parquet(f) for f in files])
        .filter(pl.col("date") <= pl.lit(as_of_date))
        .sort("date")
        .tail(35)
        .select(["date", "close_price"])
    )
    if len(df) < 2:
        return float("nan")

    closes = df["close_price"].to_list()
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] and closes[i - 1] > 0
    ]
    if len(log_returns) < 2:
        return float("nan")

    log_returns = log_returns[-30:]
    n = len(log_returns)
    # Standard HV30 estimator: sample stdev of log returns × sqrt(252).
    # Mean-centering (ddof=1) removes drift so only dispersion is captured.
    mean_r = sum(log_returns) / n
    return math.sqrt(sum((r - mean_r) ** 2 for r in log_returns) / (n - 1)) * math.sqrt(252)


def _compute_row(
    ticker: str,
    as_of_date: datetime.date,
    contracts: pl.DataFrame,
    atm_iv_history: list[float],
    ohlcv_dir: Path,
) -> dict:
    """Compute all 6 options features for a single (ticker, date).

    atm_iv_history: list of ATM IV values for this ticker for all dates ≤ as_of_date,
                    sorted oldest → newest. Precomputed by build_options_features.
    """
    zero_row: dict = {"ticker": ticker, "date": as_of_date, **{c: 0.0 for c in OPTIONS_FEATURE_COLS}}

    near_term = _select_expiry(contracts, as_of_date, target_dte=30, max_dte=45)
    if near_term.is_empty():
        return zero_row

    # Spot price proxy: median of near-term strikes
    spot_val = near_term["strike"].median()
    if spot_val is None:
        return zero_row
    spot = float(spot_val)
    if spot <= 0:
        return zero_row

    # put_call_oi_ratio
    put_oi = int(near_term.filter(pl.col("option_type") == "put")["oi"].sum() or 0)
    call_oi = int(near_term.filter(pl.col("option_type") == "call")["oi"].sum() or 0)
    put_call_oi = float(put_oi / call_oi) if call_oi > 0 else 0.0

    # put_call_vol_ratio
    put_vol = int(near_term.filter(pl.col("option_type") == "put")["volume"].sum() or 0)
    call_vol = int(near_term.filter(pl.col("option_type") == "call")["volume"].sum() or 0)
    put_call_vol = float(put_vol / call_vol) if call_vol > 0 else 0.0

    # ATM IV (near-term, call-preferred)
    atm = _atm_strike(near_term, spot)
    atm_iv_near = _get_atm_iv(near_term, atm) if atm is not None else 0.0

    # skew_otm: OTM put IV minus OTM call IV
    skew = 0.0
    otm_put = _otm_put_strike(near_term, spot)
    otm_call = _otm_call_strike(near_term, spot)
    if otm_put is not None and otm_call is not None:
        put_iv_vals = [
            v for v in near_term.filter(
                (pl.col("option_type") == "put") & (pl.col("strike") == otm_put)
            )["iv"].to_list() if v and v > 0
        ]
        call_iv_vals = [
            v for v in near_term.filter(
                (pl.col("option_type") == "call") & (pl.col("strike") == otm_call)
            )["iv"].to_list() if v and v > 0
        ]
        put_otm_iv = put_iv_vals[0] if put_iv_vals else 0.0
        call_otm_iv = call_iv_vals[0] if call_iv_vals else 0.0
        skew = put_otm_iv - call_otm_iv

    # iv_term_slope: 30d ATM IV minus 90d ATM IV
    iv_term = 0.0
    mid_term = _select_expiry(contracts, as_of_date, target_dte=90, max_dte=180)
    if not mid_term.is_empty():
        mid_atm = _atm_strike(mid_term, spot)
        if mid_atm is not None:
            atm_iv_mid = _get_atm_iv(mid_term, mid_atm)
            iv_term = atm_iv_near - atm_iv_mid

    # iv_hv_spread: ATM IV minus 30-day realized HV
    hv30 = _compute_hv30(ohlcv_dir, ticker, as_of_date)
    iv_hv = (atm_iv_near - hv30) if not math.isnan(hv30) else 0.0

    # iv_rank_30d: ATM IV percentile over rolling 52-week window
    if len(atm_iv_history) < 30:
        iv_rank = 50.0  # neutral fallback — insufficient history
    else:
        window = atm_iv_history[-252:]
        min_iv = min(window)
        max_iv = max(window)
        if max_iv <= min_iv:
            iv_rank = 50.0
        else:
            iv_rank = max(0.0, min(100.0, (atm_iv_near - min_iv) / (max_iv - min_iv) * 100.0))

    return {
        "ticker": ticker,
        "date": as_of_date,
        "iv_rank_30d": iv_rank,
        "iv_hv_spread": iv_hv,
        "put_call_oi_ratio": put_call_oi,
        "put_call_vol_ratio": put_call_vol,
        "skew_otm": skew,
        "iv_term_slope": iv_term,
    }


def build_options_features(options_dir: Path, ohlcv_dir: Path) -> pl.DataFrame:
    """Aggregate raw options contracts into (ticker, date) feature rows.

    Reads all options parquets once, then processes per (ticker, date).
    iv_rank_30d requires historical ATM IV; pre-built per ticker before the main loop.
    """
    if not options_dir.exists():
        return _empty_features()

    all_files = sorted(options_dir.glob("date=*/*.parquet"))
    if not all_files:
        return _empty_features()

    # Pre-build ATM IV history per ticker (sorted by date, oldest → newest).
    # This avoids O(n²) re-reads inside _compute_row.
    atm_iv_by_ticker: dict[str, list[float]] = {}
    atm_iv_dates: dict[str, list[datetime.date]] = {}

    for date_dir in sorted(options_dir.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        try:
            dir_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        for ticker_file in sorted(date_dir.glob("*.parquet")):
            ticker = ticker_file.stem
            contracts = pl.read_parquet(ticker_file)
            if contracts.is_empty():
                continue
            near_term = _select_expiry(contracts, dir_date, target_dte=30, max_dte=45)
            if near_term.is_empty():
                continue
            spot_val = near_term["strike"].median()
            if spot_val is None:
                continue
            spot = float(spot_val)
            if spot <= 0:
                continue
            atm = _atm_strike(near_term, spot)
            if atm is None:
                continue
            iv = _get_atm_iv(near_term, atm)
            if iv > 0:
                atm_iv_by_ticker.setdefault(ticker, []).append(iv)
                atm_iv_dates.setdefault(ticker, []).append(dir_date)

    # Main loop: compute all 6 features per (ticker, date)
    rows: list[dict] = []
    for date_dir in sorted(options_dir.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        try:
            as_of_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        for ticker_file in sorted(date_dir.glob("*.parquet")):
            ticker = ticker_file.stem
            contracts = pl.read_parquet(ticker_file)
            if contracts.is_empty():
                continue
            # Slice ATM IV history up to (and including) as_of_date
            all_dates = atm_iv_dates.get(ticker, [])
            all_ivs = atm_iv_by_ticker.get(ticker, [])
            history = [iv for d, iv in zip(all_dates, all_ivs) if d <= as_of_date]
            rows.append(_compute_row(ticker, as_of_date, contracts, history, ohlcv_dir))

    if not rows:
        return _empty_features()

    return pl.DataFrame(rows, schema=_FEATURE_SCHEMA)


def join_options_features(df: pl.DataFrame, options_dir: Path, ohlcv_dir: Path) -> pl.DataFrame:
    """Left-join options features to df by (ticker, date). Missing rows zero-fill (not null)."""
    options_df = build_options_features(options_dir, ohlcv_dir)
    result = df.join(options_df, on=["ticker", "date"], how="left")
    fill_exprs = [pl.col(col).fill_null(0.0) for col in OPTIONS_FEATURE_COLS]
    return result.with_columns(fill_exprs)
