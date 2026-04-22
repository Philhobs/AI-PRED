"""Tests for processing/options_features.py."""
import datetime
import math
from pathlib import Path

import polars as pl
import pytest

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


def _write_options(options_dir: Path, rows: list[dict]) -> None:
    """Write raw options rows to Hive-partitioned parquet."""
    df = pl.DataFrame(rows, schema=_RAW_SCHEMA)
    for date_val, date_group in df.group_by("date"):
        date_str = str(date_val[0]) if isinstance(date_val, tuple) else str(date_val)
        for ticker_val, ticker_group in date_group.group_by("ticker"):
            ticker = ticker_val[0] if isinstance(ticker_val, tuple) else ticker_val
            out = options_dir / f"date={date_str}" / f"{ticker}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            ticker_group.write_parquet(out, compression="snappy")


def _write_ohlcv(ohlcv_dir: Path, ticker: str, rows: list[dict]) -> None:
    """Write OHLCV rows to {ohlcv_dir}/{ticker}/all.parquet."""
    df = pl.DataFrame(
        rows,
        schema={"ticker": pl.Utf8, "date": pl.Date, "close_price": pl.Float64},
    )
    out = ohlcv_dir / ticker / "all.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="snappy")


def _near_term_expiry(as_of: datetime.date) -> datetime.date:
    """Return a date ~30 days from as_of."""
    return as_of + datetime.timedelta(days=30)


def _mid_term_expiry(as_of: datetime.date) -> datetime.date:
    """Return a date ~90 days from as_of."""
    return as_of + datetime.timedelta(days=90)


# ── OPTIONS_FEATURE_COLS ──────────────────────────────────────────────────────

def test_options_feature_cols_has_six_elements():
    """OPTIONS_FEATURE_COLS must have exactly 6 elements (spec requirement)."""
    from processing.options_features import OPTIONS_FEATURE_COLS
    assert len(OPTIONS_FEATURE_COLS) == 6
    expected = {
        "iv_rank_30d", "iv_hv_spread", "put_call_oi_ratio",
        "put_call_vol_ratio", "skew_otm", "iv_term_slope",
    }
    assert set(OPTIONS_FEATURE_COLS) == expected


# ── iv_rank_30d ───────────────────────────────────────────────────────────────

def test_iv_rank_in_range(tmp_path):
    """iv_rank_30d is clamped to [0, 100]."""
    as_of = datetime.date(2024, 1, 15)

    # Write 35 days of history so we have enough for iv_rank
    rows = []
    for i in range(35):
        d = as_of - datetime.timedelta(days=34 - i)
        rows.extend([
            {"ticker": "NVDA", "date": d, "expiry": d + datetime.timedelta(days=30),
             "option_type": "call", "strike": 100.0, "iv": 0.20 + i * 0.01, "oi": 100, "volume": 50},
            {"ticker": "NVDA", "date": d, "expiry": d + datetime.timedelta(days=30),
             "option_type": "put", "strike": 100.0, "iv": 0.21 + i * 0.01, "oi": 100, "volume": 50},
        ])

    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"
    _write_options(options_dir, rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    assert "iv_rank_30d" in result.columns
    iv_ranks = result["iv_rank_30d"].to_list()
    for rank in iv_ranks:
        assert 0.0 <= rank <= 100.0, f"iv_rank_30d out of range: {rank}"


def test_iv_rank_fallback_to_50_when_fewer_than_30_days(tmp_path):
    """iv_rank_30d falls back to 50.0 when fewer than 30 days of IV history exist."""
    as_of = datetime.date(2024, 1, 15)

    # Only 5 days of history (< 30 required)
    rows = []
    for i in range(5):
        d = as_of - datetime.timedelta(days=4 - i)
        rows.extend([
            {"ticker": "NVDA", "date": d, "expiry": d + datetime.timedelta(days=30),
             "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 100, "volume": 50},
        ])

    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"
    _write_options(options_dir, rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    latest = result.sort("date").tail(1)
    assert latest["iv_rank_30d"][0] == pytest.approx(50.0), (
        f"Expected 50.0 fallback, got {latest['iv_rank_30d'][0]}"
    )


# ── iv_hv_spread ──────────────────────────────────────────────────────────────

def test_iv_hv_spread_positive_when_iv_exceeds_realized_vol(tmp_path):
    """iv_hv_spread > 0 when ATM IV > 30-day realized HV (vol premium scenario)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    # ATM IV = 0.40 (40%) — well above the realized vol we'll set via flat prices
    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.40, "oi": 500, "volume": 200},
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "put", "strike": 100.0, "iv": 0.41, "oi": 400, "volume": 150},
    ])

    # Flat OHLCV → log returns ≈ 0 → HV30 ≈ 0
    ohlcv_rows = [
        {"ticker": "NVDA", "date": as_of - datetime.timedelta(days=i), "close_price": 100.0}
        for i in range(35)
    ]
    _write_ohlcv(ohlcv_dir, "NVDA", ohlcv_rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    spread = result["iv_hv_spread"][0]
    assert spread > 0.0, f"Expected positive iv_hv_spread (IV > HV), got {spread}"


def test_iv_hv_spread_negative_when_hv_exceeds_iv(tmp_path):
    """iv_hv_spread < 0 when ATM IV < 30-day realized HV (vol discount scenario)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    # Very low ATM IV = 0.05 (5%)
    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.05, "oi": 500, "volume": 200},
    ])

    # Highly volatile OHLCV → HV30 well above 0.05
    rng_prices = [100.0]
    for _ in range(34):
        rng_prices.append(rng_prices[-1] * math.exp(0.05))  # 5% daily move → HV >> 0.05 annualized

    ohlcv_rows = [
        {"ticker": "NVDA", "date": as_of - datetime.timedelta(days=34 - i), "close_price": p}
        for i, p in enumerate(rng_prices)
    ]
    _write_ohlcv(ohlcv_dir, "NVDA", ohlcv_rows)

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    spread = result["iv_hv_spread"][0]
    assert spread < 0.0, f"Expected negative iv_hv_spread (HV > IV), got {spread}"


# ── put_call ratios ───────────────────────────────────────────────────────────

def test_put_call_oi_ratio_correct(tmp_path):
    """put_call_oi_ratio = put_OI / call_OI for near-term expiry."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    _write_options(options_dir, [
        # call OI = 200, put OI = 400 → ratio = 2.0
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 100},
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "put", "strike": 100.0, "iv": 0.32, "oi": 400, "volume": 150},
    ])

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    assert result["put_call_oi_ratio"][0] == pytest.approx(2.0)


# ── skew_otm ─────────────────────────────────────────────────────────────────

def test_skew_otm_positive_when_put_iv_exceeds_call_iv(tmp_path):
    """skew_otm > 0 when OTM put IV > OTM call IV (bearish skew scenario)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"
    expiry = _near_term_expiry(as_of)
    spot = 100.0

    _write_options(options_dir, [
        # ATM
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "call", "strike": spot, "iv": 0.30, "oi": 300, "volume": 100},
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "put", "strike": spot, "iv": 0.31, "oi": 280, "volume": 90},
        # OTM put ~92% of spot
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "put", "strike": 92.0, "iv": 0.45, "oi": 200, "volume": 80},
        # OTM call ~108% of spot
        {"ticker": "NVDA", "date": as_of, "expiry": expiry,
         "option_type": "call", "strike": 108.0, "iv": 0.25, "oi": 150, "volume": 60},
    ])

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    skew = result["skew_otm"][0]
    assert skew > 0.0, f"Expected positive skew (put IV 0.45 > call IV 0.25), got {skew}"


# ── iv_term_slope ─────────────────────────────────────────────────────────────

def test_iv_term_slope_positive_when_near_term_iv_exceeds_mid_term(tmp_path):
    """iv_term_slope > 0 when near-term IV > mid-term IV (inverted term structure)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    _write_options(options_dir, [
        # Near-term ATM IV = 0.50 (fear spike)
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.50, "oi": 300, "volume": 100},
        # Mid-term ATM IV = 0.30 (lower, normal)
        {"ticker": "NVDA", "date": as_of, "expiry": _mid_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 80},
    ])

    from processing.options_features import build_options_features
    result = build_options_features(options_dir, ohlcv_dir)

    slope = result["iv_term_slope"][0]
    assert slope > 0.0, f"Expected positive iv_term_slope (0.50 - 0.30 = 0.20), got {slope}"


# ── join_options_features ─────────────────────────────────────────────────────

def test_join_options_features_adds_all_six_columns(tmp_path):
    """join_options_features adds all 6 OPTIONS_FEATURE_COLS; row count unchanged."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 100},
    ])

    input_df = pl.DataFrame([
        {"ticker": "NVDA", "date": as_of, "return_1d": 0.01},
    ], schema={"ticker": pl.Utf8, "date": pl.Date, "return_1d": pl.Float64})

    from processing.options_features import join_options_features, OPTIONS_FEATURE_COLS
    result = join_options_features(input_df, options_dir, ohlcv_dir)

    assert len(result) == len(input_df), "Row count must not change"
    assert "return_1d" in result.columns, "Original columns must be preserved"
    for col in OPTIONS_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_join_options_features_zero_fills_missing_options_dir(tmp_path):
    """join_options_features zero-fills (not null) when options_dir does not exist."""
    input_df = pl.DataFrame([
        {"ticker": "NVDA", "date": datetime.date(2024, 1, 15), "return_1d": 0.01},
    ], schema={"ticker": pl.Utf8, "date": pl.Date, "return_1d": pl.Float64})

    nonexistent_dir = tmp_path / "options_missing"
    ohlcv_dir = tmp_path / "ohlcv"

    from processing.options_features import join_options_features, OPTIONS_FEATURE_COLS
    result = join_options_features(input_df, nonexistent_dir, ohlcv_dir)

    for col in OPTIONS_FEATURE_COLS:
        assert col in result.columns
        assert result[col][0] == pytest.approx(0.0), f"{col} should be 0.0 (zero-filled), got {result[col][0]}"
        assert result[col][0] is not None, f"{col} should be 0.0 not null"


def test_join_options_features_zero_fills_ticker_absent_from_options(tmp_path):
    """Ticker present in df but absent from options data → zero-filled (not null)."""
    as_of = datetime.date(2024, 1, 15)
    options_dir = tmp_path / "options"
    ohlcv_dir = tmp_path / "ohlcv"

    # Write options for NVDA only
    _write_options(options_dir, [
        {"ticker": "NVDA", "date": as_of, "expiry": _near_term_expiry(as_of),
         "option_type": "call", "strike": 100.0, "iv": 0.30, "oi": 200, "volume": 100},
    ])

    # df includes MSFT which has no options data
    input_df = pl.DataFrame([
        {"ticker": "NVDA", "date": as_of, "return_1d": 0.01},
        {"ticker": "MSFT", "date": as_of, "return_1d": 0.02},  # no options file
    ], schema={"ticker": pl.Utf8, "date": pl.Date, "return_1d": pl.Float64})

    from processing.options_features import join_options_features, OPTIONS_FEATURE_COLS
    result = join_options_features(input_df, options_dir, ohlcv_dir)

    assert len(result) == 2
    msft_row = result.filter(pl.col("ticker") == "MSFT")
    for col in OPTIONS_FEATURE_COLS:
        val = msft_row[col][0]
        assert val == pytest.approx(0.0), f"MSFT {col} should be zero-filled, got {val}"
        assert val is not None
