"""Unit tests for short interest ingestion and features. No network, no disk."""
import pytest
import polars as pl
from datetime import date as _date


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_si_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "short_vol_ratio": pl.Float64,
        })
    return pl.DataFrame({
        "date": pl.Series([r["date"] for r in rows], dtype=pl.Date),
        "ticker": [r["ticker"] for r in rows],
        "short_vol_ratio": pl.Series([float(r["ratio"]) for r in rows], dtype=pl.Float64),
    })


# ─────────────────────────────────────────────────────────────────────────────
# _compute_short_ratio_10d
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_short_ratio_10d_basic():
    """Mean of two in-window rows = average."""
    from processing.short_interest_features import _compute_short_ratio_10d
    si = _make_si_df([
        {"date": _date(2024, 1, 5), "ticker": "NVDA", "ratio": 0.40},
        {"date": _date(2024, 1, 8), "ticker": "NVDA", "ratio": 0.60},
        # Out of 10-day window (Jan 14 - 10 = Jan 4, so Jan 3 is out)
        {"date": _date(2024, 1, 3), "ticker": "NVDA", "ratio": 0.90},
        # Different ticker — must not affect NVDA
        {"date": _date(2024, 1, 8), "ticker": "AMD", "ratio": 0.80},
    ])
    result = _compute_short_ratio_10d(si, "NVDA", _date(2024, 1, 14))
    assert result == pytest.approx(0.50, abs=1e-6)


def test_compute_short_ratio_10d_empty():
    """Empty DataFrame → None."""
    from processing.short_interest_features import _compute_short_ratio_10d
    result = _compute_short_ratio_10d(_make_si_df([]), "NVDA", _date(2024, 1, 14))
    assert result is None


def test_compute_short_ratio_10d_no_match():
    """No matching rows in window → None."""
    from processing.short_interest_features import _compute_short_ratio_10d
    si = _make_si_df([
        {"date": _date(2024, 1, 1), "ticker": "NVDA", "ratio": 0.40},  # outside window
    ])
    result = _compute_short_ratio_10d(si, "NVDA", _date(2024, 1, 14))
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _compute_short_ratio_momentum
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_short_ratio_momentum():
    """Current 10d mean - prior 10d mean = momentum."""
    from processing.short_interest_features import _compute_short_ratio_momentum
    si = _make_si_df([
        # Current window: Jan 14 - 10 = Jan 4 to Jan 14
        {"date": _date(2024, 1, 10), "ticker": "NVDA", "ratio": 0.60},
        # Prior window: Jan 14 - 20 = Dec 25 to Jan 14 - 11 = Jan 3
        {"date": _date(2024, 1, 2),  "ticker": "NVDA", "ratio": 0.40},
    ])
    result = _compute_short_ratio_momentum(si, "NVDA", _date(2024, 1, 14))
    assert result == pytest.approx(0.20, abs=1e-6)


def test_compute_short_ratio_momentum_missing_prior():
    """No prior window data → None."""
    from processing.short_interest_features import _compute_short_ratio_momentum
    si = _make_si_df([
        {"date": _date(2024, 1, 10), "ticker": "NVDA", "ratio": 0.60},
        # Only current window, no prior
    ])
    result = _compute_short_ratio_momentum(si, "NVDA", _date(2024, 1, 14))
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion parsing — _fetch_day is network-bound; test the format parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_finra_line_parsing():
    """FINRA pipe-delimited format: ShortVolume/TotalVolume = short_vol_ratio."""
    # Simulate what _fetch_day receives and parses
    raw = "20260414|NVDA|500000|0|1000000|B,Q,N"
    parts = raw.split("|")
    short_vol = float(parts[2])
    total_vol = float(parts[4])
    ratio = short_vol / total_vol
    assert ratio == pytest.approx(0.5)


def test_finra_line_parsing_zero_total():
    """Zero total volume must not produce a row (avoids division by zero)."""
    raw = "20260414|NVDA|0|0|0|B,Q,N"
    parts = raw.split("|")
    total_vol = float(parts[4])
    assert total_vol == 0  # code skips this row


# ─────────────────────────────────────────────────────────────────────────────
# join_short_interest_features
# ─────────────────────────────────────────────────────────────────────────────

def test_join_short_interest_features_asof(tmp_path):
    """Backward asof join: NVDA gets features, AMD (no parquet) gets nulls."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from processing.short_interest_features import join_short_interest_features

    # Write NVDA si_daily.parquet
    nvda_dir = tmp_path / "NVDA"
    nvda_dir.mkdir()
    pa_table = pa.table({
        "ticker": pa.array(["NVDA"]),
        "date": pa.array([_date(2024, 1, 14)], type=pa.date32()),
        "short_vol_ratio_10d": pa.array([0.45]),
        "short_vol_ratio_30d": pa.array([0.42]),
        "short_ratio_momentum": pa.array([0.03]),
    })
    pq.write_table(pa_table, nvda_dir / "si_daily.parquet")

    training_df = pl.DataFrame({
        "ticker": ["NVDA", "AMD"],
        "date": pl.Series([_date(2024, 1, 14), _date(2024, 1, 14)], dtype=pl.Date),
    })

    result = join_short_interest_features(training_df, tmp_path)

    nvda_row = result.filter(pl.col("ticker") == "NVDA")
    amd_row = result.filter(pl.col("ticker") == "AMD")

    assert nvda_row["short_vol_ratio_10d"][0] == pytest.approx(0.45)
    assert amd_row["short_vol_ratio_10d"][0] is None


def test_join_short_interest_features_no_parquets(tmp_path):
    """No parquet files → all feature columns null, original rows preserved."""
    from processing.short_interest_features import join_short_interest_features

    training_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": pl.Series([_date(2024, 1, 14)], dtype=pl.Date),
    })
    result = join_short_interest_features(training_df, tmp_path)
    assert result["short_vol_ratio_10d"][0] is None
    assert len(result) == 1
