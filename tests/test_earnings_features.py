"""Unit tests for earnings surprise features. No network, no disk."""
import pytest
import polars as pl
from datetime import date as _date


def _make_earn_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "quarter_end": pl.Date,
            "eps_actual": pl.Float64,
            "eps_estimate": pl.Float64,
            "eps_surprise": pl.Float64,
            "eps_surprise_pct": pl.Float64,
        })
    return pl.DataFrame({
        "ticker": [r["ticker"] for r in rows],
        "quarter_end": pl.Series([r["quarter_end"] for r in rows], dtype=pl.Date),
        "eps_actual": pl.Series([float(r.get("eps_actual", 0)) for r in rows], dtype=pl.Float64),
        "eps_estimate": pl.Series([float(r.get("eps_estimate", 0)) for r in rows], dtype=pl.Float64),
        "eps_surprise": pl.Series([float(r["eps_surprise"]) for r in rows], dtype=pl.Float64),
        "eps_surprise_pct": pl.Series([r.get("eps_surprise_pct") for r in rows], dtype=pl.Float64),
    })


# ─────────────────────────────────────────────────────────────────────────────
# _compute_eps_surprise_last
# ─────────────────────────────────────────────────────────────────────────────

def test_eps_surprise_last_most_recent():
    """Returns the most recent quarter's surprise pct, not an older one."""
    from processing.earnings_features import _compute_eps_surprise_last
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2023, 10, 31), "eps_surprise": 0.10, "eps_surprise_pct": 0.10},
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31),  "eps_surprise": 0.05, "eps_surprise_pct": 0.05},
    ])
    result = _compute_eps_surprise_last(earn, "NVDA", _date(2024, 3, 1))
    assert result == pytest.approx(0.05)


def test_eps_surprise_last_excludes_future_quarters():
    """quarter_end after as_of must not be returned."""
    from processing.earnings_features import _compute_eps_surprise_last
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2024, 4, 30), "eps_surprise": 0.20, "eps_surprise_pct": 0.20},
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31), "eps_surprise": 0.05, "eps_surprise_pct": 0.05},
    ])
    result = _compute_eps_surprise_last(earn, "NVDA", _date(2024, 3, 1))
    assert result == pytest.approx(0.05)  # April quarter is future, so Jan is most recent


def test_eps_surprise_last_empty():
    """Empty DataFrame → None."""
    from processing.earnings_features import _compute_eps_surprise_last
    result = _compute_eps_surprise_last(_make_earn_df([]), "NVDA", _date(2024, 3, 1))
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _compute_eps_surprise_mean_4q
# ─────────────────────────────────────────────────────────────────────────────

def test_eps_surprise_mean_4q():
    """Mean of 4 quarters = simple average."""
    from processing.earnings_features import _compute_eps_surprise_mean_4q
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2023, 4, 30), "eps_surprise": 0.01, "eps_surprise_pct": 0.10},
        {"ticker": "NVDA", "quarter_end": _date(2023, 7, 31), "eps_surprise": 0.02, "eps_surprise_pct": 0.20},
        {"ticker": "NVDA", "quarter_end": _date(2023, 10, 31),"eps_surprise": 0.03, "eps_surprise_pct": 0.30},
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31), "eps_surprise": 0.04, "eps_surprise_pct": 0.40},
    ])
    result = _compute_eps_surprise_mean_4q(earn, "NVDA", _date(2024, 3, 1))
    assert result == pytest.approx(0.25, abs=1e-6)  # (0.10+0.20+0.30+0.40)/4


def test_eps_surprise_mean_4q_uses_at_most_4():
    """5 quarters available → uses 4 most recent."""
    from processing.earnings_features import _compute_eps_surprise_mean_4q
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2023, 1, 31), "eps_surprise": 0.01, "eps_surprise_pct": -0.50},  # excluded
        {"ticker": "NVDA", "quarter_end": _date(2023, 4, 30), "eps_surprise": 0.01, "eps_surprise_pct": 0.10},
        {"ticker": "NVDA", "quarter_end": _date(2023, 7, 31), "eps_surprise": 0.02, "eps_surprise_pct": 0.20},
        {"ticker": "NVDA", "quarter_end": _date(2023, 10, 31),"eps_surprise": 0.03, "eps_surprise_pct": 0.30},
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31), "eps_surprise": 0.04, "eps_surprise_pct": 0.40},
    ])
    result = _compute_eps_surprise_mean_4q(earn, "NVDA", _date(2024, 3, 1))
    assert result == pytest.approx(0.25, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_eps_beat_streak
# ─────────────────────────────────────────────────────────────────────────────

def test_eps_beat_streak_three_beats():
    """3 consecutive beats → streak = 3."""
    from processing.earnings_features import _compute_eps_beat_streak
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2023, 4, 30),  "eps_surprise": 0.01, "eps_surprise_pct": 0.10},
        {"ticker": "NVDA", "quarter_end": _date(2023, 7, 31),  "eps_surprise": 0.02, "eps_surprise_pct": 0.20},
        {"ticker": "NVDA", "quarter_end": _date(2023, 10, 31), "eps_surprise": 0.03, "eps_surprise_pct": 0.30},
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31),  "eps_surprise": 0.04, "eps_surprise_pct": 0.40},
    ])
    result = _compute_eps_beat_streak(earn, "NVDA", _date(2024, 3, 1))
    assert result == 4  # all 4 quarters beat


def test_eps_beat_streak_miss_breaks_streak():
    """Miss in second-most-recent quarter stops streak at 1."""
    from processing.earnings_features import _compute_eps_beat_streak
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2023, 10, 31), "eps_surprise": -0.05, "eps_surprise_pct": -0.05},
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31),  "eps_surprise":  0.04, "eps_surprise_pct":  0.04},
    ])
    result = _compute_eps_beat_streak(earn, "NVDA", _date(2024, 3, 1))
    assert result == 1  # Jan beat, Oct miss → streak = 1


def test_eps_beat_streak_zero_on_miss():
    """Most recent quarter is a miss → streak = 0."""
    from processing.earnings_features import _compute_eps_beat_streak
    earn = _make_earn_df([
        {"ticker": "NVDA", "quarter_end": _date(2024, 1, 31), "eps_surprise": -0.10, "eps_surprise_pct": -0.10},
    ])
    result = _compute_eps_beat_streak(earn, "NVDA", _date(2024, 3, 1))
    assert result == 0


def test_eps_beat_streak_empty():
    """Empty DataFrame → 0."""
    from processing.earnings_features import _compute_eps_beat_streak
    result = _compute_eps_beat_streak(_make_earn_df([]), "NVDA", _date(2024, 3, 1))
    assert result == 0


# ─────────────────────────────────────────────────────────────────────────────
# join_earnings_features
# ─────────────────────────────────────────────────────────────────────────────

def test_join_earnings_features_asof(tmp_path):
    """Backward asof join: NVDA gets features, AMD (no parquet) gets nulls."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from processing.earnings_features import join_earnings_features

    nvda_dir = tmp_path / "NVDA"
    nvda_dir.mkdir()
    pa_table = pa.table({
        "ticker": pa.array(["NVDA"]),
        "date": pa.array([_date(2024, 1, 31)], type=pa.date32()),
        "eps_surprise_last": pa.array([0.08]),
        "eps_surprise_mean_4q": pa.array([0.05]),
        "eps_beat_streak": pa.array([4], type=pa.int32()),
    })
    pq.write_table(pa_table, nvda_dir / "earnings_daily.parquet")

    training_df = pl.DataFrame({
        "ticker": ["NVDA", "AMD"],
        "date": pl.Series([_date(2024, 2, 15), _date(2024, 2, 15)], dtype=pl.Date),
    })

    result = join_earnings_features(training_df, tmp_path)

    nvda_row = result.filter(pl.col("ticker") == "NVDA")
    amd_row = result.filter(pl.col("ticker") == "AMD")

    assert nvda_row["eps_surprise_last"][0] == pytest.approx(0.08)
    assert nvda_row["eps_beat_streak"][0] == 4
    assert amd_row["eps_surprise_last"][0] is None


def test_join_earnings_features_no_parquets(tmp_path):
    """No parquet files → all feature columns null, original rows preserved."""
    from processing.earnings_features import join_earnings_features

    training_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": pl.Series([_date(2024, 2, 15)], dtype=pl.Date),
    })
    result = join_earnings_features(training_df, tmp_path)
    assert result["eps_surprise_last"][0] is None
    assert len(result) == 1
