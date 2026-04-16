# tests/test_ownership_features.py
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest


def _make_raw_holdings(tmp_path: Path) -> Path:
    """Create synthetic raw 13F holdings under tmp_path/raw/2023Q4/ for 2 filers."""
    raw_dir = tmp_path / "raw"
    q_dir   = raw_dir / "2023Q4"
    q_dir.mkdir(parents=True)

    period_end = dt.date(2023, 12, 31)

    # Filer A: holds NVDA (5M shares) and MSFT (2M shares)
    filer_a = pl.DataFrame({
        "cik":                 ["0000102909", "0000102909"],
        "quarter":             ["2023Q4",     "2023Q4"],
        "period_end":          [period_end,   period_end],
        "cusip":               ["67066G104",  "594918104"],
        "ticker":              ["NVDA",        "MSFT"],
        "shares_held":         [5_000_000,    2_000_000],
        "value_usd_thousands": [2_500_000,    700_000],
    })

    # Filer B: holds NVDA (3M shares) only
    filer_b = pl.DataFrame({
        "cik":                 ["0001086364"],
        "quarter":             ["2023Q4"],
        "period_end":          [period_end],
        "cusip":               ["67066G104"],
        "ticker":              ["NVDA"],
        "shares_held":         [3_000_000],
        "value_usd_thousands": [1_500_000],
    })

    filer_a.write_parquet(q_dir / "0000102909.parquet", compression="snappy")
    filer_b.write_parquet(q_dir / "0001086364.parquet", compression="snappy")
    return raw_dir


def test_inst_ownership_pct_and_holder_count(tmp_path):
    """inst_ownership_pct = total_shares / shares_outstanding × 100."""
    from processing.ownership_features import compute_ownership_features

    raw_dir = _make_raw_holdings(tmp_path)
    shares_map = {"NVDA": 10_000_000, "MSFT": 50_000_000}

    df = compute_ownership_features(raw_dir, shares_map=shares_map)

    nvda_row = df.filter((pl.col("ticker") == "NVDA") & (pl.col("quarter") == "2023Q4"))
    assert len(nvda_row) == 1
    # 5M + 3M = 8M held, 10M outstanding → 80%
    assert abs(nvda_row["inst_ownership_pct"][0] - 80.0) < 0.01
    assert nvda_row["inst_holder_count"][0] == 2

    msft_row = df.filter((pl.col("ticker") == "MSFT") & (pl.col("quarter") == "2023Q4"))
    assert len(msft_row) == 1
    # 2M held, 50M outstanding → 4%
    assert abs(msft_row["inst_ownership_pct"][0] - 4.0) < 0.01
    assert msft_row["inst_holder_count"][0] == 1


def test_inst_concentration_top10(tmp_path):
    """Top-10 concentration = top-10 filers' shares / total inst shares."""
    from processing.ownership_features import compute_ownership_features

    # NVDA: 2 filers, top-10 covers all → concentration = 1.0
    raw_dir = _make_raw_holdings(tmp_path)
    shares_map = {"NVDA": 10_000_000, "MSFT": 50_000_000}
    df = compute_ownership_features(raw_dir, shares_map=shares_map)

    nvda_row = df.filter((pl.col("ticker") == "NVDA") & (pl.col("quarter") == "2023Q4"))
    assert abs(nvda_row["inst_concentration_top10"][0] - 1.0) < 0.001


def test_inst_net_shares_qoq_and_momentum_2q(tmp_path):
    """QoQ delta and 2Q momentum require 2+ quarters of data."""
    from processing.ownership_features import compute_ownership_features

    raw_dir = tmp_path / "raw"

    # Q3 2023: 6M shares
    q3 = raw_dir / "2023Q3"
    q3.mkdir(parents=True)
    pl.DataFrame({
        "cik": ["0000102909"], "quarter": ["2023Q3"],
        "period_end": [dt.date(2023, 9, 30)], "cusip": ["67066G104"],
        "ticker": ["NVDA"], "shares_held": [6_000_000], "value_usd_thousands": [3_000_000],
    }).write_parquet(q3 / "0000102909.parquet")

    # Q4 2023: 8M shares (filer A=5M + filer B=3M from _make_raw_holdings)
    _make_raw_holdings(tmp_path)  # writes to tmp_path/raw/2023Q4

    # Q1 2024: 9M shares
    q1 = raw_dir / "2024Q1"
    q1.mkdir(parents=True)
    pl.DataFrame({
        "cik": ["0000102909"], "quarter": ["2024Q1"],
        "period_end": [dt.date(2024, 3, 31)], "cusip": ["67066G104"],
        "ticker": ["NVDA"], "shares_held": [9_000_000], "value_usd_thousands": [4_500_000],
    }).write_parquet(q1 / "0000102909.parquet")

    shares_map = {"NVDA": 10_000_000}
    df = compute_ownership_features(raw_dir, shares_map=shares_map)
    nvda = df.filter(pl.col("ticker") == "NVDA").sort("quarter")

    # Q3: no prior quarter → net_shares_qoq is null
    assert nvda.filter(pl.col("quarter") == "2023Q3")["inst_net_shares_qoq"][0] is None

    # Q4: delta = (8M - 6M) / 10M = 0.2
    q4_row = nvda.filter(pl.col("quarter") == "2023Q4")
    assert abs(q4_row["inst_net_shares_qoq"][0] - 0.20) < 0.001

    # Q1: ownership% went from 60% (Q3=6M/10M) to 90% (Q1=9M/10M) → momentum_2q = 90 - 60 = 30
    q1_row = nvda.filter(pl.col("quarter") == "2024Q1")
    assert abs(q1_row["inst_momentum_2q"][0] - 30.0) < 0.1


def test_join_ownership_features_backward_asof(tmp_path):
    """Backward asof join attaches most recent quarterly features to daily rows."""
    from processing.ownership_features import save_ownership_features, join_ownership_features

    features = pl.DataFrame({
        "ticker":                  ["NVDA",               "NVDA"],
        "quarter":                 ["2023Q3",             "2023Q4"],
        "period_end":              [dt.date(2023, 9, 30), dt.date(2023, 12, 31)],
        "inst_ownership_pct":      [60.0,                 80.0],
        "inst_net_shares_qoq":     [None,                 0.2],
        "inst_holder_count":       [1,                    2],
        "inst_concentration_top10":[1.0,                  1.0],
        "inst_momentum_2q":        [None,                 None],
    })

    features_dir = tmp_path / "features"
    save_ownership_features(features, features_dir)

    # Training spine: one row in Q3 (after Sep 30) and one in Q4 (after Dec 31)
    spine = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   [dt.date(2023, 10, 15), dt.date(2024, 1, 10)],
    })
    result = join_ownership_features(spine, features_dir)

    # 2023-10-15 is after Q3 end (Sep 30) but before Q4 end → gets Q3 features (ownership=60%)
    oct_row = result.filter(pl.col("date") == dt.date(2023, 10, 15))
    assert abs(oct_row["inst_ownership_pct"][0] - 60.0) < 0.01

    # 2024-01-10 is after Q4 end (Dec 31) → gets Q4 features (ownership=80%)
    jan_row = result.filter(pl.col("date") == dt.date(2024, 1, 10))
    assert abs(jan_row["inst_ownership_pct"][0] - 80.0) < 0.01
