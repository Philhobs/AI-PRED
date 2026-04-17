# tests/test_13f_ranking.py
"""Tests for AUM-based 13F filer ranking."""
from __future__ import annotations
from pathlib import Path

import polars as pl
import pytest

from ingestion.sec_13f_ingestion import rank_filers_by_position_count, _prior_quarter


def _write_prior_quarter_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write mock prior-quarter holdings parquet files."""
    quarter_dir = tmp_path / "2024Q4"
    quarter_dir.mkdir(parents=True)
    for row in rows:
        cik = row["cik"]
        df = pl.DataFrame({
            "cik": [cik],
            "quarter": ["2024Q4"],
            "period_end": ["2024-12-31"],
            "cusip": ["123456789"],
            "ticker": ["NVDA"],
            "shares_held": [row["shares"]],
            "value_usd_thousands": [row["value"]],
        })
        df.write_parquet(quarter_dir / f"{cik}.parquet")
    return tmp_path


def _make_index_df(ciks: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "cik": ciks,
        "date_filed": ["2025-02-14"] * len(ciks),
        "filename": [f"edgar/data/{c}/0001.txt" for c in ciks],
    })


def test_rank_uses_prior_quarter_aum(tmp_path):
    """Filers ranked by prior-quarter total position value descending."""
    prior_rows = [
        {"cik": "0000000001", "shares": 100, "value": 1000},
        {"cik": "0000000002", "shares": 500, "value": 5000},
        {"cik": "0000000003", "shares": 900, "value": 9000},
    ]
    prior_dir = _write_prior_quarter_parquet(tmp_path, prior_rows)
    index_df = _make_index_df(["0000000001", "0000000002", "0000000003"])

    result = rank_filers_by_position_count(index_df, top_n=3, prior_quarter_dir=prior_dir / "2024Q4")

    assert result[0] == "0000000003", "Highest-value CIK should rank first"
    assert result[1] == "0000000002"
    assert result[2] == "0000000001"


def test_rank_falls_back_to_cik_age(tmp_path):
    """When prior_quarter_dir is None, falls back to CIK-integer sort (ascending)."""
    index_df = _make_index_df(["0000000300", "0000000100", "0000000200"])
    result = rank_filers_by_position_count(index_df, top_n=3, prior_quarter_dir=None)

    assert result == ["0000000100", "0000000200", "0000000300"], \
        "Fallback should sort CIK ascending (lower = older = larger institution)"


def test_rank_appends_new_ciks_not_in_prior(tmp_path):
    """CIKs in index but absent from prior quarter are appended at end."""
    prior_rows = [
        {"cik": "0000000001", "shares": 100, "value": 1000},
    ]
    prior_dir = _write_prior_quarter_parquet(tmp_path, prior_rows)
    index_df = _make_index_df(["0000000001", "0000000002"])

    result = rank_filers_by_position_count(index_df, top_n=2, prior_quarter_dir=prior_dir / "2024Q4")

    assert result[0] == "0000000001", "Known filer should rank first"
    assert result[1] == "0000000002", "New filer appended at end"


def test_prior_quarter_helper():
    """_prior_quarter returns correct (year, quarter) tuple."""
    assert _prior_quarter(2025, 1) == (2024, 4)
    assert _prior_quarter(2025, 2) == (2025, 1)
    assert _prior_quarter(2025, 4) == (2025, 3)
