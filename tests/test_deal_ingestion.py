# tests/test_deal_ingestion.py
import pytest
import polars as pl
from datetime import date as _date

_SYNTHETIC_8K = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 8-K
CURRENT REPORT

Item 1.01 Entry into a Material Definitive Agreement.

On January 15, 2025, Microsoft Corporation entered into a Power Purchase Agreement
with Constellation Energy Group, Inc. for 500 megawatts of nuclear power to be
supplied from the Three Mile Island Unit 1 facility beginning in 2028.
The agreement has a term of 20 years.
"""

def test_parse_8k_extracts_counterparty():
    from ingestion.deal_ingestion import _parse_8k_for_deals
    watchlist = {"MSFT", "CEG", "NVDA"}
    rows = _parse_8k_for_deals(_SYNTHETIC_8K, filer_ticker="MSFT", watchlist=watchlist)
    assert len(rows) >= 1
    counterparties = {r["party_b"] for r in rows}
    assert "CEG" in counterparties

def test_parse_8k_assigns_deal_type():
    from ingestion.deal_ingestion import _parse_8k_for_deals
    watchlist = {"MSFT", "CEG"}
    rows = _parse_8k_for_deals(_SYNTHETIC_8K, filer_ticker="MSFT", watchlist=watchlist)
    types = {r["deal_type"] for r in rows}
    assert "power_purchase_agreement" in types

def test_load_manual_deals():
    from ingestion.deal_ingestion import _load_manual_deals
    from pathlib import Path
    path = Path("data/manual/deals_override.csv")
    df = _load_manual_deals(path)
    assert len(df) >= 20
    assert "party_a" in df.columns
    assert "confidence" in df.columns
    # All manual deals have confidence=1.0
    assert (df["confidence"] == 1.0).all()

def test_build_edges_from_deals():
    from ingestion.deal_ingestion import _build_edges
    deals = pl.DataFrame({
        "party_a": ["MSFT", "MSFT", "GOOGL"],
        "party_b": ["CEG", "CEG", "NEE"],
        "confidence": [1.0, 1.0, 0.7],
        "date": pl.Series([_date(2024, 1, 1), _date(2023, 6, 1), _date(2024, 3, 1)],
                          dtype=pl.Date),
        "deal_type": ["power_purchase_agreement"] * 3,
    })
    edges = _build_edges(deals, as_of=_date(2025, 1, 1))
    # MSFT-CEG pair should have deal_count=2
    msft_ceg = edges.filter(
        (pl.col("ticker_from") == "MSFT") & (pl.col("ticker_to") == "CEG")
    )
    assert len(msft_ceg) == 1
    assert msft_ceg["deal_count"][0] == 2

def test_edge_weight_decays_with_age():
    from ingestion.deal_ingestion import _build_edges
    import math
    # Deal is exactly 2 years old — weight should be confidence * 0.5^2 = 0.25
    deals = pl.DataFrame({
        "party_a": ["NVDA"],
        "party_b": ["TSM"],
        "confidence": [1.0],
        "date": pl.Series([_date(2023, 1, 1)], dtype=pl.Date),
        "deal_type": ["manufacturing_agreement"],
    })
    edges = _build_edges(deals, as_of=_date(2025, 1, 1))
    weight = edges["edge_weight"][0]
    assert abs(weight - 0.25) < 0.05  # approx 2 years decay
