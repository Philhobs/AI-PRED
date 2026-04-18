"""Tests for MW extraction and buyer_type classification in deal_ingestion."""
from __future__ import annotations

import polars as pl
import pytest


def test_mw_extraction_standard_format():
    """'500 MW' in text → deal_mw == 500.0"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("The company agreed to a 500 MW power purchase agreement.") == pytest.approx(500.0)


def test_mw_extraction_with_commas():
    """'1,200 megawatts' → deal_mw == 1200.0"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("a 1,200 megawatt facility in Virginia") == pytest.approx(1200.0)


def test_mw_extraction_gigawatt():
    """'2 GW' → deal_mw == 2000.0"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("signed a 2 GW offtake agreement") == pytest.approx(2000.0)


def test_mw_extraction_returns_none_when_absent():
    """No capacity mention → None"""
    from ingestion.deal_ingestion import _extract_deal_mw
    assert _extract_deal_mw("The company entered into a supply agreement for materials.") is None


def test_buyer_type_hyperscaler():
    """'Microsoft Corporation' → buyer_type == 'hyperscaler'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Microsoft Corporation") == "hyperscaler"


def test_buyer_type_amazon():
    """'Amazon Web Services' → buyer_type == 'hyperscaler'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Amazon Web Services, Inc.") == "hyperscaler"


def test_buyer_type_crypto():
    """'Applied Digital' → buyer_type == 'crypto_miner'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Applied Digital Corporation") == "crypto_miner"


def test_buyer_type_other():
    """Unknown counterparty → buyer_type == 'other'"""
    from ingestion.deal_ingestion import _classify_buyer_type
    assert _classify_buyer_type("Acme Corporation Ltd") == "other"


def test_deals_parquet_has_new_columns(tmp_path):
    """After build_deals(), parquet includes deal_mw and buyer_type columns."""
    from ingestion.deal_ingestion import build_deals

    mock_8k = (
        "Item 1.01. Microsoft Corporation entered into a 300 MW power purchase "
        "agreement with Constellation Energy Group Inc."
    )

    manual_csv = tmp_path / "deals_override.csv"
    manual_csv.write_text("date,party_a,party_b,deal_type,description,confidence\n")

    deals = build_deals(
        filings=[{"text": mock_8k, "date": "2026-01-15", "url": "https://example.com/8k"}],
        manual_csv_path=manual_csv,
        output_path=tmp_path / "deals.parquet",
    )

    assert "deal_mw" in deals.columns, "deal_mw column must be present"
    assert "buyer_type" in deals.columns, "buyer_type column must be present"
