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

    # Verify actual values from the Microsoft 300 MW PPA mock
    assert deals["buyer_type"][0] == "hyperscaler", f"Expected hyperscaler, got {deals['buyer_type'][0]}"
    assert deals["deal_mw"][0] == pytest.approx(300.0), f"Expected 300.0 MW, got {deals['deal_mw'][0]}"


def test_energy_deal_mw_90d_feature():
    """energy_deal_mw_90d sums MW of energy deals for the ticker in last 90d."""
    from processing.graph_features import _compute_energy_deal_mw_90d
    from datetime import date, timedelta

    as_of = date(2026, 1, 15)
    deals = pl.DataFrame({
        "date": [as_of - timedelta(days=30), as_of - timedelta(days=200)],
        "party_a": ["CEG", "CEG"],
        "party_b": ["MSFT", "AMZN"],
        "deal_type": ["power_purchase_agreement", "power_purchase_agreement"],
        "deal_mw": [500.0, 300.0],
        "buyer_type": ["hyperscaler", "hyperscaler"],
    })

    result = _compute_energy_deal_mw_90d("CEG", deals, as_of)
    assert result == pytest.approx(500.0), "Only the deal within 90d should be counted"


def test_energy_deal_mw_90d_null_treated_as_zero():
    """deal_mw = None is treated as 0 MW (don't penalize deals missing capacity info)."""
    from processing.graph_features import _compute_energy_deal_mw_90d
    from datetime import date, timedelta

    as_of = date(2026, 1, 15)
    deals = pl.DataFrame({
        "date": [as_of - timedelta(days=30)],
        "party_a": ["CEG"],
        "party_b": ["MSFT"],
        "deal_type": ["power_purchase_agreement"],
        "deal_mw": [None],
        "buyer_type": ["hyperscaler"],
    }).with_columns(pl.col("deal_mw").cast(pl.Float64))

    result = _compute_energy_deal_mw_90d("CEG", deals, as_of)
    assert result == pytest.approx(0.0)


def test_hyperscaler_ppa_count_90d_feature():
    """hyperscaler_ppa_count_90d counts PPAs where buyer_type is hyperscaler."""
    from processing.graph_features import _compute_hyperscaler_ppa_count_90d
    from datetime import date, timedelta

    as_of = date(2026, 1, 15)
    deals = pl.DataFrame({
        "date": [
            as_of - timedelta(days=10),   # within window, hyperscaler PPA
            as_of - timedelta(days=20),   # within window, crypto miner PPA
            as_of - timedelta(days=200),  # outside window, hyperscaler PPA
        ],
        "party_a": ["CEG", "CEG", "CEG"],
        "party_b": ["MSFT", "IREN", "AMZN"],
        "deal_type": ["power_purchase_agreement", "power_purchase_agreement", "power_purchase_agreement"],
        "deal_mw": [500.0, 100.0, 300.0],
        "buyer_type": ["hyperscaler", "crypto_miner", "hyperscaler"],
    })

    result = _compute_hyperscaler_ppa_count_90d("CEG", deals, as_of)
    assert result == 1, "Only 1 hyperscaler PPA within 90d window"
