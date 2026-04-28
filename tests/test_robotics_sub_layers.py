"""Membership and integrity tests for the three robotics sub-layers."""
from __future__ import annotations


def test_industrial_membership_exact():
    from ingestion.ticker_registry import tickers_in_layer
    assert set(tickers_in_layer("robotics_industrial")) == {
        "ROK", "ZBRA", "CGNX", "SYM", "EMR",
        "ABBN.SW", "KGX.DE", "HEXA-B.ST",
        "6954.T", "6506.T", "6861.T",
    }


def test_medical_humanoid_membership_exact():
    from ingestion.ticker_registry import tickers_in_layer
    assert set(tickers_in_layer("robotics_medical_humanoid")) == {
        "ISRG", "TSLA", "1683.HK", "005380.KS",
        # 2026-04-28: surgical robotics expansion
        "SYK", "MDT", "GMED", "PRCT",
    }


def test_mcu_chips_membership_exact():
    from ingestion.ticker_registry import tickers_in_layer
    assert set(tickers_in_layer("robotics_mcu_chips")) == {
        "TXN", "MCHP", "ADI", "6723.T",
        # 2026-04-28: MCU/sensor expansion
        "ON", "NXPI", "MPWR",
    }


def test_no_orphan_robotics_tickers():
    """Every TICKERS_INFO entry whose layer starts with 'robotics_' is in exactly
    one of the three sub-layers."""
    from ingestion.ticker_registry import TICKERS_INFO

    sub_layers = {
        "robotics_industrial",
        "robotics_medical_humanoid",
        "robotics_mcu_chips",
    }
    robotics_entries = [t for t in TICKERS_INFO if t.layer.startswith("robotics")]
    for t in robotics_entries:
        assert t.layer in sub_layers, (
            f"Ticker {t.symbol} has unrecognised robotics layer {t.layer!r}"
        )
    # No legacy flat-robotics layer should remain
    assert all(t.layer != "robotics" for t in TICKERS_INFO)


def test_legacy_robotics_layer_absent():
    """Flat 'robotics' key must not appear in either lookup."""
    from ingestion.ticker_registry import LAYER_IDS, LAYER_LABELS
    assert "robotics" not in LAYER_IDS
    assert "robotics" not in LAYER_LABELS


def test_new_currencies_present_for_humanoid_pillar():
    """1683.HK and 005380.KS introduce HKD and KRW into the registry."""
    from ingestion.ticker_registry import TICKER_CURRENCY
    assert TICKER_CURRENCY["1683.HK"] == "HKD"
    assert TICKER_CURRENCY["005380.KS"] == "KRW"
