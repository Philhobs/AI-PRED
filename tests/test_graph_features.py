# tests/test_graph_features.py
import pytest
import polars as pl
from datetime import date as _date


def _make_edges(pairs: list[tuple]) -> pl.DataFrame:
    """pairs: [(from, to, weight, deal_count, last_date)]"""
    return pl.DataFrame({
        "ticker_from": [p[0] for p in pairs],
        "ticker_to":   [p[1] for p in pairs],
        "edge_weight": pl.Series([float(p[2]) for p in pairs], dtype=pl.Float64),
        "deal_count":  pl.Series([p[3] for p in pairs], dtype=pl.Int32),
        "last_deal_date": pl.Series([p[4] for p in pairs], dtype=pl.Date),
        "edge_types":  ["supply_agreement"] * len(pairs),
    })


def _make_ohlcv(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [r["ticker"] for r in rows],
        "date":   pl.Series([r["date"] for r in rows], dtype=pl.Date),
        "close_price": pl.Series([float(r["close"]) for r in rows], dtype=pl.Float64),
    })


def test_build_graph_nodes_and_edges():
    from processing.graph_features import build_graph
    edges = _make_edges([
        ("NVDA", "TSM", 1.0, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "ENTG", 0.7, 1, _date(2024, 3, 1)),
        ("ENTG", "NVDA", 0.7, 1, _date(2024, 3, 1)),
    ])
    g = build_graph(edges)
    assert "NVDA" in g.nodes
    assert "TSM" in g.nodes
    assert g.has_edge("NVDA", "TSM")
    assert g["NVDA"]["TSM"]["weight"] == pytest.approx(1.0)


def test_partner_momentum_weighted_average():
    """NVDA has two partners: TSM (weight=1.0, +20%) and ENTG (weight=0.5, -10%).
    Expected = (1.0*0.20 + 0.5*(-0.10)) / (1.0 + 0.5) = (0.20 - 0.05) / 1.5 = 0.10
    """
    from processing.graph_features import build_graph, _compute_partner_momentum_30d
    edges = _make_edges([
        ("NVDA", "TSM",  1.0, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "ENTG", 0.5, 1, _date(2024, 1, 1)),
        ("ENTG", "NVDA", 0.5, 1, _date(2024, 1, 1)),
    ])
    ohlcv = _make_ohlcv([
        {"ticker": "TSM",  "date": _date(2024, 1, 14), "close": 120.0},
        {"ticker": "TSM",  "date": _date(2023, 12, 15), "close": 100.0},
        {"ticker": "ENTG", "date": _date(2024, 1, 14), "close": 45.0},
        {"ticker": "ENTG", "date": _date(2023, 12, 15), "close": 50.0},
    ])
    g = build_graph(edges)
    result = _compute_partner_momentum_30d(g, "NVDA", ohlcv, _date(2024, 1, 14))
    assert result == pytest.approx(0.10, abs=1e-4)


def test_partner_momentum_no_partners():
    """Ticker with no edges → None."""
    from processing.graph_features import build_graph, _compute_partner_momentum_30d
    edges = _make_edges([])
    ohlcv = _make_ohlcv([
        {"ticker": "NVDA", "date": _date(2024, 1, 14), "close": 500.0},
    ])
    g = build_graph(edges)
    result = _compute_partner_momentum_30d(g, "NVDA", ohlcv, _date(2024, 1, 14))
    assert result is None


def test_hops_to_hyperscaler_direct():
    from processing.graph_features import build_graph, _compute_hops_to_hyperscaler
    edges = _make_edges([
        ("MSFT", "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "MSFT", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "TSM",  0.9, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 0.9, 1, _date(2024, 1, 1)),
    ])
    g = build_graph(edges)
    # MSFT is a hyperscaler → 1.0
    assert _compute_hops_to_hyperscaler(g, "MSFT") == pytest.approx(1.0)
    # NVDA is 1 hop from MSFT → 1/(1+1) = 0.5
    assert _compute_hops_to_hyperscaler(g, "NVDA") == pytest.approx(0.5)
    # TSM is 2 hops from MSFT (MSFT→NVDA→TSM) → 1/(2+1) = 0.333
    assert _compute_hops_to_hyperscaler(g, "TSM") == pytest.approx(1/3, abs=0.01)


def test_hops_to_hyperscaler_no_path():
    """Ticker with no path to hyperscaler → 0.0."""
    from processing.graph_features import build_graph, _compute_hops_to_hyperscaler
    edges = _make_edges([
        ("FCX", "SCCO", 0.5, 1, _date(2024, 1, 1)),
        ("SCCO", "FCX", 0.5, 1, _date(2024, 1, 1)),
    ])
    g = build_graph(edges)
    assert _compute_hops_to_hyperscaler(g, "FCX") == 0.0


def test_deal_count_90d():
    from processing.graph_features import build_graph, _compute_deal_count_90d
    edges = _make_edges([
        ("NVDA", "TSM",  1.0, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "ENTG", 0.7, 1, _date(2024, 3, 1)),
        ("ENTG", "NVDA", 0.7, 1, _date(2024, 3, 1)),
    ])
    deals = pl.DataFrame({
        "party_a": ["NVDA", "TSM"],
        "party_b": ["TSM",  "ENTG"],
        "date": pl.Series([_date(2024, 3, 15), _date(2024, 3, 20)], dtype=pl.Date),
        "deal_type": ["manufacturing_agreement", "supply_agreement"],
    })
    g = build_graph(edges)
    # NVDA's direct partners are TSM and ENTG.
    # TSM signed a deal on 2024-03-20 (within 90d of 2024-04-15)
    result = _compute_deal_count_90d(g, "NVDA", deals, _date(2024, 4, 15))
    assert result == 1  # TSM's deal with ENTG is within 90d
