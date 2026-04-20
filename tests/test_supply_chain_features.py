"""Tests for supply chain relationship features."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_close_wide(tickers_daily_ret: dict[str, float], as_of: date, n_days: int = 25) -> pl.DataFrame:
    """
    Build a wide close-price DataFrame where each ticker has a constant daily return.
    n_days rows before as_of, so 20d cumulative return is computable at as_of.
    """
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    data: dict = {"date": dates}
    for ticker, daily_ret in tickers_daily_ret.items():
        prices = [100.0]
        for _ in range(n_days):
            prices.append(prices[-1] * (1 + daily_ret))
        data[ticker] = prices
    return pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))


# ── Momentum tests ────────────────────────────────────────────────────────────

def test_own_layer_momentum_excludes_self():
    """Self's return should not influence own_layer_momentum (peers return ~10.5%)."""
    from processing.supply_chain_features import compute_layer_momentum, _compute_20d_returns

    as_of = date(2025, 6, 1)
    # NVDA in compute layer; AMD, AVGO also compute — give peers 0.5%/day, NVDA 2%/day
    close = _make_close_wide({"NVDA": 0.02, "AMD": 0.005, "AVGO": 0.005}, as_of, n_days=25)
    ret_20d = _compute_20d_returns(close)

    result = compute_layer_momentum("NVDA", as_of, ret_20d, exclude_own_layer=False)
    assert result is not None
    # AMD and AVGO 20d return: (1.005^20) - 1 ≈ 0.1049
    expected = (1.005 ** 20) - 1.0
    assert abs(result - expected) < 0.01, f"Expected ~{expected:.3f}, got {result:.3f}"


def test_ecosystem_momentum_excludes_own_layer():
    """Ecosystem uses only other-layer tickers; own-layer return does not affect it."""
    from processing.supply_chain_features import compute_layer_momentum, _compute_20d_returns

    as_of = date(2025, 6, 1)
    # CEG, NEE, VST in power layer → own-layer peers return 0% (3 power tickers, excl. self = 2 peers ✓)
    # MSFT, AMZN, GOOGL, NVDA, AMD in other layers → ecosystem returns ~22% over 20 days (5 tickers ✓)
    close = _make_close_wide(
        {
            "CEG": 0.0, "NEE": 0.0, "VST": 0.0,          # power layer, 0% daily
            "MSFT": 0.01, "AMZN": 0.01, "GOOGL": 0.01,   # cloud/compute, 1%/day
            "NVDA": 0.01, "AMD": 0.01,                     # compute, 1%/day
        },
        as_of,
        n_days=25,
    )
    ret_20d = _compute_20d_returns(close)

    eco  = compute_layer_momentum("CEG", as_of, ret_20d, exclude_own_layer=True)
    own  = compute_layer_momentum("CEG", as_of, ret_20d, exclude_own_layer=False)

    assert eco is not None
    assert eco > 0.05, f"Ecosystem momentum should reflect other-layer ~22% returns, got {eco}"
    assert own is not None
    assert abs(own) < 0.001, f"Own-layer momentum should be ~0 (NEE, VST return 0%), got {own}"


def test_own_layer_momentum_null_when_insufficient_peers():
    """Returns null when fewer than 2 same-layer peers have data."""
    from processing.supply_chain_features import compute_layer_momentum, _compute_20d_returns

    as_of = date(2025, 6, 1)
    # Only NVDA in the close matrix — no compute-layer peers
    close = _make_close_wide({"NVDA": 0.01}, as_of, n_days=25)
    ret_20d = _compute_20d_returns(close)

    result = compute_layer_momentum("NVDA", as_of, ret_20d, exclude_own_layer=False)
    assert result is None, f"Expected None with 0 peers, got {result}"


# ── Correlation tests ─────────────────────────────────────────────────────────

def test_supply_chain_correlation_range():
    """Correlation result must be in [-1, 1]."""
    from processing.supply_chain_features import (
        compute_supply_chain_correlation, _CORRELATION_PEERS,
    )

    ticker = "NVDA"
    peers = _CORRELATION_PEERS[ticker]
    as_of = date(2025, 6, 1)
    n_days = 65

    # Alternating returns — all peers in phase with NVDA → correlation ≈ 1.0
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    data: dict = {
        "date": dates,
        ticker: [0.01 * ((-1) ** i) for i in range(n_days + 1)],
    }
    for peer in peers:
        data[peer] = [0.005 * ((-1) ** i) for i in range(n_days + 1)]

    ret_1d = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))

    result = compute_supply_chain_correlation(ticker, as_of, ret_1d)
    assert result is not None
    assert -1.0 <= result <= 1.0, f"Correlation out of range: {result}"


def test_supply_chain_correlation_null_when_insufficient_data():
    """Returns null when fewer than 30 overlapping days exist."""
    from processing.supply_chain_features import compute_supply_chain_correlation

    as_of = date(2025, 6, 1)
    # Only 20 days — not enough for the 30-day minimum
    n_days = 20
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    ret_1d = pl.DataFrame({
        "date": dates,
        "NVDA": [0.01] * (n_days + 1),
    }).with_columns(pl.col("date").cast(pl.Date))

    result = compute_supply_chain_correlation("NVDA", as_of, ret_1d)
    assert result is None, f"Expected None with only {n_days} days of data, got {result}"


# ── Earnings tests ────────────────────────────────────────────────────────────

def test_peer_eps_surprise_excludes_self():
    """Ticker's own EPS surprise must not be counted in peer mean."""
    from processing.supply_chain_features import compute_peer_eps_surprise

    as_of = date(2025, 6, 1)
    earnings = pl.DataFrame({
        "ticker":          ["NVDA",                          "AMD"],
        "quarter_end":     [as_of - timedelta(days=30),      as_of - timedelta(days=45)],
        "eps_surprise":    [0.5,                             0.1],
        "eps_surprise_pct":[0.50,                            0.10],
    })

    result = compute_peer_eps_surprise("NVDA", as_of, earnings)
    assert result is not None
    assert abs(result - 0.10) < 0.001, f"Expected 0.10 (AMD only), got {result}"


def test_peer_eps_surprise_uses_90d_window():
    """Earnings older than 90 days are excluded."""
    from processing.supply_chain_features import compute_peer_eps_surprise

    as_of = date(2025, 6, 1)
    earnings = pl.DataFrame({
        "ticker":          ["AMD",                        "AMD"],
        "quarter_end":     [as_of - timedelta(days=89),   as_of - timedelta(days=91)],
        "eps_surprise":    [0.2,                          0.8],
        "eps_surprise_pct":[0.20,                         0.80],
    })

    result = compute_peer_eps_surprise("NVDA", as_of, earnings)
    assert result is not None
    assert abs(result - 0.20) < 0.001, f"Expected 0.20 (only 89-day report), got {result}"


def test_peer_eps_surprise_null_when_no_peers_reported():
    """Returns null (not 0.0) when no peer reported in the 90-day window."""
    from processing.supply_chain_features import compute_peer_eps_surprise

    as_of = date(2025, 6, 1)
    earnings = pl.DataFrame({
        "ticker":          ["AMD"],
        "quarter_end":     [as_of - timedelta(days=120)],   # outside 90d
        "eps_surprise":    [0.5],
        "eps_surprise_pct":[0.50],
    })

    result = compute_peer_eps_surprise("NVDA", as_of, earnings)
    assert result is None, f"Expected None, got {result}"


def test_join_supply_chain_features_adds_four_columns(tmp_path):
    """join_supply_chain_features adds exactly 4 Float64 columns (nulls when no data)."""
    from processing.supply_chain_features import join_supply_chain_features

    ohlcv_dir    = tmp_path / "ohlcv";    ohlcv_dir.mkdir()
    earnings_dir = tmp_path / "earnings"; earnings_dir.mkdir()

    spine = pl.DataFrame({
        "ticker": ["NVDA", "AMD"],
        "date":   [date(2025, 6, 1), date(2025, 6, 1)],
    })

    result = join_supply_chain_features(spine, ohlcv_dir=ohlcv_dir, earnings_dir=earnings_dir)

    for col in ["own_layer_momentum_20d", "ecosystem_momentum_20d",
                "supply_chain_correlation_60d", "peer_eps_surprise_mean"]:
        assert col in result.columns, f"Missing column: {col}"
        assert result[col].dtype == pl.Float64, f"{col} should be Float64"


def test_supply_chain_correlation_uses_usd_matrix_when_fx_dir_provided(tmp_path):
    """When fx_dir is given, correlation is computed on USD-normalised returns."""
    from processing.supply_chain_features import (
        compute_supply_chain_correlation, _CORRELATION_PEERS,
    )
    from processing.fx_features import build_usd_close_matrix

    ticker = "NVDA"
    peers = _CORRELATION_PEERS[ticker]
    as_of = date(2025, 6, 1)
    n_days = 65

    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]
    data: dict = {
        "date":  dates,
        ticker:  [0.01 * ((-1) ** i) for i in range(n_days + 1)],
    }
    for peer in peers:
        data[peer] = [0.005 * ((-1) ** i) for i in range(n_days + 1)]

    # All USD tickers — USD matrix should be identical to local matrix
    ret_1d = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))
    usd_ret_1d = build_usd_close_matrix(ret_1d, tmp_path)  # empty fx_dir → USD tickers unchanged

    result_local = compute_supply_chain_correlation(ticker, as_of, ret_1d)
    result_usd   = compute_supply_chain_correlation(ticker, as_of, usd_ret_1d)

    # For USD tickers both should give the same result
    assert result_local is not None
    assert result_usd is not None
    assert abs(result_local - result_usd) < 0.001
