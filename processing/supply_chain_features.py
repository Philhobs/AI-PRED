"""
Supply chain relationship features — peer layer momentum, ecosystem momentum,
supply chain correlation, and peer earnings contagion.

Produces 4 features per ticker per date:
  own_layer_momentum_20d       — avg 20-trading-day return of same-layer peers (excl. self)
  ecosystem_momentum_20d       — avg 20-trading-day return of all other-layer tickers
  supply_chain_correlation_60d — mean 60-day rolling Pearson corr with 20 fixed other-layer peers
  peer_eps_surprise_mean       — mean EPS surprise pct of same-layer peers in last 90 days

Called by models/train.py via join_supply_chain_features().
"""
from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ingestion.ticker_registry import TICKER_LAYERS, tickers_in_layer, layers as all_layers

_LOG = logging.getLogger(__name__)

# ── Layer maps (built at module load time) ──────────────────────────────────

_LAYER_MAP: dict[str, str] = dict(TICKER_LAYERS)  # ticker → layer name

_LAYER_TICKERS: dict[str, list[str]] = {
    layer: tickers_in_layer(layer) for layer in all_layers()
}

_ALL_TICKERS: list[str] = sorted(_LAYER_MAP.keys())


# ── Correlation peers (fixed at module load, seed=42) ──────────────────────

def _build_correlation_peers() -> dict[str, list[str]]:
    rng = random.Random(42)
    peers: dict[str, list[str]] = {}
    for ticker, layer in _LAYER_MAP.items():
        other = [t for t in _ALL_TICKERS if _LAYER_MAP[t] != layer]
        peers[ticker] = rng.sample(other, min(20, len(other)))
    return peers


_CORRELATION_PEERS: dict[str, list[str]] = _build_correlation_peers()


# ── Returns matrix helpers ──────────────────────────────────────────────────

def _build_close_matrix(
    ohlcv_dir: Path,
    min_date: date,
    max_date: date,
    extra_days: int = 100,
) -> pl.DataFrame:
    """
    Wide close-price matrix: date × ticker columns.
    Loads from (min_date - extra_days) to max_date for rolling window history.
    """
    load_start = min_date - timedelta(days=extra_days)

    frames = []
    for ticker in _ALL_TICKERS:
        ticker_dir = ohlcv_dir / ticker
        if not ticker_dir.exists():
            continue
        year_files = sorted(
            f for f in ticker_dir.glob("*.parquet")
            if int(f.stem) >= load_start.year
        )
        if not year_files:
            continue
        df = pl.concat([pl.read_parquet(f) for f in year_files])
        df = (
            df.filter((pl.col("date") >= load_start) & (pl.col("date") <= max_date))
            .sort("date")
            .select(["date", pl.col("close_price").alias(ticker)])
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="date", how="full", coalesce=True)
    return result.sort("date")


def _compute_20d_returns(close_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Wide DataFrame of 20-trading-day cumulative returns.
    ret_20d[ticker][d] = close[d] / close[d-20 rows] - 1
    """
    ticker_cols = [c for c in close_wide.columns if c != "date"]
    return close_wide.select(
        ["date"] + [
            (pl.col(t) / pl.col(t).shift(20) - 1).alias(t)
            for t in ticker_cols
        ]
    )


# ── Feature computation functions ──────────────────────────────────────────

def compute_layer_momentum(
    ticker: str,
    as_of: date,
    ret_20d_wide: pl.DataFrame,
    exclude_own_layer: bool = False,
) -> float | None:
    """
    Compute mean 20-trading-day return of:
      - same-layer peers (exclude_own_layer=False) — excludes self
      - all other-layer tickers (exclude_own_layer=True)

    Returns None when fewer than 2 (own-layer) or 5 (ecosystem) valid values exist.
    """
    layer = _LAYER_MAP.get(ticker)
    if layer is None:
        return None

    if exclude_own_layer:
        target_tickers = [t for t in _ALL_TICKERS if _LAYER_MAP.get(t) != layer]
        min_peers = 5
    else:
        target_tickers = [t for t in _LAYER_TICKERS.get(layer, []) if t != ticker]
        min_peers = 2

    row = ret_20d_wide.filter(pl.col("date") == as_of)
    if row.is_empty():
        return None

    vals = []
    for t in target_tickers:
        if t not in row.columns:
            continue
        v = row[t][0]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))

    return float(np.mean(vals)) if len(vals) >= min_peers else None


def compute_supply_chain_correlation(
    ticker: str,
    as_of: date,
    ret_1d_wide: pl.DataFrame,
) -> float | None:
    """
    Mean rolling 60-trading-day Pearson correlation between ticker and its 20
    fixed other-layer peers from _CORRELATION_PEERS.

    Returns None when fewer than 5 peers have 30+ overlapping return days.
    """
    peers = _CORRELATION_PEERS.get(ticker, [])
    if not peers or ticker not in ret_1d_wide.columns:
        return None

    all_dates = ret_1d_wide["date"].to_list()
    # Find last row index where date <= as_of
    idx = next(
        (i for i, d in reversed(list(enumerate(all_dates))) if d <= as_of),
        None,
    )
    if idx is None or idx < 1:
        return None

    start_idx = max(0, idx - 59)
    window = ret_1d_wide.slice(start_idx, idx - start_idx + 1)

    if window.height < 30:
        return None

    t_returns = window[ticker].to_numpy().astype(float)

    valid_corrs = []
    for peer in peers:
        if peer not in window.columns:
            continue
        p_returns = window[peer].to_numpy().astype(float)
        mask = ~(np.isnan(t_returns) | np.isnan(p_returns))
        if mask.sum() < 30:
            continue
        corr = float(np.corrcoef(t_returns[mask], p_returns[mask])[0, 1])
        if not np.isnan(corr):
            valid_corrs.append(corr)

    return float(np.mean(valid_corrs)) if len(valid_corrs) >= 5 else None


def compute_peer_eps_surprise(
    ticker: str,
    as_of: date,
    earnings_df: pl.DataFrame,
) -> float | None:
    """
    Mean EPS surprise pct of same-layer peers with earnings reported in last 90 days.

    Returns None (not 0) when no qualifying peer earnings exist — absence of
    earnings data is not a neutral signal.
    """
    layer = _LAYER_MAP.get(ticker)
    if layer is None:
        return None

    peers = [t for t in _LAYER_TICKERS.get(layer, []) if t != ticker]
    if not peers:
        return None

    if earnings_df.is_empty():
        return None

    window_start = as_of - timedelta(days=90)
    peer_earnings = earnings_df.filter(
        pl.col("ticker").is_in(peers)
        & (pl.col("quarter_end") >= window_start)
        & (pl.col("quarter_end") <= as_of)
    )

    if peer_earnings.is_empty():
        return None

    vals = peer_earnings["eps_surprise_pct"].drop_nulls().to_list()
    return float(np.mean(vals)) if vals else None


# ── Public join function ────────────────────────────────────────────────────

def join_supply_chain_features(
    df: pl.DataFrame,
    ohlcv_dir: Path | None = None,
    earnings_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Add own_layer_momentum_20d, ecosystem_momentum_20d,
    supply_chain_correlation_60d, peer_eps_surprise_mean to training spine.

    Args:
        df: Training spine with columns [ticker, date, ...].
        ohlcv_dir: Path to data/raw/financials/ohlcv/ (default: resolved from __file__).
        earnings_dir: Path to data/raw/financials/earnings/ (default: resolved from __file__).

    Returns df with 4 new Float64 columns. Missing data → null (not 0).
    """
    _ROOT = Path(__file__).parent.parent
    if ohlcv_dir is None:
        ohlcv_dir = _ROOT / "data" / "raw" / "financials" / "ohlcv"
    if earnings_dir is None:
        earnings_dir = _ROOT / "data" / "raw" / "financials" / "earnings"

    _NULLS = [
        pl.lit(None).cast(pl.Float64).alias("own_layer_momentum_20d"),
        pl.lit(None).cast(pl.Float64).alias("ecosystem_momentum_20d"),
        pl.lit(None).cast(pl.Float64).alias("supply_chain_correlation_60d"),
        pl.lit(None).cast(pl.Float64).alias("peer_eps_surprise_mean"),
    ]

    min_date = df["date"].min()
    max_date = df["date"].max()

    close_wide = _build_close_matrix(ohlcv_dir, min_date, max_date)
    if close_wide.is_empty() or close_wide.height < 2:
        _LOG.warning("[SupplyChain] No OHLCV data found — supply chain features will be null")
        return df.with_columns(_NULLS)

    ret_20d_wide = _compute_20d_returns(close_wide)

    ticker_cols = [c for c in close_wide.columns if c != "date"]
    ret_1d_wide = close_wide.select(
        ["date"] + [pl.col(t).pct_change(1).alias(t) for t in ticker_cols]
    )

    earnings_path = earnings_dir / "earnings_surprises.parquet"
    earnings_df = (
        pl.read_parquet(earnings_path)
        if earnings_path.exists()
        else pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "quarter_end": pl.Date,
            "eps_surprise": pl.Float64,
            "eps_surprise_pct": pl.Float64,
        })
    )

    own_mom_vals: list[float | None]  = []
    eco_mom_vals: list[float | None]  = []
    corr_vals:    list[float | None]  = []
    eps_vals:     list[float | None]  = []

    for row in df.select(["ticker", "date"]).iter_rows(named=True):
        ticker = row["ticker"]
        as_of  = row["date"]

        own_mom_vals.append(compute_layer_momentum(ticker, as_of, ret_20d_wide, exclude_own_layer=False))
        eco_mom_vals.append(compute_layer_momentum(ticker, as_of, ret_20d_wide, exclude_own_layer=True))
        corr_vals.append(compute_supply_chain_correlation(ticker, as_of, ret_1d_wide))
        eps_vals.append(compute_peer_eps_surprise(ticker, as_of, earnings_df))

    return df.with_columns([
        pl.Series("own_layer_momentum_20d",       own_mom_vals, dtype=pl.Float64),
        pl.Series("ecosystem_momentum_20d",        eco_mom_vals, dtype=pl.Float64),
        pl.Series("supply_chain_correlation_60d",  corr_vals,    dtype=pl.Float64),
        pl.Series("peer_eps_surprise_mean",         eps_vals,     dtype=pl.Float64),
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _LOG.info("Supply chain features are computed on-demand in train.py — no standalone run needed.")
    _LOG.info("Ensure OHLCV data exists at data/raw/financials/ohlcv/ before running train.py.")
