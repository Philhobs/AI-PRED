"""
Knowledge graph features derived from the AI infrastructure deal graph.

Three features per ticker per date:
  graph_partner_momentum_30d  — weighted avg 30d return of direct deal partners
  graph_deal_count_90d        — new deals by direct partners in past 90 days
  graph_hops_to_hyperscaler   — 1/(hops+1) to nearest MSFT/AMZN/GOOGL/META; 1.0 for hyperscalers

Graph is built in memory from edges.parquet using NetworkX.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import networkx as nx
import polars as pl

from ingestion.ticker_registry import HYPERSCALERS, TICKERS

_LOG = logging.getLogger(__name__)


def build_graph(edges: pl.DataFrame) -> nx.Graph:
    """Build undirected weighted graph from edges DataFrame."""
    g = nx.Graph()
    g.add_nodes_from(TICKERS)
    for row in edges.iter_rows(named=True):
        g.add_edge(
            row["ticker_from"],
            row["ticker_to"],
            weight=row["edge_weight"],
        )
    return g


def _compute_partner_momentum_30d(
    graph: nx.Graph,
    ticker: str,
    ohlcv: pl.DataFrame,
    as_of: date,
) -> float | None:
    """Weighted average 30-day return of direct deal partners.

    weight = edge_weight between ticker and partner.
    Returns None if ticker has no partners or no price data for partners.
    """
    neighbors = list(graph.neighbors(ticker))
    if not neighbors:
        return None

    window_start = as_of - timedelta(days=30)
    total_weight = 0.0
    weighted_return = 0.0
    found_any = False

    for partner in neighbors:
        edge_weight = graph[ticker][partner]["weight"]
        # Find partner close price at as_of and 30d ago
        partner_prices = ohlcv.filter(
            (pl.col("ticker") == partner) &
            (pl.col("date") >= window_start) &
            (pl.col("date") <= as_of)
        ).sort("date")

        if len(partner_prices) < 2:
            continue

        price_now = partner_prices["close_price"][-1]
        price_then = partner_prices["close_price"][0]

        if price_then == 0:
            continue

        ret = (price_now / price_then) - 1.0
        weighted_return += edge_weight * ret
        total_weight += edge_weight
        found_any = True

    if not found_any or total_weight == 0:
        return None

    return weighted_return / total_weight


def _compute_deal_count_90d(
    graph: nx.Graph,
    ticker: str,
    deals: pl.DataFrame,
    as_of: date,
) -> int:
    """Count of new deals filed by direct partners in the past 90 days."""
    neighbors = set(graph.neighbors(ticker))
    if not neighbors or deals.is_empty():
        return 0

    window_start = as_of - timedelta(days=90)
    # Count deals where a partner appears but exclude deals involving the ticker itself
    partner_deals = deals.filter(
        (pl.col("party_a").is_in(neighbors) | pl.col("party_b").is_in(neighbors)) &
        (pl.col("party_a") != ticker) &
        (pl.col("party_b") != ticker) &
        (pl.col("date") >= window_start) &
        (pl.col("date") <= as_of)
    )
    return len(partner_deals)


def _compute_hops_to_hyperscaler(graph: nx.Graph, ticker: str) -> float:
    """Encoded proximity to hyperscalers as 1/(hops+1).

    Hyperscalers themselves → 1.0.
    Direct partners → 1/(1+1) = 0.5.
    Two hops away → 1/(2+1) ≈ 0.333.
    No path → 0.0.
    """
    if ticker in HYPERSCALERS:
        return 1.0

    min_hops = None
    for hyperscaler in HYPERSCALERS:
        if hyperscaler not in graph:
            continue
        try:
            hops = nx.shortest_path_length(graph, ticker, hyperscaler)
            if min_hops is None or hops < min_hops:
                min_hops = hops
        except nx.NetworkXNoPath:
            continue

    if min_hops is None:
        return 0.0

    return 1.0 / (min_hops + 1)


def compute_graph_features(
    edges_path: Path,
    deals_path: Path,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """Compute 3 graph features for all tickers × OHLCV dates.

    Returns DataFrame: [ticker, date, graph_partner_momentum_30d,
                        graph_deal_count_90d, graph_hops_to_hyperscaler]
    """
    import duckdb

    if not edges_path.exists():
        _LOG.warning("No edges.parquet at %s — graph features will be null", edges_path)
        return pl.DataFrame()

    edges = pl.read_parquet(edges_path)
    deals = pl.read_parquet(deals_path) if deals_path.exists() else pl.DataFrame()
    graph = build_graph(edges)

    ohlcv_parquets = list(ohlcv_dir.glob("**/*.parquet"))
    if not ohlcv_parquets:
        _LOG.error("No OHLCV parquets in %s", ohlcv_dir)
        return pl.DataFrame()

    ohlcv_glob = str(ohlcv_dir / "**" / "*.parquet")
    with duckdb.connect() as con:
        spine = con.execute("""
            SELECT DISTINCT ticker, date
            FROM read_parquet(?)
            ORDER BY ticker, date
        """, [ohlcv_glob]).pl()

    ohlcv = pl.scan_parquet(ohlcv_glob).select(["ticker", "date", "close_price"]).collect()

    # Precompute hop distances (static — graph doesn't change per date)
    hop_distances = {t: _compute_hops_to_hyperscaler(graph, t) for t in TICKERS}

    rows: list[dict] = []
    for ticker in TICKERS:
        ticker_spine = spine.filter(pl.col("ticker") == ticker)
        for row_date in ticker_spine["date"].to_list():
            rows.append({
                "ticker": ticker,
                "date": row_date,
                "graph_partner_momentum_30d": _compute_partner_momentum_30d(
                    graph, ticker, ohlcv, row_date
                ),
                "graph_deal_count_90d": _compute_deal_count_90d(
                    graph, ticker, deals, row_date
                ),
                "graph_hops_to_hyperscaler": hop_distances.get(ticker, 0.0),
            })

    result = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    _LOG.info(
        "Computed graph features: %d rows for %d tickers",
        len(result), result["ticker"].n_unique(),
    )
    return result


def save_graph_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker parquets to output_dir/<TICKER>/graph_daily.parquet."""
    for ticker in df["ticker"].unique().to_list():
        out = output_dir / ticker / "graph_daily.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("ticker") == ticker).write_parquet(out, compression="snappy")
    _LOG.info("Saved graph features for %d tickers", df["ticker"].n_unique())


def join_graph_features(df: pl.DataFrame, graph_features_dir: Path) -> pl.DataFrame:
    """Backward asof join graph features onto training DataFrame."""
    feature_cols = [
        "graph_partner_momentum_30d",
        "graph_deal_count_90d",
        "graph_hops_to_hyperscaler",
    ]
    if not any(graph_features_dir.glob("*/graph_daily.parquet")):
        _LOG.warning("No graph feature parquets in %s — features will be null", graph_features_dir)
        for col in feature_cols:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    # collect() is intentional: join_asof requires materialised, sorted DataFrame.
    features = (
        pl.scan_parquet(str(graph_features_dir / "*/graph_daily.parquet"))
        .sort(["ticker", "date"])
        .collect()
    )
    features_renamed = features.rename({"date": "graph_date"})
    result = df.sort(["ticker", "date"]).join_asof(
        features_renamed,
        left_on="date",
        right_on="graph_date",
        by="ticker",
        strategy="backward",
    )
    _LOG.info("Joined graph features: %d rows", len(result))
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project_root = Path(__file__).parent.parent
    edges_path = project_root / "data" / "raw" / "graph" / "edges.parquet"
    deals_path = project_root / "data" / "raw" / "graph" / "deals.parquet"
    ohlcv_dir  = project_root / "data" / "raw" / "financials" / "ohlcv"
    output_dir = project_root / "data" / "raw" / "graph" / "features"

    df = compute_graph_features(edges_path, deals_path, ohlcv_dir)
    if df.is_empty():
        sys.exit(1)
    save_graph_features(df, output_dir)
    _LOG.info("Done. Retrain with: python models/train.py")
