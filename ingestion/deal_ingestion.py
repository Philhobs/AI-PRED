"""
Deal and partnership ingestion for the AI infrastructure supply chain graph.

Two sources:
  1. SEC 8-K filings (Item 1.01 — material definitive agreements) via EDGAR EFTS.
  2. data/manual/deals_override.csv — user-curated deals (confidence=1.0).

Output:
  data/raw/graph/deals.parquet   — one row per confirmed deal
  data/raw/graph/edges.parquet   — one row per unique ticker pair (aggregated)
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

from ingestion.ticker_registry import TICKER_LAYERS, TICKERS

_LOG = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "ai-infra-predictor research@example.com"}

# Maps deal keywords found in 8-K text to deal_type values
_DEAL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "power_purchase_agreement": [
        "power purchase agreement", "ppa", "offtake agreement",
        "energy purchase agreement", "renewable energy agreement",
    ],
    "supply_agreement": [
        "supply agreement", "supply contract", "purchase agreement",
        "procurement agreement", "materials agreement",
    ],
    "manufacturing_agreement": [
        "manufacturing agreement", "foundry agreement", "fab agreement",
        "capacity reservation", "wafer supply agreement",
    ],
    "customer_contract": [
        "customer agreement", "service agreement", "cloud agreement",
        "subscription agreement", "gpu cluster",
    ],
    "construction_contract": [
        "construction agreement", "epc agreement", "engineering",
        "procurement and construction",
    ],
    "joint_venture": ["joint venture", "jv agreement", "partnership agreement"],
    "investment": ["investment agreement", "equity investment", "financing agreement"],
    "licensing_agreement": ["license agreement", "licensing agreement", "ip agreement"],
}

# Company name → ticker for counterparty extraction from 8-K text
_NAME_TO_TICKER: dict[str, str] = {
    name.lower(): ticker
    for ticker, aliases in {
        "MSFT": ["microsoft", "microsoft corporation"],
        "AMZN": ["amazon", "amazon web services", "aws"],
        "GOOGL": ["google", "alphabet"],
        "META": ["meta", "meta platforms"],
        "ORCL": ["oracle", "oracle corporation"],
        "IBM": ["ibm", "international business machines"],
        "NVDA": ["nvidia", "nvidia corporation"],
        "AMD": ["advanced micro devices", "amd"],
        "TSM": ["tsmc", "taiwan semiconductor"],
        "ANET": ["arista", "arista networks"],
        "CEG": ["constellation energy", "constellation"],
        "NEE": ["nextera", "nextera energy"],
        "SO": ["southern company"],
        "EXC": ["exelon"],
        "ETR": ["entergy"],
        "TLN": ["talen energy", "talen"],
        "NRG": ["nrg energy"],
        "VST": ["vistra"],
        "GEV": ["ge vernova"],
        "BWX": ["bwx technologies", "bwxt"],
        "OKLO": ["oklo"],
        "SMR": ["nuscale", "nuscale power"],
        "VRT": ["vertiv"],
        "GNRC": ["generac"],
        "EQIX": ["equinix"],
        "DLR": ["digital realty"],
        "PWR": ["quanta services", "quanta"],
        "MTZ": ["mastec"],
        "EME": ["emcor"],
        "FCX": ["freeport", "freeport-mcmoran"],
        "NTAP": ["netapp"],
        "PSTG": ["pure storage"],
        "MU": ["micron", "micron technology"],
        "ENTG": ["entegris"],
        "LIN": ["linde"],
        "APD": ["air products"],
    }.items()
    for name in aliases
}


def _classify_deal_type(text: str) -> str:
    """Return the best-matching deal type for a block of 8-K text."""
    text_lower = text.lower()
    for deal_type, keywords in _DEAL_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return deal_type
    return "customer_contract"  # default for unclassified material agreements


def _parse_8k_for_deals(
    text: str,
    filer_ticker: str,
    watchlist: set[str],
) -> list[dict]:
    """Extract deal rows from 8-K Item 1.01 text.

    Finds counterparty company names in the text, maps them to tickers,
    and classifies the deal type.
    """
    rows: list[dict] = []
    text_lower = text.lower()

    for name, counterparty in _NAME_TO_TICKER.items():
        if counterparty == filer_ticker:
            continue
        if counterparty not in watchlist:
            continue
        if name not in text_lower:
            continue

        deal_type = _classify_deal_type(text)
        # Extract a short description from the first sentence mentioning the counterparty
        sentences = re.split(r"[.!?]", text)
        desc = next(
            (s.strip() for s in sentences if name in s.lower()),
            text[:200].strip(),
        )

        rows.append({
            "party_a": filer_ticker,
            "party_b": counterparty,
            "deal_type": deal_type,
            "description": desc[:300],
            "source": "8-K",
            "confidence": 0.7,
        })

    return rows


def _load_manual_deals(csv_path: Path) -> pl.DataFrame:
    """Load deals_override.csv and return with confidence=1.0 and layer columns."""
    if not csv_path.exists():
        return pl.DataFrame(schema={
            "date": pl.Date, "party_a": pl.Utf8, "party_b": pl.Utf8,
            "deal_type": pl.Utf8, "description": pl.Utf8,
            "source_url": pl.Utf8, "source": pl.Utf8, "confidence": pl.Float64,
        })

    df = pl.read_csv(csv_path).with_columns([
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.lit("manual").alias("source"),
        pl.lit(1.0).alias("confidence"),
    ])
    return df


def _build_edges(deals: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Aggregate deals into a weighted edge list.

    Edge weight = sum of (confidence × 0.5^years_since_deal) per pair.
    """
    if deals.is_empty():
        return pl.DataFrame(schema={
            "ticker_from": pl.Utf8, "ticker_to": pl.Utf8,
            "edge_weight": pl.Float64, "deal_count": pl.Int32,
            "last_deal_date": pl.Date, "edge_types": pl.Utf8,
        })

    as_of_days = (as_of - date(1970, 1, 1)).days

    rows_with_weight = deals.with_columns([
        (
            (pl.lit(as_of_days) - pl.col("date").cast(pl.Int32)) / 365.25
        ).alias("years_ago"),
    ]).with_columns([
        (pl.col("confidence") * (0.5 ** pl.col("years_ago"))).alias("w"),
    ])

    edges = (
        rows_with_weight
        .group_by(["party_a", "party_b"])
        .agg([
            pl.col("w").sum().alias("edge_weight"),
            pl.col("party_a").count().alias("deal_count").cast(pl.Int32),
            pl.col("date").max().alias("last_deal_date"),
            pl.col("deal_type").unique().sort().str.join("|").alias("edge_types"),
        ])
        .rename({"party_a": "ticker_from", "party_b": "ticker_to"})
    )

    # Make edges bidirectional
    reverse = edges.rename({
        "ticker_from": "ticker_to",
        "ticker_to": "ticker_from",
    }).select(edges.columns)

    return pl.concat([edges, reverse]).unique(["ticker_from", "ticker_to"])


def build_deals(
    manual_csv: Path,
    days_back: int = 730,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build deals and edges DataFrames from manual CSV.

    Returns (deals_df, edges_df).
    """
    manual_df = _load_manual_deals(manual_csv)

    if manual_df.is_empty():
        _LOG.warning("No manual deals loaded from %s", manual_csv)

    # Add layer columns
    layer_map = TICKER_LAYERS
    deals = manual_df.with_columns([
        pl.col("party_a").replace(layer_map).alias("layer_a"),
        pl.col("party_b").replace(layer_map).alias("layer_b"),
        (
            pl.col("party_a") + "-" + pl.col("party_b") + "-"
            + pl.col("date").cast(pl.Utf8)
        ).alias("deal_id"),
    ])

    edges = _build_edges(deals, as_of=date.today())

    _LOG.info(
        "Built deal graph: %d deals, %d edges (%d unique pairs)",
        len(deals),
        len(edges),
        len(edges) // 2,
    )
    return deals, edges


def save_deals(
    deals: pl.DataFrame,
    edges: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Write deals.parquet and edges.parquet to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    deals.write_parquet(output_dir / "deals.parquet", compression="snappy")
    edges.write_parquet(output_dir / "edges.parquet", compression="snappy")
    _LOG.info("Saved %d deals and %d edges to %s", len(deals), len(edges), output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    manual_csv = project_root / "data" / "manual" / "deals_override.csv"
    output_dir = project_root / "data" / "raw" / "graph"

    deals, edges = build_deals(manual_csv)
    save_deals(deals, edges, output_dir)
