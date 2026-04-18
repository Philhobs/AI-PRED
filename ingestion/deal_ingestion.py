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

from ingestion.ticker_registry import TICKER_LAYERS, TICKERS

_LOG = logging.getLogger(__name__)

# Maps deal keywords found in 8-K text to deal_type values
_DEAL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "power_purchase_agreement": [
        "power purchase agreement", "offtake agreement",
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
        "META": ["meta platforms"],
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


_MW_PATTERNS = [
    re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*(?:GW|gigawatt)", re.IGNORECASE),
    re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*(?:MW|megawatt)", re.IGNORECASE),
]


def _extract_deal_mw(text: str) -> float | None:
    """Extract MW capacity from 8-K deal text.

    Returns MW as float (GW converted × 1000), or None if not found.
    """
    if not text:
        return None
    for i, pat in enumerate(_MW_PATTERNS):
        m = pat.search(text)
        if m:
            raw = float(m.group(1).replace(",", ""))
            # First pattern is GW — multiply by 1000
            if i == 0:
                return raw * 1000.0
            return raw
    return None


_HYPERSCALER_KEYWORDS = {"microsoft", "amazon", "aws", "google", "alphabet", "meta", "apple"}
_CRYPTO_KEYWORDS = {
    "iren", "applied digital", "apld", "marathon digital", "riot platforms",
    "core scientific", "bit digital", "bitfarms",
}
_UTILITY_KEYWORDS = {
    "duke energy", "dominion", "southern company", "exelon", "entergy",
    "nextera", "eversource", "ameren", "xcel energy", "sempra",
}


def _classify_buyer_type(counterparty: str) -> str:
    """Classify counterparty as hyperscaler, crypto_miner, utility, or other."""
    if not counterparty:
        return "other"
    lower = counterparty.lower()
    if any(kw in lower for kw in _HYPERSCALER_KEYWORDS):
        return "hyperscaler"
    if any(kw in lower for kw in _CRYPTO_KEYWORDS):
        return "crypto_miner"
    if any(kw in lower for kw in _UTILITY_KEYWORDS):
        return "utility"
    return "other"


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
    seen_counterparties: set[str] = set()

    for company_name, ticker in _NAME_TO_TICKER.items():
        if ticker in seen_counterparties:
            continue
        if ticker == filer_ticker:
            continue
        if ticker not in watchlist:
            continue
        if company_name not in text_lower:
            continue

        deal_type = _classify_deal_type(text)
        # Extract a short description from the first sentence mentioning the counterparty
        sentences = re.split(r"[.!?]", text)
        desc = next(
            (s.strip() for s in sentences if company_name in s.lower()),
            text[:200].strip(),
        )

        deal_dict = {
            "party_a": filer_ticker,
            "party_b": ticker,
            "deal_type": deal_type,
            "description": desc[:300],
            "source": "8-K",
            "confidence": 0.7,
        }
        deal_dict["deal_mw"] = _extract_deal_mw(text)
        deal_dict["buyer_type"] = _classify_buyer_type(company_name)
        rows.append(deal_dict)
        seen_counterparties.add(ticker)

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
        # Canonical key: sort party_a < party_b so (A,B) and (B,A) fall into same bucket
        pl.when(pl.col("party_a") < pl.col("party_b"))
          .then(pl.col("party_a")).otherwise(pl.col("party_b")).alias("key_lo"),
        pl.when(pl.col("party_a") < pl.col("party_b"))
          .then(pl.col("party_b")).otherwise(pl.col("party_a")).alias("key_hi"),
    ]).with_columns([
        (pl.col("confidence") * (0.5 ** pl.col("years_ago"))).alias("w"),
    ])

    agg = (
        rows_with_weight
        .group_by(["key_lo", "key_hi"])
        .agg([
            pl.col("w").sum().alias("edge_weight"),
            pl.col("key_lo").count().alias("deal_count").cast(pl.Int32),
            pl.col("date").max().alias("last_deal_date"),
            pl.col("deal_type").unique().sort().str.join("|").alias("edge_types"),
        ])
    )

    forward = agg.rename({"key_lo": "ticker_from", "key_hi": "ticker_to"})
    reverse = agg.rename({"key_lo": "ticker_to", "key_hi": "ticker_from"}).select(forward.columns)
    return pl.concat([forward, reverse])


def build_deals(
    manual_csv: Path | None = None,
    as_of: date | None = None,
    *,
    filings: list[dict] | None = None,
    manual_csv_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame] | pl.DataFrame:
    """Build deals DataFrame from manual CSV and/or 8-K filings.

    Two calling modes:

    Legacy mode (returns (deals_df, edges_df)):
        build_deals(manual_csv=Path(...), as_of=date(...))

    Enriched mode (returns deals_df only):
        build_deals(
            filings=[{"text": ..., "date": ..., "url": ...}],
            manual_csv_path=Path(...),
            output_path=Path(...),
        )
    """
    enriched_mode = filings is not None or manual_csv_path is not None or output_path is not None

    # Resolve the manual CSV path from either argument style
    csv_path: Path | None = manual_csv_path if manual_csv_path is not None else manual_csv

    as_of = as_of or date.today()

    # --- Load manual deals ---
    if csv_path is not None:
        manual_df = _load_manual_deals(csv_path)
    else:
        manual_df = pl.DataFrame(schema={
            "date": pl.Date, "party_a": pl.Utf8, "party_b": pl.Utf8,
            "deal_type": pl.Utf8, "description": pl.Utf8,
            "source_url": pl.Utf8, "source": pl.Utf8, "confidence": pl.Float64,
        })

    if manual_df.is_empty():
        _LOG.warning("No manual deals loaded from %s", csv_path)

    # Ensure new columns exist on manual DataFrame
    if "deal_mw" not in manual_df.columns:
        manual_df = manual_df.with_columns(pl.lit(None).cast(pl.Float64).alias("deal_mw"))
    if "buyer_type" not in manual_df.columns:
        manual_df = manual_df.with_columns(pl.lit("other").alias("buyer_type"))

    # --- Parse 8-K filings (enriched mode) ---
    filing_rows: list[dict] = []
    if filings:
        watchlist = set(TICKERS)
        for filing in filings:
            text = filing.get("text", "")
            filing_date_str = filing.get("date", str(as_of))
            url = filing.get("url", "")
            # No filer ticker in generic filing dicts — use empty string to skip self-exclusion
            rows = _parse_8k_for_deals(text, filer_ticker="", watchlist=watchlist)
            for row in rows:
                row["date"] = filing_date_str
                row["source_url"] = url
            filing_rows.extend(rows)

    # --- Combine filing rows with manual deals ---
    if filing_rows:
        # Parse date strings for filing rows
        parsed_rows = []
        for row in filing_rows:
            d = row.get("date", str(as_of))
            if isinstance(d, str):
                try:
                    row["date"] = date.fromisoformat(d)
                except ValueError:
                    row["date"] = as_of
            parsed_rows.append(row)

        filing_df = pl.DataFrame(parsed_rows).with_columns(
            pl.col("date").cast(pl.Date)
        )
        # Ensure filing_df has all columns manual_df has (fill missing with null/default)
        for col, dtype in manual_df.schema.items():
            if col not in filing_df.columns:
                filing_df = filing_df.with_columns(pl.lit(None).cast(dtype).alias(col))
        # Align column order
        filing_df = filing_df.select(manual_df.columns)
        df = pl.concat([manual_df, filing_df], how="diagonal_relaxed")
    else:
        df = manual_df

    # Cast new enrichment columns to correct types
    df = df.with_columns([
        pl.col("deal_mw").cast(pl.Float64),
        pl.col("buyer_type").fill_null("other").cast(pl.Utf8),
    ])

    if enriched_mode:
        # Enriched mode: optionally write parquet and return deals only
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_path, compression="snappy")
            _LOG.info("Saved %d deals to %s", len(df), output_path)
        return df

    # Legacy mode: add layer columns, build edges, return (deals, edges)
    layer_map = TICKER_LAYERS
    deals = df.with_columns([
        pl.col("party_a").replace(layer_map).alias("layer_a"),
        pl.col("party_b").replace(layer_map).alias("layer_b"),
        (
            pl.col("party_a") + "-" + pl.col("party_b") + "-"
            + pl.col("date").cast(pl.Utf8)
        ).alias("deal_id"),
    ])

    edges = _build_edges(deals, as_of=as_of)

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
