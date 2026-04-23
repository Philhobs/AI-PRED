"""Government behavioral data features: SAM.gov contracts + FERC interconnection queue.

Features (GOV_BEHAVIORAL_FEATURE_COLS):
    gov_contract_value_90d    — rolling 90-day USD awards sum for the ticker's company
    gov_contract_count_90d    — rolling 90-day award count for the ticker's company
    gov_contract_momentum     — 30d award value minus prior 60d (positive = accelerating)
    gov_ai_spend_30d          — market-wide rolling 30-day AI/DC NAICS award total
    ferc_queue_mw_30d         — rolling 30-day MW filed in DC power states
    ferc_grid_constraint_score — ferc_queue_mw_30d / (365d_total / 12), denominator floored at 1.0

Ticker-specific features joined on (ticker, date); market-wide joined on date only.
All features zero-filled when data is absent.

DC power states: VA, TX, OH, AZ, NV, OR, GA, WA
"""
from __future__ import annotations

import difflib
import logging
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)

GOV_BEHAVIORAL_FEATURE_COLS: list[str] = [
    "gov_contract_value_90d",
    "gov_contract_count_90d",
    "gov_contract_momentum",
    "gov_ai_spend_30d",
    "ferc_queue_mw_30d",
    "ferc_grid_constraint_score",
]

# Tickers whose symbol gives no hint of their SAM.gov awardee name
GOV_TICKER_OVERRIDE_MAP: dict[str, str] = {
    "GOOGL": "Alphabet",
    "META":  "Meta Platforms",
    "MSFT":  "Microsoft Corporation",
    "AMZN":  "Amazon Web Services",
    "TSM":   "Taiwan Semiconductor",
}

# Full search-name map for fuzzy matching against SAM.gov awardee_name strings
_TICKER_NAME_MAP: dict[str, str] = {
    **GOV_TICKER_OVERRIDE_MAP,
    "NVDA": "NVIDIA",
    "AMD":  "Advanced Micro Devices",
    "INTC": "Intel",
    "ORCL": "Oracle",
    "IBM":  "International Business Machines",
    "DELL": "Dell Technologies",
    "HPE":  "Hewlett Packard Enterprise",
    "CSCO": "Cisco",
    "ACN":  "Accenture",
    "BAH":  "Booz Allen Hamilton",
    "SAIC": "Science Applications International",
    "LEIDOS": "Leidos",
    "CDW":  "CDW",
}

_LEGAL_SUFFIXES = (
    " corporation", " corp", " inc", " incorporated",
    " llc", " ltd", " limited", " co", " company",
    " holdings", " technologies", " systems", " solutions",
)

_AI_NAICS_CODES = {"541511", "541512", "541519", "518210", "334413"}

_DC_STATES = {"VA", "TX", "OH", "AZ", "NV", "OR", "GA", "WA"}


def _normalize_name(name: str) -> str:
    n = name.lower().strip()
    for suffix in _LEGAL_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _build_awardee_to_ticker(awardee_names: list[str], tickers: list[str]) -> dict[str, str]:
    """Return {awardee_name: ticker} via normalize + difflib fuzzy match (cutoff 0.85)."""
    norm_to_awardee = {_normalize_name(n): n for n in awardee_names}
    result: dict[str, str] = {}
    for ticker in tickers:
        search_name = _TICKER_NAME_MAP.get(ticker)
        if not search_name:
            continue
        matches = difflib.get_close_matches(
            _normalize_name(search_name), norm_to_awardee.keys(), n=1, cutoff=0.85
        )
        if matches:
            result[norm_to_awardee[matches[0]]] = ticker
    return result


def _load_contracts(contracts_dir: Path) -> pl.DataFrame:
    files = sorted(contracts_dir.glob("date=*/awards.parquet")) if contracts_dir.exists() else []
    if not files:
        return pl.DataFrame(schema={
            "date": pl.Date, "awardee_name": pl.Utf8, "uei": pl.Utf8,
            "contract_value_usd": pl.Float64, "naics_code": pl.Utf8, "agency": pl.Utf8,
        })
    return pl.concat([pl.read_parquet(f) for f in files])


def _load_ferc(ferc_dir: Path) -> pl.DataFrame:
    files = sorted(ferc_dir.glob("date=*/queue.parquet")) if ferc_dir.exists() else []
    if not files:
        return pl.DataFrame(schema={
            "snapshot_date": pl.Date, "queue_date": pl.Date, "project_name": pl.Utf8,
            "mw": pl.Float64, "state": pl.Utf8, "fuel": pl.Utf8,
            "status": pl.Utf8, "iso": pl.Utf8,
        })
    raw = pl.concat([pl.read_parquet(f) for f in files])
    # Sort so unique(keep="last") retains the most recent snapshot_date per project entry.
    return raw.sort("snapshot_date").unique(subset=["project_name", "queue_date"], keep="last")


def _build_ticker_features(
    contracts_dir: Path,
    tickers: list[str],
    query_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute ticker-specific rolling window features for each (ticker, date) in query_df.

    Uses a cross-join style computation so query dates need not exist in the contract data.
    """
    _empty_schema = {
        "ticker": pl.Utf8, "date": pl.Date,
        "gov_contract_value_90d": pl.Float64,
        "gov_contract_count_90d": pl.Float64,
        "gov_contract_momentum": pl.Float64,
    }

    raw = _load_contracts(contracts_dir)
    if raw.is_empty():
        return pl.DataFrame(schema=_empty_schema)

    awardee_to_ticker = _build_awardee_to_ticker(
        raw["awardee_name"].unique().to_list(), tickers
    )
    if not awardee_to_ticker:
        return pl.DataFrame(schema=_empty_schema)

    mapping_df = pl.DataFrame(
        {"awardee_name": list(awardee_to_ticker.keys()),
         "ticker": list(awardee_to_ticker.values())},
        schema={"awardee_name": pl.Utf8, "ticker": pl.Utf8},
    )
    matched = raw.join(mapping_df, on="awardee_name", how="left").filter(
        pl.col("ticker").is_not_null()
    )

    if matched.is_empty():
        return pl.DataFrame(schema=_empty_schema)

    # Aggregate to daily per ticker
    daily = (
        matched
        .group_by(["ticker", "date"])
        .agg(
            pl.col("contract_value_usd").sum().alias("daily_value"),
            pl.len().alias("daily_count"),
        )
    )

    # Extract unique (ticker, date) query pairs
    query_pairs = query_df.select(["ticker", "date"]).unique()

    with duckdb.connect() as con:
        con.register("daily", daily.to_arrow())
        con.register("query_pairs", query_pairs.to_arrow())

        # For each query (ticker, date), sum contracts in window.
        # Note: contracts are pre-aggregated to daily sums before the join.
        # Phase 1 data volumes (90-day window, ~50 tickers) fit comfortably in memory.
        # For larger scale, replace with DuckDB read_parquet() glob views.
        result = con.execute("""
            SELECT
                q.ticker,
                q.date,
                COALESCE(SUM(CASE WHEN d.date >= q.date - INTERVAL 90 DAY
                                  THEN d.daily_value ELSE 0 END), 0.0)
                    AS gov_contract_value_90d,
                COALESCE(CAST(SUM(CASE WHEN d.date >= q.date - INTERVAL 90 DAY
                                       THEN d.daily_count ELSE 0 END) AS DOUBLE), 0.0)
                    AS gov_contract_count_90d,
                2.0 * COALESCE(SUM(CASE WHEN d.date >= q.date - INTERVAL 30 DAY
                                        THEN d.daily_value ELSE 0 END), 0.0)
                    - COALESCE(SUM(CASE WHEN d.date >= q.date - INTERVAL 90 DAY
                                        THEN d.daily_value ELSE 0 END), 0.0)
                    AS gov_contract_momentum
            FROM query_pairs q
            LEFT JOIN daily d
                ON d.ticker = q.ticker
                AND d.date <= q.date
                AND d.date >= q.date - INTERVAL 90 DAY
            GROUP BY q.ticker, q.date
        """).pl()
    return result


def _build_market_features(
    contracts_dir: Path,
    ferc_dir: Path,
    query_dates: pl.Series,
) -> pl.DataFrame:
    """Compute market-wide rolling window features for each date in query_dates."""
    _empty_schema = {
        "date": pl.Date,
        "gov_ai_spend_30d": pl.Float64,
        "ferc_queue_mw_30d": pl.Float64,
        "ferc_grid_constraint_score": pl.Float64,
    }

    raw_contracts = _load_contracts(contracts_dir)
    raw_ferc = _load_ferc(ferc_dir)

    # Build a query_dates table for cross-join computations
    query_date_df = pl.DataFrame({"date": query_dates}).unique()

    with duckdb.connect() as con:
        con.register("query_date_df", query_date_df.to_arrow())

        # Market-wide SAM.gov 30-day rolling sum
        ai_contracts = raw_contracts.filter(pl.col("naics_code").is_in(_AI_NAICS_CODES))
        if not ai_contracts.is_empty():
            market_daily = (
                ai_contracts
                .group_by("date")
                .agg(pl.col("contract_value_usd").sum().alias("daily_total"))
            )
            con.register("market_daily", market_daily.to_arrow())
            sam_df = con.execute("""
                SELECT
                    q.date,
                    COALESCE(SUM(m.daily_total), 0.0) AS gov_ai_spend_30d
                FROM query_date_df q
                LEFT JOIN market_daily m
                    ON m.date <= q.date AND m.date >= q.date - INTERVAL 30 DAY
                GROUP BY q.date
            """).pl()
        else:
            sam_df = pl.DataFrame(schema={"date": pl.Date, "gov_ai_spend_30d": pl.Float64})

        # FERC rolling window features
        ferc_df: pl.DataFrame
        if not raw_ferc.is_empty() and "queue_date" in raw_ferc.columns:
            ferc_valid = raw_ferc.filter(pl.col("queue_date").is_not_null())
            ferc_dc = ferc_valid.filter(pl.col("state").is_in(_DC_STATES))
            if not ferc_dc.is_empty():
                ferc_daily = (
                    ferc_dc
                    .rename({"queue_date": "date"})
                    .group_by("date")
                    .agg(pl.col("mw").sum().alias("daily_mw"))
                )
                con.register("ferc_daily", ferc_daily.to_arrow())
                ferc_raw_result = con.execute("""
                    SELECT
                        q.date,
                        COALESCE(SUM(CASE WHEN fd.date >= q.date - INTERVAL 30 DAY
                                          THEN fd.daily_mw ELSE 0 END), 0.0)
                            AS ferc_queue_mw_30d,
                        COALESCE(SUM(fd.daily_mw), 0.0) AS _mw_365d
                    FROM query_date_df q
                    LEFT JOIN ferc_daily fd
                        ON fd.date <= q.date AND fd.date >= q.date - INTERVAL 365 DAY
                    GROUP BY q.date
                """).pl()
                ferc_df = ferc_raw_result.with_columns(
                    (pl.col("ferc_queue_mw_30d")
                     / (pl.col("_mw_365d") / 12.0).clip(lower_bound=1.0))
                    .alias("ferc_grid_constraint_score")
                ).drop("_mw_365d")
            else:
                ferc_df = pl.DataFrame(schema={
                    "date": pl.Date,
                    "ferc_queue_mw_30d": pl.Float64,
                    "ferc_grid_constraint_score": pl.Float64,
                })
        else:
            ferc_df = pl.DataFrame(schema={
                "date": pl.Date,
                "ferc_queue_mw_30d": pl.Float64,
                "ferc_grid_constraint_score": pl.Float64,
            })

    if sam_df.is_empty() and ferc_df.is_empty():
        return pl.DataFrame(schema=_empty_schema)

    if sam_df.is_empty():
        return ferc_df.with_columns(pl.lit(0.0).alias("gov_ai_spend_30d")).select(
            ["date", "gov_ai_spend_30d", "ferc_queue_mw_30d", "ferc_grid_constraint_score"]
        )
    if ferc_df.is_empty():
        return sam_df.with_columns([
            pl.lit(0.0).alias("ferc_queue_mw_30d"),
            pl.lit(0.0).alias("ferc_grid_constraint_score"),
        ])

    return sam_df.join(ferc_df, on="date", how="outer_coalesce").sort("date")


def join_gov_behavioral_features(
    df: pl.DataFrame,
    contracts_dir: Path,
    ferc_dir: Path,
) -> pl.DataFrame:
    """Left-join government behavioral features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'ticker' (Utf8) and 'date' (Date) columns.
        contracts_dir: Root of data/raw/gov_contracts/ Hive tree.
        ferc_dir: Root of data/raw/ferc_queue/ Hive tree.

    Returns:
        df with GOV_BEHAVIORAL_FEATURE_COLS appended (Float64). Zero-filled.
    """
    from ingestion.ticker_registry import TICKERS

    ticker_df = _build_ticker_features(contracts_dir, TICKERS, df)
    market_df = _build_market_features(contracts_dir, ferc_dir, df["date"])

    if not ticker_df.is_empty():
        df = df.join(ticker_df, on=["ticker", "date"], how="left")
    else:
        for col in ["gov_contract_value_90d", "gov_contract_count_90d", "gov_contract_momentum"]:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))

    if not market_df.is_empty():
        df = df.join(market_df, on="date", how="left")
    else:
        for col in ["gov_ai_spend_30d", "ferc_queue_mw_30d", "ferc_grid_constraint_score"]:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))

    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
