"""USPTO patent signal features.

Features (USPTO_PATENT_FEATURE_COLS):
    patent_app_count_90d      — count of AI/semiconductor patent applications in 90d
    patent_app_momentum       — recent 90d apps minus prior 90d apps (R&D acceleration)
    patent_grant_count_365d   — count of granted patents in 365d
    patent_grant_rate_365d    — grants_365d / max(apps_365d, 1) — IP portfolio quality
    patent_ai_cpc_share_90d   — fraction of 90d apps in G06N (AI/ML) vs all CPC codes
    patent_citation_count_365d— forward citations on patents granted in 365d window

Ticker matching: reuses _TICKER_NAME_MAP and _normalize_name from gov_behavioral_features.
All features zero-filled when data is absent.

Tier routing: medium + long only (patent cycles too slow for 5d/20d horizons).
"""
from __future__ import annotations

import difflib
import logging
from pathlib import Path

import duckdb
import polars as pl

from processing.gov_behavioral_features import _TICKER_NAME_MAP, _normalize_name

_LOG = logging.getLogger(__name__)

USPTO_PATENT_FEATURE_COLS: list[str] = [
    "patent_app_count_90d",
    "patent_app_momentum",
    "patent_grant_count_365d",
    "patent_grant_rate_365d",
    "patent_ai_cpc_share_90d",
    "patent_citation_count_365d",
]

_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}


def _load_apps(apps_dir: Path) -> pl.DataFrame:
    files = sorted(apps_dir.glob("date=*/apps.parquet")) if apps_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_APP_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files])


def _load_grants(grants_dir: Path) -> pl.DataFrame:
    files = sorted(grants_dir.glob("date=*/grants.parquet")) if grants_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_GRANT_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files])


def _build_assignee_to_ticker(assignee_names: list[str], tickers: list[str]) -> dict[str, str]:
    """Return {assignee_name: ticker} via normalize + difflib fuzzy match (cutoff 0.85)."""
    norm_to_assignee = {_normalize_name(n): n for n in assignee_names}
    result: dict[str, str] = {}
    for ticker in tickers:
        search_name = _TICKER_NAME_MAP.get(ticker)
        if not search_name:
            continue
        matches = difflib.get_close_matches(
            _normalize_name(search_name), norm_to_assignee.keys(), n=1, cutoff=0.85
        )
        if matches:
            result[norm_to_assignee[matches[0]]] = ticker
    return result


def join_patent_features(
    df: pl.DataFrame,
    apps_dir: Path,
    grants_dir: Path,
) -> pl.DataFrame:
    """Left-join USPTO patent features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'ticker' (Utf8) and 'date' (Date) columns.
        apps_dir: Root of data/raw/patents/applications/ Hive tree.
        grants_dir: Root of data/raw/patents/grants/ Hive tree.

    Returns:
        df with USPTO_PATENT_FEATURE_COLS appended (Float64). Zero-filled.
    """
    from ingestion.ticker_registry import TICKERS

    raw_apps = _load_apps(apps_dir)
    raw_grants = _load_grants(grants_dir)

    all_assignees: set[str] = set()
    if not raw_apps.is_empty():
        all_assignees.update(raw_apps["assignee_name"].unique().to_list())
    if not raw_grants.is_empty():
        all_assignees.update(raw_grants["assignee_name"].unique().to_list())

    if not all_assignees:
        for col in USPTO_PATENT_FEATURE_COLS:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        return df

    assignee_to_ticker = _build_assignee_to_ticker(list(all_assignees), TICKERS)
    if not assignee_to_ticker:
        for col in USPTO_PATENT_FEATURE_COLS:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        return df

    mapping_df = pl.DataFrame(
        {"assignee_name": list(assignee_to_ticker.keys()),
         "ticker": list(assignee_to_ticker.values())},
        schema={"assignee_name": pl.Utf8, "ticker": pl.Utf8},
    )

    # Map assignee_name → ticker via vectorized join
    if not raw_apps.is_empty():
        apps = raw_apps.join(mapping_df, on="assignee_name", how="left").filter(
            pl.col("ticker").is_not_null()
        )
    else:
        apps = pl.DataFrame(schema={**_APP_SCHEMA, "ticker": pl.Utf8})

    if not raw_grants.is_empty():
        grants = raw_grants.join(mapping_df, on="assignee_name", how="left").filter(
            pl.col("ticker").is_not_null()
        )
    else:
        grants = pl.DataFrame(schema={**_GRANT_SCHEMA, "ticker": pl.Utf8})

    query_pairs = df.select(["ticker", "date"]).unique()

    with duckdb.connect() as con:
        con.register("apps", apps.to_arrow())
        con.register("grants", grants.to_arrow())
        con.register("query_pairs", query_pairs.to_arrow())

        # JOIN covers full 365d window to correctly compute _apps_365d for grant rate
        # CASE WHEN filters then narrow the sub-windows (90d, 180d) as needed
        app_result = con.execute("""
            SELECT
                q.ticker,
                q.date,

                -- patent_app_count_90d: apps in [date-90d, date]
                COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                         AND a.filing_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS patent_app_count_90d,

                -- patent_app_momentum: recent_90d - prior_90d (91-180d)
                COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                         AND a.filing_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                - COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 180 DAY
                         AND a.filing_date < q.date - INTERVAL 90 DAY
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS patent_app_momentum,

                -- patent_ai_cpc_share_90d: G06N / GREATEST(total, 1)
                COALESCE(
                    CAST(SUM(CASE
                        WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                             AND a.filing_date <= q.date
                             AND a.cpc_group = 'G06N'
                        THEN 1 ELSE 0 END) AS DOUBLE)
                    / GREATEST(CAST(SUM(CASE
                        WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                             AND a.filing_date <= q.date
                        THEN 1 ELSE 0 END) AS DOUBLE), 1.0),
                0.0) AS patent_ai_cpc_share_90d,

                -- _apps_365d for grant rate denominator
                COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 365 DAY
                         AND a.filing_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS _apps_365d

            FROM query_pairs q
            LEFT JOIN apps a
                ON a.ticker = q.ticker
                AND a.filing_date <= q.date
                AND a.filing_date >= q.date - INTERVAL 365 DAY
            GROUP BY q.ticker, q.date
        """).pl()

        grant_result = con.execute("""
            SELECT
                q.ticker,
                q.date,

                -- patent_grant_count_365d
                COALESCE(CAST(SUM(CASE
                    WHEN g.grant_date >= q.date - INTERVAL 365 DAY
                         AND g.grant_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS patent_grant_count_365d,

                -- patent_citation_count_365d
                COALESCE(SUM(CASE
                    WHEN g.grant_date >= q.date - INTERVAL 365 DAY
                         AND g.grant_date <= q.date
                    THEN CAST(g.forward_citation_count AS DOUBLE) ELSE 0.0 END), 0.0)
                    AS patent_citation_count_365d

            FROM query_pairs q
            LEFT JOIN grants g
                ON g.ticker = q.ticker
                AND g.grant_date <= q.date
                AND g.grant_date >= q.date - INTERVAL 365 DAY
            GROUP BY q.ticker, q.date
        """).pl()

    # Compute grant_rate_365d = patent_grant_count_365d / GREATEST(_apps_365d, 1.0)
    combined = app_result.join(grant_result, on=["ticker", "date"], how="left")
    combined = combined.with_columns(
        (pl.col("patent_grant_count_365d") / pl.col("_apps_365d").clip(lower_bound=1.0))
        .alias("patent_grant_rate_365d")
    ).drop(["_apps_365d"])

    # Left-join features back to original df
    df = df.join(combined, on=["ticker", "date"], how="left")

    # Zero-fill backstop: catches nulls from partial left-joins and missing dates
    for col in USPTO_PATENT_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
