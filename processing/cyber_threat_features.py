"""Cyber threat regime features — market-wide signals joined by date.

Reads raw threat events from data/raw/cyber_threat/date=*/threats.parquet
(written by ingestion/cyber_threat_ingestion.py) and produces 7-day and
30-day rolling window aggregate features.

Features:
    cve_critical_7d      — CVSS >= 9.0 CVEs published, 7-day rolling sum
    cve_high_7d          — CVSS 7-8.9 CVEs published, 7-day rolling sum
    cisa_kev_7d          — CISA KEV entries added, 7-day rolling sum
    otx_pulse_7d         — AlienVault OTX threat pulses, 7-day rolling sum
    cyber_threat_index_7d — composite normalized score in [0, 1]
    cve_critical_30d     — CVSS >= 9.0 CVEs published, 30-day rolling sum
    cisa_kev_30d         — CISA KEV entries added, 30-day rolling sum

These are date-keyed (not ticker-specific). All tickers on a given date
receive the same threat feature values.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

CYBER_THREAT_FEATURE_COLS: list[str] = [
    "cve_critical_7d",
    "cve_high_7d",
    "cisa_kev_7d",
    "otx_pulse_7d",
    "cyber_threat_index_7d",
    "cve_critical_30d",
    "cisa_kev_30d",
]

_EMPTY_SCHEMA = {"date": pl.Date} | {c: pl.Float64 for c in CYBER_THREAT_FEATURE_COLS}


def build_cyber_threat_features(threats_dir: Path) -> pl.DataFrame:
    """Aggregate raw threat events into daily rolling-window features.

    Args:
        threats_dir: Root of the Hive-partitioned raw threat data,
                     e.g. data/raw/cyber_threat/ (contains date=*/threats.parquet).

    Returns:
        DataFrame with columns: [date] + CYBER_THREAT_FEATURE_COLS.
        Empty DataFrame (correct schema) if no data found.
    """
    files = sorted(threats_dir.glob("date=*/threats.parquet")) if threats_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    raw = pl.concat([pl.read_parquet(f) for f in files])
    if raw.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    # Pivot to wide format: one row per date, one column per metric
    daily = (
        raw.group_by(["date", "metric"])
        .agg(pl.col("value").sum())
        .pivot(on="metric", values="value", index="date")
        .fill_null(0.0)
        .sort("date")
    )

    # Ensure all expected raw metric columns exist
    for col in ("cve_critical", "cve_high", "cisa_kev", "otx_pulse"):
        if col not in daily.columns:
            daily = daily.with_columns(pl.lit(0.0).alias(col))

    # 7-day rolling sums (calendar-aware: window = (date - 7d, date], closed="right")
    daily = daily.with_columns([
        pl.col("cve_critical").rolling_sum_by("date", window_size="7d").alias("cve_critical_7d"),
        pl.col("cve_high").rolling_sum_by("date", window_size="7d").alias("cve_high_7d"),
        pl.col("cisa_kev").rolling_sum_by("date", window_size="7d").alias("cisa_kev_7d"),
        pl.col("otx_pulse").rolling_sum_by("date", window_size="7d").alias("otx_pulse_7d"),
    ])

    # 30-day rolling sums (calendar-aware: window = (date - 30d, date], closed="right")
    daily = daily.with_columns([
        pl.col("cve_critical").rolling_sum_by("date", window_size="30d").alias("cve_critical_30d"),
        pl.col("cisa_kev").rolling_sum_by("date", window_size="30d").alias("cisa_kev_30d"),
    ])

    # Composite threat index: weighted sum normalised to [0, 1]
    # weighted = cve_critical_7d * 3 + cve_high_7d + cisa_kev_7d * 2
    # divided by 30-day rolling max of weighted (floor 1 to avoid div-by-zero)
    daily = daily.with_columns(
        (pl.col("cve_critical_7d") * 3 + pl.col("cve_high_7d") + pl.col("cisa_kev_7d") * 2)
        .alias("_weighted")
    )
    daily = daily.with_columns(
        pl.col("_weighted").rolling_max_by("date", window_size="30d").alias("_weighted_max_30d")
    )
    daily = daily.with_columns(
        (pl.col("_weighted") / pl.col("_weighted_max_30d").clip(lower_bound=1.0))
        .clip(lower_bound=0.0, upper_bound=1.0)
        .alias("cyber_threat_index_7d")
    )

    # Select only the output columns, dropping intermediate _weighted and _weighted_max_30d
    return daily.select(["date"] + CYBER_THREAT_FEATURE_COLS)


def join_cyber_threat_features(
    df: pl.DataFrame,
    threats_dir: Path,
) -> pl.DataFrame:
    """Left-join cyber threat features to df by date. Missing dates zero-fill.

    Args:
        df: Input DataFrame with a 'date' column (pl.Date).
        threats_dir: Root of raw cyber threat parquet tree.

    Returns:
        df with CYBER_THREAT_FEATURE_COLS appended. All values are Float64.
        Zero-filled (not null) when no threat data is available.
    """
    features = build_cyber_threat_features(threats_dir)

    if features.is_empty():
        for col in CYBER_THREAT_FEATURE_COLS:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        return df

    result = df.join(features, on="date", how="left")

    # Zero-fill any dates not in the threat data
    for col in CYBER_THREAT_FEATURE_COLS:
        result = result.with_columns(pl.col(col).fill_null(0.0))

    return result
