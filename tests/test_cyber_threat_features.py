"""Tests for processing/cyber_threat_features.py."""
import datetime
from pathlib import Path

import polars as pl
import pytest


def _write_threats(threats_dir: Path, rows: list[dict]) -> None:
    """Write raw threat rows to the Hive-partitioned parquet structure."""
    df = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "source": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )
    for date_val, group in df.group_by("date"):
        date_str = str(date_val[0]) if isinstance(date_val, tuple) else str(date_val)
        out = threats_dir / f"date={date_str}" / "threats.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        group.write_parquet(out, compression="snappy")


def test_build_cyber_threat_features_has_all_columns(tmp_path):
    """build_cyber_threat_features returns all 7 CYBER_THREAT_FEATURE_COLS."""
    threats_dir = tmp_path / "cyber_threat"
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 15), "source": "cisa", "metric": "cisa_kev", "value": 1.0},
    ])

    from processing.cyber_threat_features import build_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = build_cyber_threat_features(threats_dir)

    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"
    assert "date" in result.columns


def test_build_cyber_threat_features_7d_rolling_sum(tmp_path):
    """cve_critical_7d is the 7-day rolling count of cve_critical events."""
    threats_dir = tmp_path / "cyber_threat"
    # 3 critical CVEs on Jan 15, 2 on Jan 16, 2 on Jan 23 (8 days after Jan 15)
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 16), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 16), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 23), "source": "nvd", "metric": "cve_critical", "value": 1.0},
        {"date": datetime.date(2024, 1, 23), "source": "nvd", "metric": "cve_critical", "value": 1.0},
    ])

    from processing.cyber_threat_features import build_cyber_threat_features
    result = build_cyber_threat_features(threats_dir).sort("date")

    jan15 = result.filter(pl.col("date") == datetime.date(2024, 1, 15))
    jan16 = result.filter(pl.col("date") == datetime.date(2024, 1, 16))

    assert jan15["cve_critical_7d"][0] == pytest.approx(3.0)
    assert jan16["cve_critical_7d"][0] == pytest.approx(5.0)  # 3 + 2 within 7 days

    jan23 = result.filter(pl.col("date") == datetime.date(2024, 1, 23))
    # Jan 23 window: (Jan 16, Jan 23] — Jan 15 and Jan 16 events excluded
    assert jan23["cve_critical_7d"][0] == pytest.approx(2.0)


def test_build_cyber_threat_features_index_in_bounds(tmp_path):
    """cyber_threat_index_7d is always in [0, 1]."""
    threats_dir = tmp_path / "cyber_threat"
    # Large spike to trigger normalization
    rows = []
    for d in range(1, 20):
        for _ in range(50):
            rows.append({
                "date": datetime.date(2024, 1, d),
                "source": "nvd",
                "metric": "cve_critical",
                "value": 1.0,
            })
    _write_threats(threats_dir, rows)

    from processing.cyber_threat_features import build_cyber_threat_features
    result = build_cyber_threat_features(threats_dir)

    assert result["cyber_threat_index_7d"].min() >= 0.0
    assert result["cyber_threat_index_7d"].max() <= 1.0


def test_build_cyber_threat_features_missing_dir_returns_empty(tmp_path):
    """Returns empty DataFrame (with correct schema) when threats_dir doesn't exist."""
    from processing.cyber_threat_features import build_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = build_cyber_threat_features(tmp_path / "nonexistent")

    assert result.is_empty()
    for col in ["date"] + CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns, f"Missing column in empty result: {col}"


def test_join_cyber_threat_features_adds_columns(tmp_path):
    """join_cyber_threat_features adds all 7 feature columns to the input df."""
    threats_dir = tmp_path / "cyber_threat"
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
    ])

    df = pl.DataFrame({
        "ticker": ["NVDA", "MSFT"],
        "date": [datetime.date(2024, 1, 15), datetime.date(2024, 1, 15)],
    })

    from processing.cyber_threat_features import join_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = join_cyber_threat_features(df, threats_dir)

    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns, f"Missing column after join: {col}"
    # Original columns preserved
    assert "ticker" in result.columns
    assert len(result) == 2


def test_join_cyber_threat_features_missing_dir_zero_fills(tmp_path):
    """When threats_dir missing, all cyber threat columns are 0.0 (not null)."""
    df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": [datetime.date(2024, 1, 15)],
    })

    from processing.cyber_threat_features import join_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = join_cyber_threat_features(df, tmp_path / "nonexistent")

    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in result.columns
        assert result[col][0] == pytest.approx(0.0), f"{col} should be 0.0 when no data"


def test_join_cyber_threat_features_partial_date_zero_fills(tmp_path):
    """Dates in df with no matching threat data get 0.0 (not null) for all feature columns."""
    threats_dir = tmp_path / "cyber_threat"
    _write_threats(threats_dir, [
        {"date": datetime.date(2024, 1, 15), "source": "nvd", "metric": "cve_critical", "value": 1.0},
    ])

    # df has Jan 15 (has data) and Jan 16 (no data)
    df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date": [datetime.date(2024, 1, 15), datetime.date(2024, 1, 16)],
    })

    from processing.cyber_threat_features import join_cyber_threat_features, CYBER_THREAT_FEATURE_COLS
    result = join_cyber_threat_features(df, threats_dir)

    assert len(result) == 2
    jan16 = result.filter(pl.col("date") == datetime.date(2024, 1, 16))
    for col in CYBER_THREAT_FEATURE_COLS:
        assert jan16[col][0] == pytest.approx(0.0), f"{col} should be 0.0 for missing date"
