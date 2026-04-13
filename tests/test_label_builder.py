import datetime
import pytest
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def _write_ohlcv_fixture(ohlcv_dir: Path, ticker: str = "NVDA", n: int = 300) -> None:
    """Write n consecutive trading days of OHLCV for one ticker."""
    path = ohlcv_dir / ticker / "2024.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    prices = [100.0 + i * 0.1 for i in range(n)]  # linearly increasing
    table = pa.table({
        "ticker": [ticker] * n,
        "date": pa.array(dates, type=pa.date32()),
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low":  [p - 1.0 for p in prices],
        "close_price": prices,
        "volume": [1_000_000] * n,
    })
    pq.write_table(table, str(path))


def test_build_labels_returns_n_minus_252_rows(tmp_path):
    """With 300 rows, exactly 300-252=48 rows have complete forward windows."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir)

    assert len(result) == 300 - 252


def test_build_labels_computes_correct_return(tmp_path):
    """label_return_1y[0] == close_price[252] / close_price[0] - 1."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir).sort(["ticker", "date"])

    # close_price[0] = 100.0, close_price[252] = 100.0 + 252*0.1 = 125.2
    expected_label = 125.2 / 100.0 - 1  # 0.252
    assert result["label_return_1y"][0] == pytest.approx(expected_label, rel=1e-4)


def test_build_labels_has_no_null_labels(tmp_path):
    """All returned rows have non-null label_return_1y."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir)

    assert result["label_return_1y"].null_count() == 0


def test_build_labels_returns_empty_dataframe_for_missing_data(tmp_path):
    """Returns empty DataFrame with correct schema when no OHLCV data exists."""
    ohlcv_dir = tmp_path / "ohlcv_empty"
    ohlcv_dir.mkdir()

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir)

    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()
    assert "ticker" in result.columns
    assert "date" in result.columns
    assert "label_return_1y" in result.columns
