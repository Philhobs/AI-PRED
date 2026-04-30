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


# ── Multi-horizon label tests ─────────────────────────────────────────────────


def test_build_multi_horizon_labels_has_expected_columns(tmp_path):
    """Returns a DataFrame with one label column per requested horizon."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=400)

    from processing.label_builder import build_multi_horizon_labels
    horizons = {"5d": 5, "20d": 20, "252d": 252}
    result = build_multi_horizon_labels(ohlcv_dir=ohlcv_dir, horizons=horizons)

    assert "ticker" in result.columns
    assert "date" in result.columns
    assert "label_return_5d" in result.columns
    assert "label_return_20d" in result.columns
    assert "label_return_252d" in result.columns


def test_build_multi_horizon_labels_correct_shift_values(tmp_path):
    """label_return_5d[0] == close_price[5] / close_price[0] - 1."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=400)

    from processing.label_builder import build_multi_horizon_labels
    result = build_multi_horizon_labels(
        ohlcv_dir=ohlcv_dir, horizons={"5d": 5, "252d": 252}
    ).sort(["ticker", "date"])

    # close_price[0] = 100.0, close_price[5] = 100.0 + 5*0.1 = 100.5
    expected_5d = 100.5 / 100.0 - 1  # 0.005
    assert result["label_return_5d"][0] == pytest.approx(expected_5d, rel=1e-4)

    # close_price[0] = 100.0, close_price[252] = 100.0 + 252*0.1 = 125.2
    expected_252d = 125.2 / 100.0 - 1  # 0.252
    assert result["label_return_252d"][0] == pytest.approx(expected_252d, rel=1e-4)


def test_build_multi_horizon_labels_row_count(tmp_path):
    """With n=300 rows, 5d yields 295 non-null rows; 756d yields 0 non-null rows."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_multi_horizon_labels
    result = build_multi_horizon_labels(
        ohlcv_dir=ohlcv_dir, horizons={"5d": 5, "756d": 756}
    )

    assert result["label_return_5d"].null_count() == 5  # last 5 rows are null
    assert result["label_return_756d"].null_count() == 300  # all null (300 < 756)


def test_build_multi_horizon_labels_returns_empty_for_missing_data(tmp_path):
    """Returns empty DataFrame when no OHLCV data exists."""
    ohlcv_dir = tmp_path / "ohlcv_empty"
    ohlcv_dir.mkdir()

    from processing.label_builder import build_multi_horizon_labels
    result = build_multi_horizon_labels(
        ohlcv_dir=ohlcv_dir, horizons={"5d": 5, "252d": 252}
    )

    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()


# ── Excess (layer-residualized) labels — Phase B ─────────────────────────────


def _write_const_return_ohlcv(ohlcv_dir: Path, ticker: str, daily_return: float, n: int = 30) -> None:
    """Write n trading days where the daily return is exactly daily_return.

    Lets us compute deterministic forward returns for each ticker.
    """
    path = ohlcv_dir / ticker / "2024.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    prices = [100.0 * ((1.0 + daily_return) ** i) for i in range(n)]
    pq.write_table(pa.table({
        "ticker": [ticker] * n,
        "date":  pa.array(dates, type=pa.date32()),
        "open":  prices,
        "high":  [p + 0.01 for p in prices],
        "low":   [p - 0.01 for p in prices],
        "close_price": prices,
        "volume": [1] * n,
    }), str(path))


def test_build_multi_horizon_excess_labels_residualizes_against_layer_mean(tmp_path):
    """For each (date, layer): label_excess = ticker_return - layer_mean. Mean within
    a (date, layer) is exactly 0."""
    from processing.label_builder import build_multi_horizon_excess_labels

    ohlcv_dir = tmp_path / "ohlcv"
    # Two tickers in the same layer with different daily returns: NVDA at +1%/d,
    # AMD at -1%/d. Layer mean per date is 0 → label_excess = label_return.
    _write_const_return_ohlcv(ohlcv_dir, "NVDA",  0.01, n=30)
    _write_const_return_ohlcv(ohlcv_dir, "AMD",  -0.01, n=30)
    ticker_layers = {"NVDA": "compute", "AMD": "compute"}

    result = build_multi_horizon_excess_labels(
        ohlcv_dir=ohlcv_dir,
        horizons={"5d": 5},
        ticker_layers=ticker_layers,
    )
    assert "label_excess_5d" in result.columns

    # Within each date, the two excess values must sum to 0 (residualized vs mean).
    by_date = result.group_by("date").agg(pl.col("label_excess_5d").sum().alias("s"))
    sums = by_date["s"].drop_nulls().to_list()
    assert all(abs(s) < 1e-12 for s in sums), (
        f"layer-residualized excess returns don't sum to 0 within (date, layer); sums: {sums[:5]}"
    )

    # And NVDA's excess > AMD's excess on every populated date.
    nvda = result.filter(pl.col("ticker") == "NVDA").drop_nulls("label_excess_5d")
    amd  = result.filter(pl.col("ticker") == "AMD") .drop_nulls("label_excess_5d")
    assert nvda.height == amd.height
    paired = nvda.join(amd, on="date", suffix="_amd")
    deltas = (paired["label_excess_5d"] - paired["label_excess_5d_amd"]).to_list()
    assert all(d > 0 for d in deltas), "NVDA (+1%) should outperform AMD (-1%) on every date"


def test_build_multi_horizon_excess_labels_drops_unmapped_tickers(tmp_path):
    """Tickers not in the ticker_layers map are excluded from the output."""
    from processing.label_builder import build_multi_horizon_excess_labels

    ohlcv_dir = tmp_path / "ohlcv"
    _write_const_return_ohlcv(ohlcv_dir, "NVDA",   0.01, n=20)
    _write_const_return_ohlcv(ohlcv_dir, "UNKNOWN", 0.01, n=20)

    result = build_multi_horizon_excess_labels(
        ohlcv_dir=ohlcv_dir,
        horizons={"5d": 5},
        ticker_layers={"NVDA": "compute"},  # UNKNOWN omitted
    )
    assert set(result["ticker"].unique().to_list()) == {"NVDA"}


def test_build_multi_horizon_excess_labels_returns_empty_for_missing_data(tmp_path):
    """Returns empty DataFrame with the excess-column schema when no OHLCV exists."""
    from processing.label_builder import build_multi_horizon_excess_labels

    result = build_multi_horizon_excess_labels(
        ohlcv_dir=tmp_path / "missing",
        horizons={"5d": 5, "252d": 252},
        ticker_layers={"NVDA": "compute"},
    )
    assert result.is_empty()
    assert "label_excess_5d" in result.columns
    assert "label_excess_252d" in result.columns
    assert "label_return_5d" not in result.columns  # no raw labels in the excess output
