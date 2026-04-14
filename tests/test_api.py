"""
Tests for the FastAPI serving layer (api/main.py).

Eight tests covering:
1. /health returns 200 with correct fields when predictions exist
2. /predictions/latest returns ranked list with expected fields
3. /predictions/{ticker} returns history rows for a valid ticker
4. /predictions/{ticker} returns 404 for unknown ticker
5. /features/{ticker} returns 404 for unknown ticker
6. /health returns 503 when no date=* subdirs exist
7. /predictions/latest returns 503 when no date=* subdirs exist
8. /features/{ticker} returns 503 when no OHLCV parquet files exist
"""
import datetime
import pytest
import polars as pl
from fastapi.testclient import TestClient


def make_client():
    """Create and return a TestClient for the API app."""
    import api.main as main_mod
    client = TestClient(main_mod.app)
    return client, main_mod


@pytest.fixture(scope="module")
def predictions_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("preds")
    date_str = "2026-01-15"
    pred_dir = base / f"date={date_str}"
    pred_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "ticker": ["NVDA", "CEG", "MSFT"],
        "rank": pl.Series([1, 2, 3], dtype=pl.Int32),
        "expected_annual_return": [0.25, 0.18, 0.12],
        "confidence_low": [0.10, 0.05, 0.02],
        "confidence_high": [0.40, 0.31, 0.22],
        "lgbm_return": [0.24, 0.17, 0.11],
        "rf_return": [0.26, 0.19, 0.13],
        "ridge_return": [0.25, 0.18, 0.12],
        "as_of_date": pl.Series(
            [datetime.date(2026, 1, 15)] * 3, dtype=pl.Date
        ),
    })
    df.write_parquet(str(pred_dir / "predictions.parquet"), compression="snappy")
    return base


# ── Test 1: /health returns 200 with correct fields ───────────────────────────

def test_health_returns_ok(predictions_env):
    import api.main as main_mod
    client, _ = make_client()
    original = main_mod.PREDICTIONS_DIR
    main_mod.PREDICTIONS_DIR = predictions_env
    try:
        resp = client.get("/health")
    finally:
        main_mod.PREDICTIONS_DIR = original
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["last_prediction_date"] == "2026-01-15"
    assert body["ticker_count"] == 3


# ── Test 2: /predictions/latest returns ranked list ───────────────────────────

def test_predictions_latest_returns_ranked_list(predictions_env):
    import api.main as main_mod
    client, _ = make_client()
    original = main_mod.PREDICTIONS_DIR
    main_mod.PREDICTIONS_DIR = predictions_env
    try:
        resp = client.get("/predictions/latest")
    finally:
        main_mod.PREDICTIONS_DIR = original
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3
    first = rows[0]
    assert first["ticker"] == "NVDA"
    assert first["rank"] == 1
    assert "expected_annual_return" in first
    assert "as_of_date" in first


# ── Test 3: /predictions/{ticker} returns history for valid ticker ─────────────

def test_predictions_ticker_returns_history(predictions_env):
    import api.main as main_mod
    client, _ = make_client()
    original = main_mod.PREDICTIONS_DIR
    main_mod.PREDICTIONS_DIR = predictions_env
    try:
        resp = client.get("/predictions/CEG")
    finally:
        main_mod.PREDICTIONS_DIR = original
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) >= 1
    assert rows[0]["ticker"] == "CEG"


# ── Test 4: /predictions/{ticker} returns 404 for unknown ticker ───────────────

def test_predictions_unknown_ticker_returns_404(predictions_env):
    client, _ = make_client()
    resp = client.get("/predictions/FAKE")
    assert resp.status_code == 404


# ── Test 5: /features/{ticker} returns 404 for unknown ticker ─────────────────

def test_features_unknown_ticker_returns_404(predictions_env):
    client, _ = make_client()
    resp = client.get("/features/FAKE")
    assert resp.status_code == 404


# ── Test 6: /health returns 503 when no date=* subdirs ────────────────────────

def test_health_503_when_no_predictions(tmp_path):
    import api.main as main_mod
    empty_dir = tmp_path / "empty_preds"
    empty_dir.mkdir()
    client, _ = make_client()
    original = main_mod.PREDICTIONS_DIR
    main_mod.PREDICTIONS_DIR = empty_dir
    try:
        resp = client.get("/health")
    finally:
        main_mod.PREDICTIONS_DIR = original
    assert resp.status_code == 503


# ── Test 7: /predictions/latest returns 503 when no date=* subdirs ────────────

def test_predictions_latest_503_when_no_predictions(tmp_path):
    import api.main as main_mod
    empty_dir = tmp_path / "empty_preds"
    empty_dir.mkdir()
    client, _ = make_client()
    original = main_mod.PREDICTIONS_DIR
    main_mod.PREDICTIONS_DIR = empty_dir
    try:
        resp = client.get("/predictions/latest")
    finally:
        main_mod.PREDICTIONS_DIR = original
    assert resp.status_code == 503


# ── Test 8: /features/{ticker} returns 503 when no OHLCV parquet files ────────

def test_features_ticker_503_when_no_ohlcv(tmp_path):
    import api.main as main_mod
    empty_ohlcv_dir = tmp_path / "empty_ohlcv"
    empty_ohlcv_dir.mkdir()
    client, _ = make_client()
    original = main_mod.OHLCV_DIR
    main_mod.OHLCV_DIR = empty_ohlcv_dir
    try:
        resp = client.get("/features/NVDA")
    finally:
        main_mod.OHLCV_DIR = original
    assert resp.status_code == 503


# ── Dashboard data loading tests ──────────────────────────────────────────────

from dashboard.app import load_latest_predictions, load_ticker_predictions


def test_load_latest_predictions_returns_df(predictions_env):
    import dashboard.app as dash_mod
    original = dash_mod.PREDICTIONS_DIR
    dash_mod.PREDICTIONS_DIR = predictions_env
    try:
        df = load_latest_predictions()
    finally:
        dash_mod.PREDICTIONS_DIR = original
    assert "ticker" in df.columns
    assert "expected_annual_return" in df.columns
    assert len(df) == 3


def test_load_ticker_predictions_filters_correctly(predictions_env):
    import dashboard.app as dash_mod
    original = dash_mod.PREDICTIONS_DIR
    dash_mod.PREDICTIONS_DIR = predictions_env
    try:
        df = load_ticker_predictions("CEG")
    finally:
        dash_mod.PREDICTIONS_DIR = original
    assert len(df) >= 1
    assert all(t == "CEG" for t in df["ticker"].to_list())
