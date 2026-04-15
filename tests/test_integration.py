"""
Integration tests — call real external APIs.
Skip in CI or offline: pytest tests/ -m 'not integration'
Run manually: pytest tests/test_integration.py -m integration -v
"""
import pytest
import polars as pl
from pathlib import Path


@pytest.mark.integration
def test_gdelt_api_returns_articles():
    """GDELT API is reachable and returns article data for AI infrastructure query."""
    from ingestion.news_ingestion import fetch_gdelt_events
    results = fetch_gdelt_events("semiconductor data center", days_back=1)
    assert isinstance(results, list)
    # Empty is acceptable — GDELT may have no matching articles


@pytest.mark.integration
def test_edgar_xbrl_msft_capex_has_records():
    """EDGAR XBRL returns MSFT quarterly capex records, all from 10-K/10-Q."""
    from ingestion.financial_ingestion import fetch_edgar_xbrl, CAPEX_CONCEPT, CIK_MAP
    results = fetch_edgar_xbrl(CIK_MAP["MSFT"], CAPEX_CONCEPT)
    assert len(results) > 10
    assert all(r["form"] in ("10-K", "10-Q", "20-F") for r in results)
    assert all(r["value"] > 0 for r in results)


@pytest.mark.integration
def test_rss_feeds_write_parquet(tmp_path):
    """RSS feeds are reachable, articles are parsed, and Parquet is written."""
    from ingestion.news_ingestion import scrape_rss_feeds
    scrape_rss_feeds(tmp_path)
    parquet_files = list(tmp_path.glob("news/rss/date=*/data.parquet"))
    assert len(parquet_files) == 1


@pytest.mark.integration
def test_opensky_arrivals_at_klax():
    """OpenSky API returns arrivals list for KLAX (may be empty if no flights)."""
    import time
    from ingestion.flight_ingestion import fetch_arrivals_at_airport
    end_ts = int(time.time())
    start_ts = end_ts - 3600  # Last hour
    results = fetch_arrivals_at_airport("KLAX", start_ts, end_ts)
    assert isinstance(results, list)


@pytest.mark.integration
def test_fetch_form4_nvda():
    """Fetch real NVDA Form 4 data from EDGAR. Requires network (~2-5 min)."""
    from ingestion.insider_trading_ingestion import fetch_corporate_insider_trades
    df = fetch_corporate_insider_trades("NVDA")
    assert isinstance(df, pl.DataFrame)
    assert len(df) >= 10, f"Expected >=10 rows, got {len(df)}"
    expected_cols = {
        "ticker", "filed_date", "transaction_date", "insider_name",
        "insider_title", "transaction_code", "shares", "price_per_share", "value",
    }
    assert expected_cols.issubset(set(df.columns))
    assert set(df["transaction_code"].unique().to_list()).issubset({"P", "S"})


@pytest.mark.integration
def test_fetch_congressional_house():
    """Real House Stock Watcher fetch — >=100 records, all required columns present."""
    from ingestion.insider_trading_ingestion import fetch_congressional_trades_house
    df = fetch_congressional_trades_house()
    assert isinstance(df, pl.DataFrame)
    assert len(df) >= 100, f"Expected >=100 rows, got {len(df)}"
    expected_cols = {
        "ticker", "trade_date", "politician_name", "chamber",
        "party", "transaction_type", "amount_low", "amount_high", "amount_mid",
    }
    assert expected_cols.issubset(set(df.columns))
    assert set(df["chamber"].unique().to_list()) == {"house"}


@pytest.mark.integration
def test_news_ingestion_tags_tickers():
    """Real GDELT fetch — articles about NVIDIA get mentioned_tickers=['NVDA']."""
    from ingestion.news_ingestion import fetch_gdelt_events
    results = fetch_gdelt_events("NVIDIA datacenter semiconductor", days_back=7)
    assert isinstance(results, list)
    if not results:
        pytest.skip("GDELT returned no results — acceptable for this query/window")
    # All records must have the mentioned_tickers key
    for rec in results:
        assert "mentioned_tickers" in rec
        assert isinstance(rec["mentioned_tickers"], list)
    # At least one article should tag NVDA given the query
    all_tagged = [t for rec in results for t in rec["mentioned_tickers"]]
    assert "NVDA" in all_tagged, (
        f"Expected NVDA in tagged tickers but got: {set(all_tagged)}"
    )
