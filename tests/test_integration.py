"""
Integration tests — call real external APIs.
Skip in CI or offline: pytest tests/ -m 'not integration'
Run manually: pytest tests/test_integration.py -m integration -v
"""
import pytest
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
