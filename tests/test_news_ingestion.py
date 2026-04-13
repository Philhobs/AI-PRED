import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pyarrow.parquet as pq


def test_fetch_gdelt_events_returns_list():
    """fetch_gdelt_events returns a list of article dicts with required keys."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://example.com/article1",
                "title": "NVIDIA Exports Hit by New Controls",
                "seendate": "20240101T120000Z",
                "socialimage": "0.3",
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.news_ingestion import fetch_gdelt_events
        results = fetch_gdelt_events("semiconductor export control", days_back=1)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["source"] == "gdelt"
    assert results[0]["title"] == "NVIDIA Exports Hit by New Controls"
    assert results[0]["url"] == "https://example.com/article1"
    assert "timestamp" in results[0]
    assert "theme_tags" in results[0]


def test_fetch_gdelt_events_empty_response():
    """fetch_gdelt_events handles empty articles list without error."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"articles": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.news_ingestion import fetch_gdelt_events
        results = fetch_gdelt_events("semiconductor", days_back=1)

    assert results == []


def test_scrape_rss_feeds_writes_parquet(tmp_data_dir):
    """scrape_rss_feeds writes Parquet to output_dir/news/rss/date=.../data.parquet."""
    mock_feed = MagicMock()
    mock_entry = MagicMock()
    mock_entry.get = lambda k, d="": {
        "link": "https://example.com/story",
        "title": "Data Center Energy Demand Surges",
        "summary": "AI workloads drive unprecedented electricity demand.",
    }.get(k, d)
    mock_feed.entries = [mock_entry] * 3

    with patch("feedparser.parse", return_value=mock_feed):
        from ingestion.news_ingestion import scrape_rss_feeds
        scrape_rss_feeds(tmp_data_dir)

    parquet_files = list(tmp_data_dir.glob("news/rss/date=*/data.parquet"))
    assert len(parquet_files) == 1

    table = pq.read_table(parquet_files[0])
    assert table.num_rows > 0
    assert "title" in table.schema.names
    assert "source" in table.schema.names


def test_search_edgar_fulltext_returns_hits():
    """search_edgar_fulltext returns list of EDGAR search hits."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "file_date": "2024-01-15",
                        "entity_name": "NVIDIA Corp",
                        "form_type": "8-K",
                    }
                }
            ]
        }
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.news_ingestion import search_edgar_fulltext
        results = search_edgar_fulltext('"power purchase agreement" AND "data center"', days_back=7)

    assert len(results) == 1
    assert results[0]["_source"]["entity_name"] == "NVIDIA Corp"
