"""Unit tests for news sentiment features. No network, no disk."""
import pytest


def test_tag_tickers_single():
    from ingestion.news_ingestion import _tag_tickers
    assert _tag_tickers("NVIDIA reports record earnings", "") == ["NVDA"]


def test_tag_tickers_multi():
    from ingestion.news_ingestion import _tag_tickers
    result = _tag_tickers("NVIDIA beats AMD in datacenter", "")
    assert result == ["AMD", "NVDA"]


def test_tag_tickers_case_insensitive():
    from ingestion.news_ingestion import _tag_tickers
    assert _tag_tickers("nvidia quarterly results", "") == ["NVDA"]


def test_tag_tickers_word_boundary():
    from ingestion.news_ingestion import _tag_tickers
    # "Meta" must not fire on "metadata"
    assert _tag_tickers("metadata analysis platform", "") == []


def test_tag_tickers_no_match():
    from ingestion.news_ingestion import _tag_tickers
    assert _tag_tickers("central bank raises interest rates", "") == []
