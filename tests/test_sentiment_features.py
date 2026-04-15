"""Unit tests for news sentiment features. No network, no disk."""
import pytest
from ingestion.news_ingestion import _tag_tickers


def test_tag_tickers_single():
    assert _tag_tickers("NVIDIA reports record earnings", "") == ["NVDA"]


def test_tag_tickers_multi():
    result = _tag_tickers("NVIDIA beats AMD in datacenter", "")
    assert result == ["AMD", "NVDA"]


def test_tag_tickers_case_insensitive():
    assert _tag_tickers("nvidia quarterly results", "") == ["NVDA"]


def test_tag_tickers_word_boundary():
    # "Meta" must not fire on "metadata"
    assert _tag_tickers("metadata analysis platform", "") == []


def test_tag_tickers_no_match():
    assert _tag_tickers("central bank raises interest rates", "") == []


def test_tag_tickers_self_alias():
    # Ticker symbols used directly in headlines must be caught
    assert _tag_tickers("MSFT beats estimates", "") == ["MSFT"]


def test_tag_tickers_searches_content_field():
    # The second (content) argument must also be searched
    assert _tag_tickers("", "NVIDIA earnings beat") == ["NVDA"]


def test_tag_tickers_multi_word_alias():
    # Multi-word alias "Amazon Web Services" must match
    assert "AMZN" in _tag_tickers("Amazon Web Services launches new chip", "")
