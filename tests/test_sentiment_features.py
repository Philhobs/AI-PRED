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


from datetime import date
import polars as pl


def _make_articles(rows: list[dict]) -> pl.DataFrame:
    """Build a fixture articles DataFrame matching the scored-news schema."""
    return pl.DataFrame(
        {
            "article_date": pl.Series(
                [r["article_date"] for r in rows], dtype=pl.Date
            ),
            "mentioned_tickers": [r["mentioned_tickers"] for r in rows],
            "net_sentiment": pl.Series(
                [r["net_sentiment"] for r in rows], dtype=pl.Float64
            ),
        }
    )


def test_compute_sentiment_mean_7d():
    from processing.sentiment_features import _compute_sentiment_mean_7d

    articles = _make_articles([
        {"article_date": date(2024, 1, 10), "mentioned_tickers": ["NVDA"], "net_sentiment": 0.8},
        {"article_date": date(2024, 1, 12), "mentioned_tickers": ["NVDA"], "net_sentiment": 0.4},
        # AMD article in window — must not affect NVDA mean
        {"article_date": date(2024, 1, 11), "mentioned_tickers": ["AMD"],  "net_sentiment": -0.5},
        # NVDA article outside the 7-day window (Jan 14 - 7 = Jan 7, so Jan 6 is excluded)
        {"article_date": date(2024, 1, 6),  "mentioned_tickers": ["NVDA"], "net_sentiment": -0.9},
    ])
    result = _compute_sentiment_mean_7d(articles, "NVDA", date(2024, 1, 14))
    assert result == pytest.approx(0.6, abs=1e-6)  # (0.8 + 0.4) / 2


def test_compute_sentiment_momentum_14d():
    from processing.sentiment_features import _compute_sentiment_momentum_14d

    articles = _make_articles([
        # Current window: Jan 8–14 (date - 7 to date)
        {"article_date": date(2024, 1, 10), "mentioned_tickers": ["NVDA"], "net_sentiment": 0.8},
        # Prior window: Jan 1–7 (date - 14 to date - 8)
        {"article_date": date(2024, 1, 3),  "mentioned_tickers": ["NVDA"], "net_sentiment": 0.2},
    ])
    result = _compute_sentiment_momentum_14d(articles, "NVDA", date(2024, 1, 14))
    assert result == pytest.approx(0.6, abs=1e-6)  # 0.8 - 0.2


def test_article_count_zero_is_zero_not_null():
    from processing.sentiment_features import _compute_article_count_7d

    # Only AMD articles — NVDA count must be 0, not None
    articles = _make_articles([
        {"article_date": date(2024, 1, 10), "mentioned_tickers": ["AMD"], "net_sentiment": 0.5},
    ])
    result = _compute_article_count_7d(articles, "NVDA", date(2024, 1, 14))
    assert result == 0
    assert result is not None
