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


def test_compute_ticker_sentiment_features_article_count(tmp_path):
    """Bulk function must return article_count_7d == actual article count, not fanout multiple."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from processing.sentiment_features import compute_ticker_sentiment_features

    # Write two scored news articles for NVDA, one for AMD, all within a 7d window
    news_dir = tmp_path / "scored" / "date=2024-01-14"
    news_dir.mkdir(parents=True)
    from datetime import datetime, timezone
    news_table = pa.table({
        "timestamp": pa.array(
            [
                datetime(2024, 1, 10, tzinfo=timezone.utc),
                datetime(2024, 1, 12, tzinfo=timezone.utc),
                datetime(2024, 1, 11, tzinfo=timezone.utc),
            ],
            type=pa.timestamp("s", tz="UTC"),
        ),
        "mentioned_tickers": pa.array([["NVDA"], ["NVDA"], ["AMD"]], type=pa.list_(pa.string())),
        "net_sentiment": pa.array([0.8, 0.4, 0.2], type=pa.float64()),
    })
    pq.write_table(news_table, news_dir / "data.parquet")

    # Write OHLCV spine with two tickers
    ohlcv_dir = tmp_path / "ohlcv" / "NVDA"
    ohlcv_dir.mkdir(parents=True)
    ohlcv_table = pa.table({
        "ticker": pa.array(["NVDA", "NVDA", "AMD", "AMD"]),
        "date": pa.array(
            [date(2024, 1, 14), date(2024, 1, 13), date(2024, 1, 14), date(2024, 1, 13)],
            type=pa.date32(),
        ),
        "close": pa.array([500.0, 490.0, 150.0, 148.0]),
    })
    pq.write_table(ohlcv_table, ohlcv_dir / "data.parquet")

    result = compute_ticker_sentiment_features(
        scored_news_dir=tmp_path / "scored",
        ohlcv_dir=tmp_path / "ohlcv",
    )

    nvda_jan14 = result.filter(
        (pl.col("ticker") == "NVDA") & (pl.col("date") == date(2024, 1, 14))
    )
    assert len(nvda_jan14) == 1
    # Must be exactly 2 — not inflated by fanout from prior-window or market joins
    assert nvda_jan14["article_count_7d"][0] == 2
    # AMD has 1 article in the window
    amd_jan14 = result.filter(
        (pl.col("ticker") == "AMD") & (pl.col("date") == date(2024, 1, 14))
    )
    assert amd_jan14["article_count_7d"][0] == 1


def test_join_sentiment_features_backward_asof(tmp_path):
    """join_sentiment_features: backward asof join fills features and null-fills missing ticker."""
    from processing.sentiment_features import join_sentiment_features, save_sentiment_features

    # Write NVDA sentiment feature parquet (one row for Jan 14)
    nvda_features = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": pl.Series([date(2024, 1, 14)], dtype=pl.Date),
        "sentiment_mean_7d": [0.6],
        "sentiment_std_7d": [0.2],
        "article_count_7d": pl.Series([2], dtype=pl.Int64),
        "sentiment_momentum_14d": [0.1],
        "ticker_vs_market_7d": [0.05],
    })
    save_sentiment_features(nvda_features, tmp_path)

    # Training DataFrame: NVDA on Jan 14 (exact match) and AMD on Jan 14 (no features)
    training_df = pl.DataFrame({
        "ticker": ["NVDA", "AMD"],
        "date": pl.Series([date(2024, 1, 14), date(2024, 1, 14)], dtype=pl.Date),
    })

    result = join_sentiment_features(training_df, tmp_path)

    nvda_row = result.filter(pl.col("ticker") == "NVDA")
    amd_row = result.filter(pl.col("ticker") == "AMD")

    # NVDA gets its features
    assert nvda_row["sentiment_mean_7d"][0] == pytest.approx(0.6)
    assert nvda_row["article_count_7d"][0] == 2
    # AMD (no parquet written) gets null features
    assert amd_row["sentiment_mean_7d"][0] is None
    assert amd_row["article_count_7d"][0] is None
