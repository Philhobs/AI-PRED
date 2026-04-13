import pytest
from unittest.mock import patch, MagicMock
import torch


def test_score_batch_returns_correct_fields():
    """score_batch returns list of dicts with positive/negative/neutral/net_sentiment/label."""
    with patch("processing.nlp_pipeline.BertTokenizer") as MockTok, \
         patch("processing.nlp_pipeline.BertForSequenceClassification") as MockModel:

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        MockTok.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        # logits: positive=2.0, negative=0.5, neutral=0.5
        mock_model.return_value = MagicMock(logits=torch.tensor([[2.0, 0.5, 0.5]]))
        mock_model_instance = mock_model
        MockModel.from_pretrained.return_value = mock_model_instance

        from processing.nlp_pipeline import FinBERTScorer
        scorer = FinBERTScorer(device="cpu")
        results = scorer.score_batch(["NVIDIA reports record data center revenue"])

    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) >= {"positive", "negative", "neutral", "net_sentiment", "label"}
    assert r["label"] in ("positive", "negative", "neutral")
    assert r["positive"] > r["negative"]  # logit 2.0 dominates
    assert r["net_sentiment"] == pytest.approx(r["positive"] - r["negative"], abs=1e-5)


def test_score_batch_empty_input():
    """score_batch with empty list returns [] without calling the model."""
    with patch("processing.nlp_pipeline.BertTokenizer") as MockTok, \
         patch("processing.nlp_pipeline.BertForSequenceClassification") as MockModel:
        MockTok.from_pretrained.return_value = MagicMock()
        MockModel.from_pretrained.return_value = MagicMock()

        from processing.nlp_pipeline import FinBERTScorer
        scorer = FinBERTScorer(device="cpu")
        assert scorer.score_batch([]) == []


def test_compute_daily_sentiment_features_returns_list(tmp_path):
    """compute_daily_sentiment_features runs against minimal parquet without error."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import datetime, timezone

    schema = pa.schema([
        pa.field("timestamp", pa.timestamp("s", tz="UTC")),
        pa.field("net_sentiment", pa.float32()),
        pa.field("label", pa.string()),
    ])
    path = tmp_path / "scored" / "date=2024-01-15" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {"timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
             "net_sentiment": 0.3, "label": "positive"},
            {"timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
             "net_sentiment": -0.1, "label": "negative"},
        ],
        schema=schema,
    )
    pq.write_table(table, path)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW v_news AS SELECT * FROM read_parquet('{tmp_path}/scored/date=*/data.parquet')")

    from processing.nlp_pipeline import compute_daily_sentiment_features
    result = compute_daily_sentiment_features(con, "2024-01-15")

    assert isinstance(result, list)
