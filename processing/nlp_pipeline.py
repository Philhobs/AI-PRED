from transformers import BertTokenizer, BertForSequenceClassification
import torch
import duckdb
from pathlib import Path


class FinBERTScorer:
    """
    ProsusAI/finbert wrapper for financial sentiment scoring.
    Labels: positive=0, negative=1, neutral=2 (FinBERT ordering).
    net_sentiment = P(positive) - P(negative) → core feature signal.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        path = model_path or self.MODEL_NAME
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.model.to(device)
        self.model.eval()
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Score texts for financial sentiment. Returns [] for empty input."""
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, p in enumerate(probs):
                results.append({
                    "text": batch[j][:200],
                    "positive": float(p[0]),
                    "negative": float(p[1]),
                    "neutral": float(p[2]),
                    "net_sentiment": float(p[0] - p[1]),
                    "label": self.label_map[int(p.argmax())],
                })
        return results

    def score_articles_from_parquet(
        self,
        input_path: str | list[str],
        output_dir: Path,
        text_column: str = "content_snippet",
    ):
        """Score all articles in a Parquet glob (or list of globs) and write scored output.

        Pass a list to score multiple sources (e.g. RSS + GDELT) in a single
        write. DuckDB's read_parquet accepts a list and unions the files.
        """
        df = duckdb.read_parquet(input_path).fetchdf()
        texts = df[text_column].fillna("").tolist()
        scores = self.score_batch(texts)

        scored_df = df.copy()
        for field in ["positive", "negative", "neutral", "net_sentiment", "label"]:
            scored_df[field] = [s[field] for s in scores]

        import pandas as pd
        date_str = pd.to_datetime(df["timestamp"]).dt.date.mode()[0].strftime("%Y-%m-%d")
        out_path = output_dir / "news" / "scored" / f"date={date_str}" / "data.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_parquet(str(out_path), compression="snappy")
        print(f"[NLP] Scored {len(df)} articles → {out_path}")


def compute_daily_sentiment_features(duckdb_conn, date_str: str) -> list:
    """
    Aggregate daily FinBERT scores into model features.
    Requires v_news view (register via storage.create_views).
    Returns empty list if no data for the date.
    """
    import re
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_str):
        raise ValueError(f"date_str must be YYYY-MM-DD, got: {date_str!r}")
    query = f"""
    WITH daily AS (
        SELECT
            date_trunc('day', timestamp) as date,
            AVG(net_sentiment) as mean_sentiment,
            STDDEV(net_sentiment) as sentiment_dispersion,
            COUNT(*) as article_volume,
            SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) as negative_count
        FROM v_news
        WHERE timestamp >= DATE '{date_str}' - INTERVAL '61 days'
        GROUP BY 1
    ),
    with_momentum AS (
        SELECT *,
            mean_sentiment - LAG(mean_sentiment, 7) OVER (ORDER BY date)
                AS sentiment_momentum_7d,
            mean_sentiment - LAG(mean_sentiment, 30) OVER (ORDER BY date)
                AS sentiment_momentum_30d
        FROM daily
    )
    SELECT * FROM with_momentum
    WHERE date = DATE '{date_str}'
    """
    try:
        return duckdb_conn.execute(query).fetchdf().to_dict("records")
    except Exception:
        return []


if __name__ == "__main__":
    import glob
    from dotenv import load_dotenv
    load_dotenv()

    # Score RSS + GDELT together. EDGAR full-text hits are not yet persisted
    # (see _planning_seed_sentiment_coverage.md gap #2); add their glob here
    # once that lands.
    input_globs = [
        "data/raw/news/rss/date=*/data.parquet",
        "data/raw/news/gdelt/date=*/data.parquet",
    ]
    parquet_files = [f for g in input_globs for f in glob.glob(g)]
    if not parquet_files:
        print("[NLP] No RSS or GDELT news parquet files found. Run news_ingestion.py first.")
    else:
        print(f"[NLP] Found {len(parquet_files)} news parquet files across {len(input_globs)} sources.")
        print("[NLP] Loading FinBERT model (first run downloads ~440MB)...")
        scorer = FinBERTScorer()
        scorer.score_articles_from_parquet(input_globs, Path("data/raw"))
