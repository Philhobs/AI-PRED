#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/9  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/9  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 3/9  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 4/9  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 5/9  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 6/9  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 7/9  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 8/9  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 9/9  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
