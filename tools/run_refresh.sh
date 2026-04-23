#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/11  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/11  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 3/11  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 4/11  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 5/11  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 6/11  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 7/11  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 8/11  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 9/11  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== 10/11  SAM.gov government contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 11/11  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
