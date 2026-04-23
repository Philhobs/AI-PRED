#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/12  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/12  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 3/12  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 4/12  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 5/12  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 6/12  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 7/12  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 8/12  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 9/12  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== 10/12  SAM.gov government contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 11/12  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== 12/12  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
