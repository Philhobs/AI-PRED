#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/14  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/14  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 3/14  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 4/14  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 5/14  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 6/14  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 7/14  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 8/14  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 9/14  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== 10/14  SAM.gov government contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 11/14  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== 12/14  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== 13/14  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 14/14  BLS JOLTS tech sector job openings ==="
python ingestion/bls_jolts_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
