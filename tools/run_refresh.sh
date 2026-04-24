#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/16  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/16  EDGAR fundamentals (reads OHLCV for valuation ratios) ==="
python ingestion/edgar_fundamentals_ingestion.py

echo ""
echo "=== 3/16  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 4/16  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 5/16  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 6/16  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 7/16  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 8/16  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 9/16  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 10/16  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== 11/16  SAM.gov government contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 12/16  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== 13/16  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== 14/16  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 15/16  BLS JOLTS tech sector job openings ==="
python ingestion/bls_jolts_ingestion.py

echo ""
echo "=== 16/16  Census international trade (semiconductors + DC equipment) ==="
python ingestion/census_trade_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
