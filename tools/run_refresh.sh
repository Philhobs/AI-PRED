#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"
python -c "from ingestion.ticker_registry import PENDING_IPO_WATCHLIST; print(f'  ({len(PENDING_IPO_WATCHLIST)} pending-IPO tickers awaiting listing)')"

echo ""
echo "=== 1/17  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/17  EDGAR fundamentals (reads OHLCV for valuation ratios) ==="
python ingestion/edgar_fundamentals_ingestion.py

echo ""
echo "=== 3/17  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 4/17  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 5/17  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 6/17  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 7/17  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 8/17  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 9/17  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 10/17  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== 11/17  SAM.gov government contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 12/17  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== 13/17  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== 14/17  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 15/17  Robotics macro signals (FRED) ==="
python ingestion/robotics_signals_ingestion.py

echo ""
echo "=== 16/17  BLS JOLTS sector job openings (NAICS 51 + 333) ==="
python ingestion/bls_jolts_ingestion.py

echo ""
echo "=== 17/17  Census international trade (semiconductors + DC equipment) ==="
python ingestion/census_trade_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
