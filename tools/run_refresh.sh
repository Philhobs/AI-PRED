#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
#
# Best-effort: each step's failure is logged but the script continues.
# At the end, tools/pipeline_health.py reports per-source freshness.

cd "$(dirname "$0")/.."

# ── Per-step runner ─────────────────────────────────────────────────────────
declare -a _FAILED_STEPS=()
_step() {
    local step_label="$1"
    shift
    echo ""
    echo "=== ${step_label} ==="
    if ! "$@"; then
        echo "  ⚠ ${step_label} FAILED (continuing)"
        _FAILED_STEPS+=("${step_label}")
    fi
}

echo "Starting full pipeline refresh at $(date)"
python -c "from ingestion.ticker_registry import PENDING_IPO_WATCHLIST; print(f'  ({len(PENDING_IPO_WATCHLIST)} pending-IPO tickers awaiting listing)')"

# ── Foundational price + FX ─────────────────────────────────────────────────
_step "1/25  OHLCV price data"                              python ingestion/ohlcv_ingestion.py
_step "2/25  FX rates (yfinance, 9 currency pairs)"         python ingestion/fx_ingestion.py

# ── Fundamentals ─────────────────────────────────────────────────────────────
_step "3/25  EDGAR fundamentals (reads OHLCV)"              python ingestion/edgar_fundamentals_ingestion.py
_step "4/25  yfinance fundamental ratios (P/E, P/S, ...)"   python ingestion/fundamental_ingestion.py
_step "5/25  AI economics (hyperscaler raw capex+revenue)"  python ingestion/ai_economics_ingestion.py

# ── Market microstructure ───────────────────────────────────────────────────
_step "6/25  Short interest (FINRA)"                        python ingestion/short_interest_ingestion.py
_step "7/25  Earnings surprises"                            python ingestion/earnings_ingestion.py

# ── News + sentiment ────────────────────────────────────────────────────────
_step "8/25  News articles (GDELT + RSS)"                   python ingestion/news_ingestion.py
_step "9/25  NLP sentiment scoring (FinBERT)"               python processing/nlp_pipeline.py
_step "10/25 Sentiment features"                            python processing/sentiment_features.py
_step "11/25 Graph features"                                python processing/graph_features.py

# ── Ownership + insider activity ─────────────────────────────────────────────
_step "12/25 13F institutional holdings (incremental)"      python ingestion/sec_13f_ingestion.py
_step "13/25 Ownership features"                            python processing/ownership_features.py
_step "14/25 Insider trading (Form 4, --years 5)"           python ingestion/insider_trading_ingestion.py --years 5
_step "15/25 Deal ingestion (SEC 8-K material agreements)"  python ingestion/deal_ingestion.py

# ── Government / regulatory signals ──────────────────────────────────────────
_step "16/25 SAM.gov government contract awards"            python ingestion/sam_gov_ingestion.py
_step "17/25 FERC interconnection queue"                    python ingestion/ferc_queue_ingestion.py
_step "18/25 USPTO patent applications + grants + physAI"   python ingestion/uspto_ingestion.py
_step "19/25 USAJOBS federal AI/ML job postings"            python ingestion/usajobs_ingestion.py

# ── Macro indicators ─────────────────────────────────────────────────────────
_step "20/25 Robotics macro signals (FRED 4 series)"        python ingestion/robotics_signals_ingestion.py
_step "21/25 Financial / energy macro (FRED + EIA)"         python ingestion/financial_ingestion.py
_step "22/25 BLS JOLTS sector job openings (NAICS 51+333)"  python ingestion/bls_jolts_ingestion.py
_step "23/25 Census international trade (semis + DC eq.)"   python ingestion/census_trade_ingestion.py

# ── Geographic + threat signals ──────────────────────────────────────────────
_step "24/25 OWID energy geography (country mix)"           python ingestion/energy_geo_ingestion.py
_step "25/25 Cyber threat (NVD CVEs)"                       python ingestion/cyber_threat_ingestion.py

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Refresh complete at $(date) ==="
if (( ${#_FAILED_STEPS[@]} > 0 )); then
    echo "Failed steps (${#_FAILED_STEPS[@]}):"
    for s in "${_FAILED_STEPS[@]}"; do
        echo "  - ${s}"
    done
fi
echo ""
echo "Running pipeline health check..."
python tools/pipeline_health.py || true

echo ""
echo "Run: python models/train.py  (to retrain with fresh data)"
