# AI Infrastructure Predictor — Claude Code Instructions

## Context
Financial prediction tool for AI infrastructure and energy stocks (NVDA, TSM, MSFT, AMZN,
GOOGL, META, CEG, VST, VRT, ASML + supply chain). Prediction target: 5-day directional
returns for 50-ticker watchlist.

## Architecture
- Language: Python 3.11+
- Storage: Parquet files (Hive-partitioned by date=YYYY-MM-DD) + DuckDB (no server)
- NLP: FinBERT (ProsusAI/finbert — local CPU inference, ~440MB)
- Model: Temporal Fusion Transformer (pytorch-forecasting) — Phase 2
- Orchestration: APScheduler — Phase 2

## Key rules
1. Never write files to project root — use data/, ingestion/, processing/, models/
2. Each pipeline module is self-contained — check imports before adding new deps
3. Always write Parquet with snappy compression
4. Use DuckDB for all analytical queries — never load full datasets into memory
5. Rate limits: add time.sleep(1) between API calls for free APIs
6. .env must never be committed — use .env.example only
7. All financial calculations must use vectorised operations (Polars or DuckDB)
8. Run tests before marking any pipeline complete: pytest tests/ -m 'not integration'

## Current phase
Phase 1 MVP — free data sources only ($0–7/month)

## Priority order for next session
1. python ingestion/news_ingestion.py    # GDELT + RSS — no key needed
2. python ingestion/financial_ingestion.py  # EDGAR XBRL + FRED + EIA — no key needed
3. python ingestion/flight_ingestion.py     # OpenSky — no key needed
4. python ingestion/ais_ingestion.py        # needs free AISSTREAM_API_KEY from aisstream.io
5. python processing/nlp_pipeline.py        # score news with FinBERT
6. python processing/feature_engineering.py # build feature matrix
