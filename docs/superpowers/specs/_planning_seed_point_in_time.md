# Planning Seed: Point-in-Time Correctness (Architecture v2 — Phase A)

**Branch:** `experiment/architecture-v2`
**Goal:** Every feature value joined at training spine date `T` must have been *publicly knowable* by `T`.
**Status:** In progress

## Why this matters

Without this, every backtest IC, hit-rate, top-decile-return result is potentially contaminated by lookahead. The IC we observe today is partly *future-IC*: the model has seen 13F holdings, fundamentals, BLS data, and patent filings *before any human in the market could*. Until this is fixed, no validation tells the truth — including validation of any *other* improvement.

## Audit (2026-04-28)

| Module | Current join key | True availability | Lag | Severity |
|---|---|---|---|---|
| **patent_features** | `filing_date` | `publication_date` (~18 mo) | +540 days | 🚨 CRITICAL |
| **fundamental_features** | `period_end` | 10-Q/10-K `filed_date` | +40-60 days | 🚨 HIGH |
| **ownership_features (13F)** | `period_end` | `filed_date` | +45 days | 🚨 HIGH |
| **ai_economics_features** | `period_end` | release date | +30-60 days | ⚠️ MEDIUM |
| **labor_features (BLS)** | `period_date` | release date | +30-45 days | ⚠️ MEDIUM |
| **census_trade_features** | `period_date` | release date | +50 days | ⚠️ MEDIUM |
| **physical_ai_features (FRED + JOLTS)** | `period_date` | release date | +30-60 days | ⚠️ MEDIUM |
| earnings_features | `earn_date` (announcement) | same | 0 | OK |
| insider_features | Form 4 `filed_date` | same | ≤2d | OK |
| sentiment_features | article timestamp | same | 0 | OK |
| short_interest_features | `si_date` (FINRA daily) | next-day | 1d | OK |
| price / fx / graph | real-time | same | 0 | OK |

## Implementation pattern

Per affected module:

1. **Ingestion side:** persist `available_date` alongside `period_end` / `period_date` in the output parquet.
   - Where the source has the actual filed/published date (USPTO, SEC submissions index, 13F filings index), use it.
   - Otherwise apply a static publication lag.

2. **Processing side:** the asof join uses `available_date` as the right key.

```python
# Lag heuristic (used when actual filed/published date not available)
_PUBLICATION_LAG_DAYS = {
    "fundamentals_10q":    45,   # SEC requires within 40d; pad slightly
    "fundamentals_10k":    60,
    "13f":                 45,
    "bls_jolts":           30,
    "bls_employment":      30,
    "census_intl_trade":   50,
    "fred_default":        45,
}
```

## Order of operations

1. **Patents** — switch `filing_date` → `publication_date`. Highest impact for least code change.
2. **13F ownership** — `filed_date` already in the EDGAR index; just persist + use it.
3. **Fundamentals (10-Q/10-K)** — fetch `filed_date` from the SEC submissions index for each filing.
4. **BLS / Census / FRED / JOLTS** — static publication lag (per the table above).
5. **Regression test:** `tests/test_point_in_time.py` — pick random spine dates, assert no feature row has `available_date > spine_date`.
6. **Rebuild affected feature parquets + retrain** all 9 layers × 8 horizons (or just compare deltas).

## Expected impact

The model's apparent IC will likely **drop** after this fix. That is correct and informative — the prior IC was partly lookahead. The honest IC after this fix is the real starting point for the architecture v2 evaluation.

## Open questions

- For long-history backtests, do we have actual filed_dates for old fundamentals filings? SEC EDGAR submissions API has filing dates for everything since ~2001. For older history, fall back to lag heuristic.
- For FRED series with widely-varying release lags, do we need per-series lookup (via FRED's `release_dates` API endpoint) or is a generic 45-day default acceptable? Generic is fine for v1.
- Should the v0.1 baseline (master tag) be re-tagged with a "v0.1 — INVALID for backtest, contains lookahead" note? Probably yes, as a future-self warning.

## Done criteria

- All 7 affected modules persist + use `available_date`
- Regression test passes; no feature value violates point-in-time
- Affected feature parquets rebuilt
- Affected layers retrained on the experiment branch
- Comparison report: post-fix IC vs. v0.1 baseline IC (expect drop on lookahead-heavy features)
