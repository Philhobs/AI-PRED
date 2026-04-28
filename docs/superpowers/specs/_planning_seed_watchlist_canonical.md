# Planning Seed: Canonical Watchlist Source for Ingestion Modules

**Trigger phrase:** "fix watchlist source" or "audit ingestion ticker scope"
**Effort:** 1-2h
**Created:** 2026-04-27 (after enterprise_saas/robotics/cyber layers showed 0 non-null short interest + earnings)

## Problem

Several ingestion modules grab the watchlist from `insider_trading_ingestion.CIK_MAP.keys()` rather than the canonical `ingestion.ticker_registry.TICKERS`. CIK_MAP grew incrementally over time and is currently the legacy ~91-ticker subset — every layer added since (robotics_industrial through enterprise_saas) is missing. Those tickers therefore get no FINRA short volume, no yfinance earnings surprise, etc., and the per-layer training warns `0 non-null` on those features.

## Modules audited

Found 4 occurrences of `list(CIK_MAP.keys())`:

| Module | Status | Reason |
|---|---|---|
| `ingestion/short_interest_ingestion.py:114` | **FIXED 2026-04-27** | Switched to `us_listed_tickers()` — FINRA covers US-listed only |
| `ingestion/earnings_ingestion.py:106` | **FIXED 2026-04-27** | Switched to `TICKERS` — yfinance covers most foreign listings |
| `ingestion/edgar_fundamentals_ingestion.py:747` | OK as-is | Genuinely needs CIKs (SEC submissions API). Coverage gap is fixed by extending CIK_MAP, which we did this session for the 14 new tickers |
| `ingestion/insider_trading_ingestion.py:446, 556` | OK as-is | This is the module that *defines* CIK_MAP. Coverage = CIK_MAP itself |

## What still needs auditing

A grep of the form `list(CIK_MAP.keys())` was exhaustive for that exact pattern, but there may be ingestion modules that hardcoded a list of tickers, copied a subset, or were written before `ticker_registry.TICKERS` existed. Worth a sweep:

```bash
grep -rn "TICKERS\|CIK_MAP\|watchlist\|tickers =" ingestion/ | grep -v test_
```

Specific suspects to verify:
- `news_ingestion.py` — does the GDELT/RSS path tag mentioned tickers from the full registry?
- `insider_trading_ingestion.py` — CIK_MAP still missing entries for any current registry symbol?
- `ownership_ingestion.py` (if exists) — may have its own scope
- Anything querying market cap, options chains, or sector indices

## What was added (2026-04-27)

```python
# ingestion/ticker_registry.py
def us_listed_tickers() -> list[str]:
    """Return sorted list of tickers listed on US exchanges (incl. ADRs)."""
    return sorted(t.symbol for t in TICKERS_INFO if t.exchange == "US")
```

123 tickers (vs 91 in legacy CIK_MAP). Includes PLTR, ISRG, NET, TSLA, the cyber layer, the enterprise_saas layer, and all robotics tickers. Excludes SAP.DE, 6954.T, etc.

## Success criteria

- After re-running short_interest + earnings ingestion, `Joined short interest features` and `Joined earnings features` log lines from `models/train.py` show non-zero counts for layers 11-16
- A new grep audit confirms no remaining ingestion module pulls watchlist from CIK_MAP unless it specifically needs CIKs
