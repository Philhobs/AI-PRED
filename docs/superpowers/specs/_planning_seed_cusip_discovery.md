# Planning Seed: CUSIP Map Coverage for New Layers

**Trigger phrase:** "fix CUSIP coverage" or "extend cusip_map for layers 11-16"
**Effort:** 2-3h (CUSIP-extraction pass) or 1h (manual SEC EDGAR lookups for the 26)
**Created:** 2026-04-27 (after fundamentals fix turned up the same problem in ownership features)

## Problem

`data/raw/financials/cusip_map.json` has 97 entries. The US-listed registry has 123. Therefore 26 US-listed tickers have **zero institutional ownership data** — `processing/ownership_features.py` produces no quarterly parquet for them, and `models/train.py` reports `0 non-null inst_ownership_pct` on layers containing them (11 robotics_industrial, 12 robotics_medical_humanoid, 13 robotics_mcu_chips, 14 cyber_pureplay, 15 cyber_platform).

## Missing tickers (2026-04-27)

```
Layer 11 robotics_industrial: ROK, ZBRA, CGNX, SYM, EMR
Layer 12 robotics_medical_humanoid: ISRG, TSLA
Layer 13 robotics_mcu_chips: TXN, MCHP, ADI
Layer 14 cyber_pureplay: CRWD, ZS, S
Layer 15 cyber_platform: PANW, FTNT, CHKP, CYBR, TENB, QLYS, OKTA, AKAM, RPD, VRNS
Other:                    ERIC, JNPR, STM
```

JNPR is post-acquisition (HPE 2025) and may not need a CUSIP going forward.

## Why this is hard

`ingestion/sec_13f_ingestion.py::parse_holdings_xml` filters every holding through `cusip_map` *during ingest*. Holdings whose CUSIP isn't in the map are dropped — we never persist them. So we can't reverse-engineer CUSIPs from the existing 13F raw parquets; they only contain rows for already-known tickers.

CUSIPs are also not in any free SEC bulk file (the standard `company_tickers.json` has CIKs only). They're licensed by ANNA/CUSIP Global Services.

## Two paths

### Path A: Extract from a re-fetched 13F (preferred, ~2h)

1. Download one large 13F-HR XML (Vanguard/BlackRock CIK 0000102909/0001364742) without the CUSIP filter
2. Parse the full holdings: `(cusip, name_of_issuer, value_usd)`
3. Build a name → CUSIP table for everything in the filing (10k+ rows for a top 5 fund)
4. Match the 26 missing tickers by issuer-name fuzzy match against the registry
5. Write the resulting CUSIPs into `_STATIC_CUSIP_MAP` in `ingestion/build_cusip_map.py`
6. Re-run `python ingestion/build_cusip_map.py` to refresh `cusip_map.json`
7. Re-run `python ingestion/sec_13f_ingestion.py` to backfill 13F data for the new CUSIPs
8. Re-run `python processing/ownership_features.py` to compute features

### Path B: Manual EDGAR full-text search (~1h)

For each of the 26 tickers, search EDGAR full-text for the company name, find a 13F-HR filing that holds them, extract the CUSIP. This is what the original `build_cusip_map.py` docstring says was done for the existing 97 entries.

Path A is cleaner and recurring (any future ticker addition gets free CUSIP discovery). Path B is faster *this once* but doesn't scale.

## Success criteria

- All 123 US-listed registry tickers have a CUSIP map entry
- After re-running 13F ingestion + ownership feature build, layers 11-15 show non-zero `Joined ownership features` non-null counts in `models/train.py` log

## What was already shipped (2026-04-27)

- 27 new CIK_MAP entries added (covers all 29 newly-found US-listed tickers via SEC `company_tickers.json`; CYBR + JNPR not in SEC map, may need manual lookup)
- 31 of 38 missing fundamentals dirs populated via `edgar_fundamentals_ingestion.fetch_edgar_fundamentals` (the other 7 are foreign issuers filing 20-F whose XBRL doesn't hit our concept whitelist — separate issue, see `_planning_seed_foreign_filers_20f.md` if/when written)
