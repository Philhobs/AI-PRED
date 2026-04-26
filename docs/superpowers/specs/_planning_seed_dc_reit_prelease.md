# Planning Seed — Data Center REIT Pre-Lease Tracking (deferred)

**Status:** Planning seed. Run `/brainstorm` next session.
**Driver:** Trend #8 — "data-center finance becoming a separate asset class" with key risk being whether sites have power and long-term leases. Strategy explicitly asks: pre-leased capacity, tenant quality, power-secured sites.

## Why

We have 8 DC REIT tickers (EQIX, DLR, AMT, CCI, IREN, APLD, 9432.T, CLNX.MC) and currently no signal beyond their generic stock metrics. The leasing rate is the canonical operating metric — full pre-lease at high rents is the bullish signal; rising vacancy or declining lease length is the bearish signal.

## Scope sketch

**Source: NLP on REIT 10-Q / 10-K filings.**

Each major DC REIT publishes:
- Stabilized portfolio occupancy %
- Pre-leased percentage of pipeline
- Average lease term remaining
- Tenant concentration (% revenue from top 5/10 tenants)

These appear in the MD&A section of 10-Qs/10-Ks as natural language. Two extraction approaches:

**Option A — Targeted regex/keyword extraction:**
- Find sentences containing "occupancy", "pre-leased", "stabilised utilization", "weighted average lease term"
- Extract percentages with regex
- Attribute to ticker × period

**Option B — LLM-assisted extraction (better accuracy, more cost):**
- Pass the relevant 10-K sections to a small LLM with a structured schema
- Get back JSON: {occupancy_pct, pre_leased_pct, avg_lease_years, tenant_concentration_pct}

**Recommended:** A first (cheaper, simpler). Upgrade to B if accuracy is poor.

**Output schema:**
```
ticker (Utf8), period_end (Date), occupancy_pct (Float64),
pre_leased_pct (Float64), avg_lease_years (Float64),
tenant_concentration_top5 (Float64)
```
Saved to: `data/raw/financials/dc_reit_metrics/{ticker}.parquet`

**Features (3-4):**
- `dc_reit_occupancy_yoy` — change in occupancy
- `dc_reit_prelease_pct` — current pre-leased pipeline %
- `dc_reit_lease_length_change` — direction of avg lease term

Apply ONLY to DC REIT layer (Layer 6) tickers. Other tickers get null.

## Open design questions

1. **Which 10-Q sections to parse?** REIT supplemental disclosures often have it cleaner than the full 10-K. Some REITs have "Operating Metrics" tables.
2. **A vs B implementation:** if A is brittle, jumping to B means adding LLM dependencies (Anthropic Claude Haiku is cheap enough — ~$0.001 per filing).
3. **Historical backfill:** parse last 4-8 quarters per ticker on first run.

## Implementation effort

- Option A regex extractor + tests: 3-4 hours
- Option B LLM extractor + tests: 4-6 hours (incl. LLM client)
- Per-REIT schema verification (each may format differently): 2-3 hours
- Total: 1 day

## Why deferred

- Not high-frequency: REITs file quarterly, so the signal moves slowly
- More design ambiguity than the simpler signals — need to pick A or B carefully
- Lower priority than Sequoia ratio / model benchmark gap / inference pricing

## Trigger

Run when: user says "DC REIT pre-lease" / "track REIT occupancy". Start with `/brainstorm`.
