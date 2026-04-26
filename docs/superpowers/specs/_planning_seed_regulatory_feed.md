# Planning Seed — Regulatory / AI Safety Feed (deferred)

**Status:** Planning seed. Run `/brainstorm` next session.
**Driver:** Trend #9 — regulation/safety creates new software markets. Strategy explicitly asks for: EU AI Act enforcement, US/UK AISI activity, model-weight theft incidents.

## Why

The cyber + enterprise SaaS layers are exposed to regulatory uplift (e.g., compliance mandates create demand for our cyber tickers). We currently have no live feed. News ingestion catches general headlines but doesn't filter for the regulatory subset.

## Scope sketch

**Sources (RSS / API):**

| Source | Type | Content |
|---|---|---|
| EU Commission press releases | RSS | AI Act enforcement actions |
| AISI (US, NIST) updates | RSS | model evaluation announcements, safety guidance |
| AISI (UK) reports | RSS / scrape | analogous to US |
| FTC / SEC AI-related actions | RSS | enforcement or rules |
| Anthropic / OpenAI safety announcements | RSS or scrape | RSP updates, capability thresholds |

**Strategy:** extend `ingestion/news_ingestion.py` with a new "regulatory" RSS source list + keyword filter (`AI Act`, `AI Safety Institute`, `model weights theft`, `RSP`, `Frontier Model Forum`).

**Output:**
```
date (Date), source (Utf8), headline (Utf8), url (Utf8),
keyword_matched (Utf8), severity (Utf8: announcement | enforcement | breach)
```
Saved to: `data/raw/news/regulatory/date=YYYY-MM-DD/events.parquet`

**Features (2-3):**
- `regulatory_event_count_30d` — rolling event count (broad indicator)
- `enforcement_event_count_90d` — narrower, only items classified as "enforcement"
- `breach_event_count_90d` — model-weight theft / safety incidents (rare → high signal)

Apply differentially:
- All tickers: regulatory_event_count_30d (macro)
- Cyber + enterprise SaaS layers: weight more heavily (these benefit from compliance mandates)

## Open design questions

1. **Severity classification — manual rules or LLM?** Rules ("contains 'fine' or 'penalty'" → enforcement) work for ~80% of cases. LLM classifier handles edge cases but adds cost.
2. **Cadence:** RSS feeds are real-time but enforcement events are sporadic. Daily batch pull is fine.
3. **How to handle non-English EU sources?** EU Commission usually publishes in English; if not, translate via LLM or skip.
4. **Threshold for "noise":** AI-related headlines flood news daily. Need narrow keyword set + source whitelist or it'll dominate.

## Implementation effort

- New RSS source list + keyword filter: 1-2 hours
- Severity classification rules: 1-2 hours
- Tests with fixture RSS: 1 hour
- Differential layer-weighted feature: 1 hour
- Total: half day

## Why deferred

- Lower priority than Sequoia ratio / model benchmark gap (those are direct signals; this is supportive)
- Risk of low-signal-to-noise: regulatory events are sporadic and hard to attribute price impact to
- Wants design-time review of source list to avoid headline noise dominating the signal

## Trigger

Run when: user says "regulatory feed" / "EU AI Act tracking". Start with `/brainstorm`.
