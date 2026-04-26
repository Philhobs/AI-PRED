# Strategy Alignment — Ticker Expansion (Bottlenecks + HBM + SaaS)

**Date:** 2026-04-26
**Status:** Approved
**Driver:** Coverage gap analysis vs. external strategy doc (compute sovereigns + power infra + agentic SaaS).

## Goal

Add **16 new tickers** to close coverage gaps surfaced by an external "AI bottleneck thesis" strategy. One new layer (`enterprise_saas`) for the agentic-software pillar. Ticker count 149 → 165, layer count 15 → 16. Currency support unchanged (KRW already added in Spec 1).

## Motivation

The external strategy document organizes AI exposure around six buckets (compute sovereigns, power/DC infra, govt AI, AI safety, agentic SaaS, biosecurity). Mapping it against our registry surfaced three concrete gaps worth fixing immediately:

1. **Public misses in buckets we already cover** — Palantir (govt AI), Eaton (DC electrical), Cameco (uranium fuel cycle), Cloudflare (edge security).
2. **HBM bottleneck blind spot** — SK Hynix and Samsung are the dominant HBM3/HBM4 suppliers; we have no Korean exposure on the memory side. KRW FX support already in place from Spec 1.
3. **Whole missing pillar (Agentic SaaS)** — the "unhobbling" / enterprise-AI software layer. Hyperscalers ≠ enterprise-SaaS workflow vendors. Distinct economics (NRR-driven, agent-deployment cost vs. inference economics).

Biosecurity (bucket F) is deferred to a separate spec — different cadence, different signal sources.

## Ticker additions (16)

### To existing layers (6)

| Ticker | Layer | Currency | Country | Rationale |
|---|---|---|---|---|
| **PLTR** | `enterprise_saas` (new) | USD | US | See bucket E below — placed there pragmatically; could split to `govt_ai` if it grows |
| **ETN** | `cooling` | USD | IE | Electrical equipment for AI data centers; same tier as VRT/SU.PA |
| **CCJ** | `power` | USD | CA | Uranium fuel cycle — completes the nuclear thesis (CEG/OKLO/SMR/BWX downstream) |
| **NET** | `cyber_pureplay` | USD | US | Cloudflare — edge security + AI-Worker platform |
| **000660.KS** | `compute` | KRW | KR | SK Hynix — HBM3/HBM4 dominant supplier |
| **005930.KS** | `compute` | KRW | KR | Samsung — HBM3 + foundry leader |

### To new `enterprise_saas` layer (10)

| Ticker | Currency | Country | Rationale |
|---|---|---|---|
| **NOW** | USD | US | ServiceNow — workflow automation, govt cloud |
| **CRM** | USD | US | Salesforce — Einstein/Agentforce |
| **ADBE** | USD | US | Adobe — generative-AI in creative |
| **INTU** | USD | US | Intuit — AI in finance/tax |
| **DDOG** | USD | US | Datadog — observability for AI workloads |
| **SNOW** | USD | US | Snowflake — Cortex AI for enterprise data |
| **GTLB** | USD | US | GitLab — code intelligence |
| **TEAM** | USD | AU | Atlassian — knowledge work agents |
| **PATH** | USD | US | UiPath — RPA → AI agents transition |
| **MNDY** | USD | IL | Monday.com — work management AI |
| **PLTR** | USD | US | (listed above) |

**Total in `enterprise_saas`: 11 tickers** (10 + PLTR).

## Layer restructure

Add `enterprise_saas` as **layer id 16** (after `cyber_platform=15`):

```python
LAYER_IDS = {
    ..., "cyber_pureplay": 14, "cyber_platform": 15,
    "enterprise_saas": 16,
}

LAYER_LABELS = {
    ...,
    "enterprise_saas": "Enterprise SaaS / Agentic Software",
}
```

No other layer changes. Existing 15 layers keep their ids.

**Counts after change:** 149 → 165 tickers; 15 → 16 layers.

## Tier assignment

`enterprise_saas` features apply uniformly via the same cross-cutting feature modules — no new feature columns. The `LAYER_IDS[ticker]` value is itself the only "feature" specific to this layer (model learns per-layer relevance). All cross-cutting features (price, fundamentals, sentiment, FX, etc.) continue to apply via the existing joins.

`PHYSICAL_AI_FEATURE_COLS` / `CYBER_THREAT_FEATURE_COLS` etc. are macro features applied to all tickers — no per-layer routing change.

## Test plan

- `tests/test_ticker_registry.py`: ticker count 149 → 165, layer count 15 → 16, new `test_enterprise_saas_layer_populated`, new `test_enterprise_saas_includes_pltr`.
- `tests/test_ticker_registry.py::test_layers_count_matches_layer_ids` already count-agnostic — no change.
- `tests/test_per_layer_models.py::test_train_all_layers_creates_NN_dirs` already count-agnostic — no change.
- `non_usd_tickers()` count: 40 → 42 (+SK Hynix, Samsung). Existing test asserts `40` — needs bump.

## Files touched

| File | Change |
|---|---|
| `ingestion/ticker_registry.py` | +16 `TickerInfo` rows, +1 `LAYER_IDS` entry, +1 `LAYER_LABELS` entry |
| `tests/test_ticker_registry.py` | +2 new tests, count assertions 149→165 / 15→16 / 40→42 |

## Files NOT touched

- `models/train.py` — no new `*_FEATURE_COLS` constant, just consumes the registry generically
- `models/inference.py` — same
- `processing/*.py` — no change (cross-cutting features apply to all tickers)
- `tools/run_refresh.sh` — no change (no new ingestion)
- FX modules — KRW already supported

## Out-of-scope (deferred)

- **Biosecurity pillar** (strategy bucket F) — separate spec: different signal sources (FDA/CDC/sequencing-cost), different financial regime (clinical-trial cycles).
- **Pre-IPO watchlist extension** (Anduril, Scale, Anthropic, Helsing, Apptronik, Figure) — separate one-line PR.
- **FERC interconnection queue resurrection** — strategy explicitly calls out grid queue as a top indicator. Our FERC ingestion is broken (URL gone). Separate spec.
- **CoWoS/HBM capacity feature signal** — would require new ingestion. Not in this scope.
- **Splitting PLTR into a separate `govt_ai` layer** — would be a 1-ticker layer today. Defer until 2-3 more pure-play govt-AI names are public.

## Risks

- **Sentiment data sparsity**: new tickers have no historical news/sentiment records. They'll get null sentiment features until news_ingestion has run for ≥30 days post-add.
- **CIK_MAP**: new US tickers (PLTR, ETN, CCJ, NET, NOW, CRM, ADBE, INTU, DDOG, SNOW, GTLB, PATH, TEAM, MNDY) need CIK entries for EDGAR fundamentals. Without CIK, those features are null. **Out of scope for this spec but flagged.**
- **000660.KS, 005930.KS** Korean fundamentals: EDGAR doesn't cover them. They get null on EDGAR features (already true for other foreign tickers like ABBN.SW, KGX.DE).
