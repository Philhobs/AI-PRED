# Planning Seed — Model Benchmark Gap Ingestion (deferred)

**Status:** Planning seed. Run `/brainstorm` next session.
**Driver:** Stanford AI Index trend #5 — open-vs-closed model benchmark gap (8% → 1.7% in one year). Direct signal for "is the open-source race commoditising application-layer AI?"

## Why

If open-weight models keep closing the gap, generic AI app companies (chatbot wrappers, copilots without proprietary data) lose pricing power. Most of our SaaS layer (NOW, CRM, ADBE, INTU) is partly exposed to this. We currently have no live signal for the gap.

## Scope sketch

**Source candidates (in priority order):**
1. **Hugging Face Open LLM Leaderboard** — JSON-y API or GitHub-mirrored CSV; updates frequently; covers MMLU/GPQA/MATH/IFEval/HumanEval
2. **Artificial Analysis** (artificialanalysis.ai) — has a quality vs price chart; less obvious API
3. **LMSys Chatbot Arena ELO** — leaderboard with open vs closed columns

**Recommended:** start with HF Open LLM Leaderboard (most accessible, weekly cadence works).

**Output schema:**
```
date (Date), benchmark (Utf8), best_open_score (Float64), 
best_closed_score (Float64), gap_pct (Float64)
```
Saved to: `data/raw/model_benchmarks/{benchmark}.parquet`

**Features (2-3):**
- `open_vs_closed_gap_mmlu_pct` — current gap on MMLU
- `open_vs_closed_gap_avg_pct` — average across benchmarks
- `gap_yoy_change` — direction of convergence (negative = open closing the gap)

Apply uniformly to all tickers (macro). Tier: medium + long.

## Open design questions

1. **Which benchmarks to track?** MMLU is the canonical headline; HumanEval matters for coding agents; GPQA is the new frontier. Pick 1 headline + 1 secondary, or aggregate across.
2. **Weekly vs monthly cadence?** Leaderboard updates weekly but the gap moves slowly — monthly may be sufficient.
3. **How to handle proprietary models that don't release scores?** GPT-5 / Claude 4.5+ don't always publish on the same benchmarks as open models. Use last-published or skip.

## Why deferred

- HF API quirks (no official endpoint, may need to scrape the leaderboard's underlying dataset HF Hub)
- Cadence + benchmark choice should be deliberately picked — not just "implement everything HF has"
- Wants a /brainstorm flow

## Trigger

Run when: user says "lets do model benchmarks" / "build the open-vs-closed feed". Start with `/brainstorm`.
