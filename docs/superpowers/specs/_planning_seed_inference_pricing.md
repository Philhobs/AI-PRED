# Planning Seed — Inference Pricing Tracker (deferred)

**Status:** Planning seed. Run `/brainstorm` next session.
**Driver:** Trend #3 — inference cost per token dropped 280x in 2 years (Stanford AI Index). Direct measure of whether the inference economics curve is still bending the right way.

## Why

If cost per token keeps falling, AI applications become more profitable. If it stalls (e.g., due to HBM bottlenecks or capacity constraints), application-layer margins compress. We currently have no live tracker.

## Scope sketch

**Source: scrape API pricing pages of frontier providers.**

| Provider | Page | Models to track |
|---|---|---|
| OpenAI | platform.openai.com/docs/pricing | GPT-4o, GPT-5 (whatever's current frontier) |
| Anthropic | docs.anthropic.com/en/api/pricing | Claude Opus, Sonnet, Haiku (latest) |
| Google | ai.google.dev/pricing | Gemini Pro, Flash |
| Together / Fireworks (open hosting) | together.ai/pricing | Llama 3.1, Mixtral, DeepSeek |

**Output schema:**
```
date (Date), provider (Utf8), model (Utf8), 
input_per_1m_tokens (Float64), output_per_1m_tokens (Float64),
context_window (Int32)
```
Saved to: `data/raw/inference_pricing/date=YYYY-MM-DD/prices.parquet`

**Features (3):**
- `frontier_input_cost_yoy` — yoy change in cheapest frontier input price
- `frontier_output_cost_yoy` — yoy change in cheapest frontier output price
- `closed_vs_open_inference_premium` — closed cost / equivalent-quality open cost

Apply uniformly to all tickers (macro). Tier: medium + long.

## Open design questions

1. **Cadence?** Pricing pages change ~quarterly. Monthly snapshot is fine; weekly is overkill.
2. **Scrape vs API?** No provider offers a real pricing API. Scraping HTML is brittle. Could use Common Crawl / web archive snapshots as a backup data source for resilience.
3. **"Equivalent quality" matching for closed-vs-open premium?** Hardest part. Match GPT-5 ↔ DeepSeek-R1-equivalent based on a benchmark score band. Requires the model_benchmark_gap signal as input — design these together.
4. **Cache invalidation:** providers occasionally rename models (GPT-4 → GPT-4o → GPT-5), need a model-equivalence registry.

## Implementation effort

- Scraping ~4 HTML pages monthly: 1-2 hours
- Cleanup logic for model renaming: 1-2 hours
- Tests with fixture HTML: 1-2 hours
- Total: ~half day

## Why deferred

- Best implemented after model_benchmark_gap (provides the quality-band matching needed for the premium feature)
- Brittle to layout changes; want to design retry/version-pinning carefully

## Trigger

Run when: user says "inference pricing" / "track cost per token". Start with `/brainstorm`. Recommend doing model_benchmark_gap first.
