# Planning Seed: Prediction-Market Features (Polymarket + Kalshi)

**Trigger phrase:** "add prediction market features" or "do the polymarket module"
**Effort:** 8-12h end-to-end (3-4h ingestion+contracts curation, 2-3h feature module + tests, 1-2h IC validation, 2-3h integration if validation passes)
**Branch:** start on a fresh sub-branch off `experiment/architecture-v2` (e.g. `experiment/prediction-markets`)
**Created:** 2026-05-03 (proposed by user; refined per code review during Phase C)

## When to invoke

**NOT YET.** Three preconditions:

1. **Phase D ablation must complete first.** Phase D answers "do our existing 112 features actually add IC over a 30-feature baseline?" If existing features already barely beat baseline, adding 6-35 more *novel* features is unlikely to help and harder to validate.
2. **Phase C v2 walk-forward must produce a real IC distribution** so we have a baseline for the IC-validation gate ("does PM add ε on top of feature set X?").
3. **You should have ~30 minutes to manually curate ~10 specific Polymarket contract IDs** before kicking off the work. The contract list IS the differentiator; without curation this becomes a search problem instead of a feature-engineering problem.

If Phase D shows our existing 112 features barely beat baseline, REMOVE features instead of adding — don't trigger this seed.

## Hypothesis under test

Implied probabilities from liquid USD-denominated event contracts on Polymarket and Kalshi have nonzero **incremental** IC against the existing model's residual returns at 5d-252d horizons for the relevant layers.

The strongest a-priori reason to believe this: prediction markets carry **zero filing lag** and **zero amendment risk** — they're real-time public quotes that aggregate forward-looking consensus. PIT correctness is naturally satisfied (`available_date = quote_date`).

## Architecture (refined from the original proposal)

### Modules

```
ingestion/
  prediction_markets.py            # Polymarket Gamma + Kalshi REST, daily snapshots
  prediction_market_contracts.py   # static curation: ~10 contracts ↦ layers ↦ weight
processing/
  prediction_market_features.py    # 6-8 features (NOT 35 — see below)
tools/
  validate_pm_features_ic.py       # IC + ablation gating script
tests/
  test_prediction_market_features.py
```

### Feature column count: 6-8 total, NOT ~35

The original proposal's 4 stats × 8 categories = 32 + 3 cross-cutting = 35 features is too aggressive. Start narrow, expand only after validation.

**v1 feature column list (8):**
- `pm_prob_mean_macro_rates` — mean implied prob, Fed-rate contracts (applies to all layers, weight 0.5)
- `pm_prob_momentum_7d_macro_rates` — 7-day change in same
- `pm_prob_mean_geopolitics_taiwan` — Taiwan/China contracts (semi_equipment / compute / networking)
- `pm_prob_momentum_7d_geopolitics_taiwan` — 7-day change
- `pm_prob_mean_energy` — energy/power milestone contracts (power / cooling / grid)
- `pm_prob_mean_recession` — recession-by-date contracts (enterprise_saas / cloud)
- `pm_prob_mean_ai_milestones` — OpenAI IPO / GPT-X release / etc (cloud / compute)
- `pm_total_volume_log` — log of total USD volume across this layer's relevant contracts (proxy for attention/uncertainty)

If v1 ablation shows IC > 0.03 incremental, expand by adding `_30d` momentum, `_dispersion`, and category breadth. If IC < 0.03 incremental, **delete the feature module** rather than tuning.

### Skip from the original proposal

- `pm_cross_venue_spread_mean` — Polymarket and Kalshi rarely cover the same contract; this would be near-empty. Defer; revisit if IC validation surfaces a specific contract pair on both venues.
- `pm_resolution_proximity_weighted_prob` — clever but adds complexity; first prove vanilla mean+momentum has signal.
- The full 8-category breadth — start with 5 categories (rates, taiwan, energy, recession, ai_milestones); add `export_controls`, `policy`, `risk_appetite` only if validation passes and we have liquid contracts in those categories.

## Hard constraint: historical depth

Kalshi launched mid-2023. Polymarket Gamma's `/prices-history` is patchy for old markets. Realistic historical coverage is **18-30 months**, vs the 26 *years* of price/fundamentals data the rest of the model trains on.

**Implication:** PM features will be null for ~95% of training history. The model will either:
- Learn "ignore this column" (LightGBM does this gracefully via missing-value handling) — SAFE
- Or skew toward recent rows (if we shorten the spine to where PM is non-null) — LOSES decades of training data

Decision: keep training spine long; let PM features be null in pre-2024 history. Validate that this doesn't degrade IC on rows that DON'T have PM data.

## Contract curation gate (highest-leverage decision)

Polymarket has thousands of markets; most are illiquid (<$10k volume) or off-thesis (sports / celebrity / crypto-meme). The actual hard work is **identifying ~10 specific contracts** that satisfy ALL of:

1. ≥$50k cumulative volume (liquidity proxy — narrow bid-ask spreads)
2. Resolves within 12-24 months from quote date (not too far / not too near)
3. Has a clear AI-infra mechanism (e.g. "rate cut by Q3" → discount-rate effect on enterprise_saas valuations; not "Drake vs Kendrick" → no mechanism)
4. Has at least 30 days of price history (so we can compute 7d momentum)

The ingestion module should ASSERT on these criteria at curation time and refuse to ingest contracts that fail.

## IC validation gate (refined from original)

Original proposal: standalone IC ≥ 0.03 → PASS.

**Refined: ablation IC ≥ 0.01 increment over baseline**, computed as:

```
baseline_ic = corr(predict(X_existing), realized_return)
augmented_ic = corr(predict(X_existing + X_pm), realized_return)
incremental_ic = augmented_ic - baseline_ic
PASS iff incremental_ic >= 0.01 on at least one (layer, horizon) pair
```

This catches the "PM adds standalone signal but it's already subsumed by an existing feature" trap. A Fed-rate-cut contract's information might already be in `phys_ai_cfnai_level` — standalone IC of 0.03 doesn't help if the augmented model is no better than baseline.

`tools/validate_pm_features_ic.py` should:
1. Pick 10 trading days from the last 90 with matured 5d/20d returns
2. For each ticker × date, compute baseline prediction (no PM) and augmented prediction (with PM)
3. Spearman IC on each predicted-vs-realized pairing
4. Print: `(layer, horizon, baseline_ic, augmented_ic, delta_ic, PASS|FAIL)`

## PIT correctness

`available_date = date` for prediction markets. They're real-time public quotes — no lag, no amendment, no restatement. Add the module to `tests/test_point_in_time.py::PIT_CORRECTED_INGESTION_MODULES` once shipped.

## Integration cost (after PASS)

If IC validation passes:
1. Add `PREDICTION_MARKET_FEATURE_COLS` to `models/train.py::FEATURE_COLS`
2. Mirror the join in `models/inference.py` (the `test_train_and_inference_import_same_join_functions` regression test enforces parity)
3. Update test count in `tests/test_train.py::test_feature_cols_total`
4. **Retrain all 16 layers × 8 horizons × 2 targets = 256 model triples** (~14h; resume idempotency from commit `c5bb82d` makes this restartable)
5. Re-run inference; re-run the Phase C harness to confirm the integrated model's IC went UP, not down

If IC validation fails: delete the feature module and the contract curation file. Don't try to tune.

## Out of scope (preserved from original)

- No Limitless, Manifold, Myriad, ForecastEx — Polymarket + Kalshi cover ~95% of liquid USD-denominated event contracts
- No whale/wallet activity tracking — the implied probability already aggregates whales' views
- No live websocket — daily snapshots are sufficient for our horizons (5d minimum)
- No integration into per-layer training before IC validation passes

## Acceptance criteria

- New ingestion + processing + tools modules following our exact pattern
- 6-8 PIT-correct features (NOT 35)
- All existing tests still pass
- Ablation-IC validation script reports PASS or FAIL with quantitative table
- Curated contract list (~10) committed to `prediction_market_contracts.py` with mechanism rationale per entry
- Planning seed status updated to "executed PASS" or "executed FAIL — abandoned" depending on outcome

## What to do if you don't meet a precondition

- **No Phase D yet?** Run Phase D first. It's cheap once C v2 exists.
- **No clear contract list?** Spend 30 min on Polymarket and Kalshi UIs; pick 10 contracts that meet the curation criteria; commit them as a static config; THEN start the ingestion work.
- **Phase D showed our existing features barely beat baseline?** Don't add this module. Instead, prune the existing feature set.
