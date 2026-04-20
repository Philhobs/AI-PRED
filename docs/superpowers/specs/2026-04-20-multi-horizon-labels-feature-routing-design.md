# Multi-Horizon Labels + Feature Routing Design

## Problem

The current model trains a single 1-year forward return label (`label_return_1y = shift(-252)`)
against 48 features that span wildly different signal decay timescales:

| Feature group | Signal decay | Trained on 1y label? |
|---|---|---|
| `SENTIMENT_FEATURE_COLS` (7-day window) | Days | Yes — noise |
| `INSIDER_FEATURE_COLS` (30/90d windows) | Weeks | Yes — mostly noise |
| `SHORT_INTEREST_FEATURE_COLS` | Days–weeks | Yes — noise |
| `FUND_FEATURE_COLS` | Quarters–years | Yes — valid |
| `GRAPH_FEATURE_COLS` | Years | Yes — valid |
| `ENERGY_FEATURE_COLS` | Years | Yes — valid |

Short-decay features contribute statistical noise at the 1-year horizon, diluting the
model's predictive capacity and making feature importance misleading. The fix is to
train separate models per horizon, each with only the features whose signal persists
over that horizon.

## Goal

Produce multi-horizon predictions across 8 horizons (5d, 20d, 65d, 252d, 756d, 1260d,
2520d, 5040d) with each horizon trained on a curated feature subset. Inference returns
a ranked prediction table per horizon. The 252d (1y) horizon remains backward-compatible
with existing consumers.

## Horizons and Feature Tiers

### Horizon registry

```python
# models/train.py — HORIZON_CONFIGS
HORIZON_CONFIGS = {
    "5d":   {"shift": 5,    "tier": "short"},
    "20d":  {"shift": 20,   "tier": "short"},
    "65d":  {"shift": 65,   "tier": "medium"},
    "252d": {"shift": 252,  "tier": "medium"},   # ← current 1y model
    "756d": {"shift": 756,  "tier": "long"},
    "1260d":{"shift": 1260, "tier": "long"},
    "2520d":{"shift": 2520, "tier": "long"},
    "5040d":{"shift": 5040, "tier": "long"},
}
```

### Feature tiers

| Tier | Feature groups included |
|---|---|
| `short` | `PRICE_FEATURE_COLS`, `SENTIMENT_FEATURE_COLS`, `INSIDER_FEATURE_COLS`, `SHORT_INTEREST_FEATURE_COLS` |
| `medium` | All 48 features (`FEATURE_COLS`) |
| `long` | `PRICE_FEATURE_COLS`, `FUND_FEATURE_COLS`, `EARNINGS_FEATURE_COLS`, `GRAPH_FEATURE_COLS`, `OWNERSHIP_FEATURE_COLS`, `ENERGY_FEATURE_COLS`, `SUPPLY_CHAIN_FEATURE_COLS`, `FX_FEATURE_COLS` |

`long` tier **excludes** sentiment, insider, and short interest (stale noise by year 3).
`short` tier **excludes** graph, energy, supply chain, and FX (too slow-moving to register
in 5–20 day windows).

### Minimum label coverage gate

A horizon is skipped for a layer if the dataset has fewer than 100 non-null label rows
after shifting. This handles long-horizon data scarcity gracefully without crashing.

```
2018 data start → ~2026 gives:
  5d/20d/65d/252d/756d/1260d — sufficient rows (3y+ back)
  2520d (10y)  — insufficient (only 8y of data)
  5040d (20y+) — insufficient (only 8y of data)
```

The 2520d and 5040d models will train once more data accumulates. Until then they are
skipped with a `_LOG.warning("Horizon {tag}: only {n} labeled rows, need ≥100 — skipping")`.

## File Changes

### `processing/label_builder.py`

**Add** `build_multi_horizon_labels(ohlcv_dir, horizons)` alongside existing `build_labels`.

Returns a wide DataFrame:
```
ticker | date | label_return_5d | label_return_20d | label_return_65d |
        label_return_252d | label_return_756d | label_return_1260d |
        label_return_2520d | label_return_5040d
```

Each column uses `shift(-N).over("ticker")`. Rows are kept if **any** label is non-null
(the training join handles per-horizon null filtering per column). `build_labels` remains
unchanged for backward compatibility.

### `models/train.py`

**Add** constants:
- `HORIZON_CONFIGS` dict (as above)
- `TIER_FEATURE_COLS: dict[str, list[str]]` mapping `"short"`, `"medium"`, `"long"` to
  their respective feature lists

**Modify** `build_training_dataset` to accept an optional `horizon_tag: str` parameter.
When given, it joins `build_multi_horizon_labels` instead of `build_labels` and returns
the column `label_return_{horizon_tag}` renamed to `label_return`.

**Modify** `train_layer` to accept `horizon_tag: str`. Artifacts are saved to:
```
models/artifacts/layer_{id:02d}_{name}/horizon_{tag}/
```

The artifact directory structure per horizon is identical to the current flat structure
(`lgbm_q10.pkl`, `rf_model.pkl`, `feature_names.json`, etc.), with `feature_names.json`
containing the tier-appropriate feature list (not all 48).

**Modify** `train_all` to loop over `HORIZON_CONFIGS` × `all_layers()`. This produces at
most 8 × 11 = 88 model directories. Walk-forward CV stays unchanged per model.

**`__main__`** gains an optional `--horizon` flag to train a single horizon for iteration:
```bash
python models/train.py --horizon 5d
```

### `models/inference.py`

**Modify** `_predict_layer` to accept `horizon_tag: str` and look in
`layer_dir / f"horizon_{horizon_tag}"`. Feature columns are read from
`feature_names.json` inside that horizon dir (not the global `FEATURE_COLS`).

**Modify** `run_inference` to loop over horizons. When `horizon_tag` is not specified,
it runs all trained horizons. When specified, it runs only that horizon.

Output path: `data/predictions/date={date}/horizon={tag}/predictions.parquet`

The existing `date={date}/predictions.parquet` path is written as an alias for
`horizon=252d` to preserve backward compatibility with any existing scripts.

**Add** `horizon` column to the output DataFrame.

**`__main__`** gains an optional `--horizon` flag:
```bash
python models/inference.py --horizon 5d
```
With no flag, it runs all available horizons.

### `tests/`

- `tests/processing/test_label_builder.py` — extend with tests for `build_multi_horizon_labels`:
  - columns present for all requested horizons
  - shift values are correct (spot-check a known ticker)
  - minimum coverage gate: returns empty for shift > available rows
- `tests/models/test_train.py` — add fixture with 2-layer minimal OHLCV, verify
  horizon artifact directory structure
- `tests/models/test_inference.py` — mock artifact dirs with short-tier features,
  verify `horizon` column in output, verify backward-compat alias path

## Architecture Constraints

- `FEATURE_COLS` (48-feature list) is **not** changed. It remains the union of all
  features and continues to guard column drift in the current (252d) model path.
- `TIER_FEATURE_COLS["medium"]` equals `FEATURE_COLS` — the 252d model is unmodified.
- No new data sources are required. All labels derive from existing OHLCV files.
- Training is sequential per (horizon, layer). No parallelism is added in this sub-project.

## What This Does NOT Cover

- Options-derived features (Sub-project 3)
- International fundamentals (Sub-project 3)
- feature_engineering.py DuckDB assembler (Sub-project 2)
- Government behavioral data (Sub-project 5)
- Transaction cost model (Sub-project 6)
- A UI to browse multi-horizon predictions — inference writes Parquet, consumers read it

## Success Criteria

1. `pytest tests/ -m 'not integration'` passes with no regressions
2. `python models/train.py --horizon 5d` completes without error on dev data
3. `python models/inference.py --horizon 5d` writes `data/predictions/date={d}/horizon=5d/predictions.parquet`
4. 252d backward-compat path still written: `data/predictions/date={d}/predictions.parquet`
5. `feature_names.json` in each horizon artifact contains only the tier's feature list
