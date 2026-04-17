# Phase C-B — Backtest Output Enrichment

**Goal:** Extend the existing walk-forward training loop in `models/train.py` to write richer backtest output: per-fold per-layer metrics and a per-ticker accuracy parquet. No new modules, no new CLI commands.

**Architecture:** train.py already runs 3 walk-forward folds internally with NNLS ensemble weight fitting. After each fold's test-set prediction, capture directional accuracy and IC. Write results to two files after all folds complete.

**Tech Stack:** Python 3.11, Polars, NumPy (already used in train.py), scipy (already used)

---

## 1. What Already Exists

`models/train.py` runs walk-forward CV internally:
- Splits data into 3 folds by date
- Trains LightGBM + RF + Ridge on train portion
- Predicts on test portion
- Fits NNLS weights on test predictions
- Writes `data/backtest/walk_forward_results.json` with IC and hit_rate summary

The existing JSON has: `feature_count`, `ic`, `hit_rate` — a single number each, no per-fold or per-layer breakdown.

---

## 2. Extended Output

### 2a. `data/backtest/walk_forward_results.json` (extended)

Replace the flat summary with full fold detail:

```json
{
  "as_of": "2026-04-17",
  "feature_count": 39,
  "folds": [
    {
      "fold": 1,
      "train_end": "2023-12-31",
      "test_start": "2024-01-01",
      "test_end": "2024-12-31",
      "n_samples": 18240,
      "ic": 0.204,
      "hit_rate": 0.541,
      "top_decile_return": 0.182,
      "per_layer": {
        "cloud":          {"ic": 0.18, "hit_rate": 0.52, "n": 1560},
        "compute":        {"ic": 0.23, "hit_rate": 0.57, "n": 2808},
        "semi_equipment": {"ic": 0.21, "hit_rate": 0.55, "n": 2964},
        "networking":     {"ic": 0.15, "hit_rate": 0.50, "n": 1872},
        "servers":        {"ic": 0.19, "hit_rate": 0.53, "n": 1560},
        "datacenter":     {"ic": 0.25, "hit_rate": 0.58, "n": 1248},
        "power":          {"ic": 0.17, "hit_rate": 0.51, "n": 2496},
        "cooling":        {"ic": 0.20, "hit_rate": 0.54, "n": 1248},
        "grid":           {"ic": 0.22, "hit_rate": 0.56, "n": 936},
        "metals":         {"ic": 0.16, "hit_rate": 0.51, "n": 1872}
      }
    }
  ],
  "summary": {
    "mean_ic": 0.191,
    "mean_hit_rate": 0.533,
    "mean_top_decile_return": 0.162,
    "best_layer": "datacenter",
    "worst_layer": "networking"
  }
}
```

**Metric definitions:**
- `ic` — Pearson correlation between predicted return and actual return (information coefficient). Range [-1, 1]; >0.05 is considered meaningful in financial ML.
- `hit_rate` — fraction of predictions where sign(predicted) == sign(actual). 0.5 = random; >0.53 is considered good.
- `top_decile_return` — mean actual return of the top-10% predicted tickers in the test period. Measures whether high-conviction picks outperform.

### 2b. `data/backtest/per_ticker_accuracy.parquet` (new)

One row per (ticker, fold). Schema:

| Column | Type | Notes |
|--------|------|-------|
| `ticker` | str | |
| `layer` | str | supply chain layer |
| `fold` | int32 | 1, 2, or 3 |
| `test_start` | date | fold test window start |
| `test_end` | date | fold test window end |
| `predicted_return` | float64 | ensemble predicted annual return |
| `actual_return` | float64 | realized return over test window |
| `predicted_direction` | int8 | sign of predicted_return (+1/-1) |
| `actual_direction` | int8 | sign of actual_return (+1/-1) |
| `correct` | bool | predicted_direction == actual_direction |
| `n_observations` | int32 | number of training spine rows for this ticker in fold |

Enables queries like:
- "Which tickers does the model predict correctly most often?" → `group_by(ticker).agg(pl.col("correct").mean())`
- "How does datacenter layer accuracy compare to metals?" → filter by layer
- "Is accuracy improving fold over fold?" → group_by(fold)

---

## 3. Implementation in `models/train.py`

### 3a. Per-fold capture

In the walk-forward loop, after computing test predictions, calculate and store:

```python
# After ensemble prediction on test set
import scipy.stats

fold_actual = y_test  # actual returns
fold_predicted = ensemble_pred  # weighted combination

# IC
ic, _ = scipy.stats.pearsonr(fold_predicted, fold_actual)

# Hit rate
hit_rate = float(np.mean(np.sign(fold_predicted) == np.sign(fold_actual)))

# Top decile return
top_decile_mask = fold_predicted >= np.percentile(fold_predicted, 90)
top_decile_return = float(fold_actual[top_decile_mask].mean())

# Per-layer metrics
layer_metrics = {}
for layer_name in df_test["layer"].unique():
    mask = df_test["layer"] == layer_name
    if mask.sum() < 10:
        continue
    layer_ic, _ = scipy.stats.pearsonr(fold_predicted[mask], fold_actual[mask])
    layer_hit = float(np.mean(np.sign(fold_predicted[mask]) == np.sign(fold_actual[mask])))
    layer_metrics[layer_name] = {
        "ic": round(float(layer_ic), 4),
        "hit_rate": round(layer_hit, 4),
        "n": int(mask.sum()),
    }

# Per-ticker accuracy rows
for ticker in df_test["ticker"].unique():
    tmask = df_test["ticker"] == ticker
    if tmask.sum() == 0:
        continue
    t_pred = float(fold_predicted[tmask].mean())
    t_actual = float(fold_actual[tmask].mean())
    per_ticker_rows.append({
        "ticker": ticker,
        "layer": df_test.loc[tmask, "layer"].iloc[0],
        "fold": fold_idx,
        "test_start": test_start_date,
        "test_end": test_end_date,
        "predicted_return": t_pred,
        "actual_return": t_actual,
        "predicted_direction": int(np.sign(t_pred)),
        "actual_direction": int(np.sign(t_actual)),
        "correct": bool(np.sign(t_pred) == np.sign(t_actual)),
        "n_observations": int(tmask.sum()),
    })
```

### 3b. Write outputs after all folds

```python
# After fold loop completes
backtest_dir = project_root / "data" / "backtest"
backtest_dir.mkdir(parents=True, exist_ok=True)

# Extended JSON
results = {
    "as_of": date.today().isoformat(),
    "feature_count": len(FEATURE_COLS),
    "folds": fold_results,  # list accumulated during loop
    "summary": {
        "mean_ic": round(np.mean([f["ic"] for f in fold_results]), 4),
        "mean_hit_rate": round(np.mean([f["hit_rate"] for f in fold_results]), 4),
        "mean_top_decile_return": round(np.mean([f["top_decile_return"] for f in fold_results]), 4),
        "best_layer": max(
            fold_results[-1]["per_layer"],
            key=lambda k: fold_results[-1]["per_layer"][k]["ic"]
        ),
        "worst_layer": min(
            fold_results[-1]["per_layer"],
            key=lambda k: fold_results[-1]["per_layer"][k]["ic"]
        ),
    }
}
(backtest_dir / "walk_forward_results.json").write_text(json.dumps(results, indent=2))

# Per-ticker parquet
if per_ticker_rows:
    pl.DataFrame(per_ticker_rows).write_parquet(
        backtest_dir / "per_ticker_accuracy.parquet",
        compression="snappy",
    )
_LOG.info("[Backtest] Wrote walk_forward_results.json and per_ticker_accuracy.parquet")
```

---

## 4. Testing

**`tests/test_backtest_output.py`** (new):

- `test_walk_forward_results_json_schema` — load the JSON, assert keys `folds`, `summary`, `feature_count` present; assert `folds` is a list of length ≥ 1; assert each fold has `ic`, `hit_rate`, `top_decile_return`, `per_layer`
- `test_per_ticker_accuracy_schema` — load parquet, assert columns `ticker`, `layer`, `fold`, `correct`, `predicted_return`, `actual_return` present; assert `correct` is bool; assert `predicted_direction` is in {-1, 0, 1}
- `test_per_layer_metrics_coverage` — assert all 10 layer names appear in `per_layer` dict of at least one fold
- `test_hit_rate_in_valid_range` — assert `0 ≤ hit_rate ≤ 1` for all folds

These are schema/integrity tests that run against the actual output files (integration-style, skip if files don't exist with `pytest.importorskip` pattern).

---

## 5. Files Created / Modified

**New:**
```
tests/test_backtest_output.py
data/backtest/per_ticker_accuracy.parquet   ← generated by train.py
```

**Modified:**
```
models/train.py   ← capture fold metrics, per-ticker rows, write extended JSON + parquet
```
