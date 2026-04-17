# Backtest Output Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing walk-forward CV loop in `models/train.py` to write richer backtest output — per-fold IC/hit-rate/top-decile metrics with per-layer breakdown, and a per-ticker accuracy parquet.

**Architecture:** The existing `train()` function (not `train_single_layer` / `train_all_layers`) already runs 3 walk-forward folds and stacks predictions. We add metric capture inside that loop, then write two files after it completes: an extended `walk_forward_results.json` and a new `per_ticker_accuracy.parquet`. The `train_all_layers()` path is unchanged.

**Tech Stack:** Python 3.11, NumPy, SciPy (already imported), Polars, json (stdlib)

---

## Context: How train.py Works

`models/train.py` has two training paths:

1. **`train()`** — global model trained on all tickers combined. Runs 3-fold walk-forward CV internally to learn NNLS ensemble weights. Saves one set of artifacts to `artifacts_dir/` directly.

2. **`train_all_layers()`** → calls **`train_single_layer()`** per layer — no CV, just fit on all data. Saves per-layer artifacts.

The backtest enrichment targets **`train()`** only. The fold loop is at lines 390–421. After the loop, predictions are stacked and NNLS weights are solved.

The global `df` (built by `build_training_dataset()` with no `layer=` argument) contains all tickers. It has a `ticker` column and a `date` column, plus `FEATURE_COLS` and `label_return_1y`. It does **not** have a `layer` column — we need to add that by joining with the ticker registry.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `models/train.py` | Modify | Capture fold metrics + write backtest outputs |
| `tests/test_backtest_output.py` | Create | Schema + integrity tests for output files |

---

## Task 1: Capture Per-Fold Metrics in the Walk-Forward Loop

**Files:**
- Modify: `models/train.py:388-421` (inside the fold loop)

- [ ] **Step 1: Write the test first**

```python
# tests/test_backtest_output.py
"""
Tests for walk_forward_results.json and per_ticker_accuracy.parquet.

These are integration-style tests — they load actual output files produced by train.py.
They are skipped if the files don't exist (run train.py first to generate them).
"""
from __future__ import annotations
import json
from pathlib import Path

import polars as pl
import pytest

_BACKTEST_DIR = Path("data/backtest")
_WF_JSON = _BACKTEST_DIR / "walk_forward_results.json"
_TICKER_PARQUET = _BACKTEST_DIR / "per_ticker_accuracy.parquet"

_REQUIRED_LAYERS = {
    "cloud", "compute", "semi_equipment", "networking",
    "servers", "datacenter", "power", "cooling", "grid", "metals",
}


@pytest.fixture
def wf_results():
    if not _WF_JSON.exists():
        pytest.skip("walk_forward_results.json not found — run python models/train.py first")
    return json.loads(_WF_JSON.read_text())


@pytest.fixture
def ticker_df():
    if not _TICKER_PARQUET.exists():
        pytest.skip("per_ticker_accuracy.parquet not found — run python models/train.py first")
    return pl.read_parquet(_TICKER_PARQUET)


def test_walk_forward_json_top_level_keys(wf_results):
    assert "as_of" in wf_results
    assert "feature_count" in wf_results
    assert "folds" in wf_results
    assert "summary" in wf_results
    assert isinstance(wf_results["folds"], list)
    assert len(wf_results["folds"]) >= 1


def test_walk_forward_json_fold_keys(wf_results):
    for fold in wf_results["folds"]:
        assert "fold" in fold
        assert "train_end" in fold
        assert "test_start" in fold
        assert "ic" in fold
        assert "hit_rate" in fold
        assert "top_decile_return" in fold
        assert "per_layer" in fold
        assert 0.0 <= fold["hit_rate"] <= 1.0, "hit_rate must be in [0, 1]"


def test_walk_forward_json_per_layer_coverage(wf_results):
    for fold in wf_results["folds"]:
        layer_names = set(fold["per_layer"].keys())
        # At least half the layers should have data (some may have too few samples)
        assert len(layer_names) >= 5, f"Expected ≥5 layers in per_layer, got {layer_names}"


def test_per_ticker_accuracy_schema(ticker_df):
    required_cols = {
        "ticker", "layer", "fold", "test_start", "test_end",
        "predicted_return", "actual_return",
        "predicted_direction", "actual_direction", "correct",
    }
    assert required_cols.issubset(set(ticker_df.columns)), \
        f"Missing columns: {required_cols - set(ticker_df.columns)}"


def test_per_ticker_accuracy_correct_is_bool(ticker_df):
    assert ticker_df["correct"].dtype == pl.Boolean, \
        f"'correct' column should be Boolean, got {ticker_df['correct'].dtype}"


def test_per_ticker_accuracy_directions_valid(ticker_df):
    valid = {-1, 0, 1}
    pred_dirs = set(ticker_df["predicted_direction"].unique().to_list())
    actual_dirs = set(ticker_df["actual_direction"].unique().to_list())
    assert pred_dirs.issubset(valid), f"Invalid predicted_direction values: {pred_dirs - valid}"
    assert actual_dirs.issubset(valid), f"Invalid actual_direction values: {actual_dirs - valid}"
```

- [ ] **Step 2: Run tests to verify they skip (files don't exist yet)**

```bash
pytest tests/test_backtest_output.py -v
```

Expected: All 6 SKIPPED (files not generated yet — correct pre-condition).

- [ ] **Step 3: Add imports to train.py**

At the top of `models/train.py`, add `scipy.stats` to the existing scipy import:

```python
from scipy.optimize import nnls
import scipy.stats  # add this line
```

Also verify `json` is already imported (it is — line ~16). Confirm `pl` (polars) is imported (it is).

- [ ] **Step 4: Add ticker→layer mapping helper to train.py**

After the existing imports in `models/train.py`, add this helper function before `build_training_dataset()`:

```python
def _ticker_layer_map() -> dict[str, str]:
    """Return {ticker: layer_name} for all registered tickers."""
    from ingestion.ticker_registry import layers as all_layers, tickers_in_layer
    result = {}
    for layer in all_layers():
        for ticker in tickers_in_layer(layer):
            result[ticker] = layer
    return result
```

- [ ] **Step 5: Add fold metric capture inside the walk-forward loop**

In `train()`, find the fold loop (lines 390–421). Replace:

```python
    val_lgbm_preds, val_rf_preds, val_ridge_preds, val_y_all = [], [], [], []

    for split in splits:
        train_dates = dates_sorted[:split]
        val_dates = dates_sorted[split: split + val_window_days]

        train_mask = df["date"].is_in(train_dates)
        val_mask = df["date"].is_in(val_dates)

        X_tr = df.filter(train_mask).select(FEATURE_COLS).to_numpy().astype(float)
        y_tr = df.filter(train_mask)["label_return_1y"].to_numpy().astype(float)
        X_val = df.filter(val_mask).select(FEATURE_COLS).to_numpy().astype(float)
        y_val = df.filter(val_mask)["label_return_1y"].to_numpy().astype(float)

        medians_fold = _compute_medians(X_tr)
        X_tr_imp = _impute(X_tr, medians_fold)
        X_val_imp = _impute(X_val, medians_fold)

        scaler_fold = StandardScaler().fit(X_tr_imp)
        X_tr_sc = scaler_fold.transform(X_tr_imp)
        X_val_sc = scaler_fold.transform(X_val_imp)

        # LightGBM q50 (point estimate) — handles NaN natively (raw features)
        lgbm_fold = lgb.LGBMRegressor(
            objective="quantile", alpha=0.50, verbose=-1, **lgbm_base
        ).fit(X_tr, y_tr, feature_name=FEATURE_COLS)

        rf_fold = RandomForestRegressor(**rf_base).fit(X_tr_imp, y_tr)
        ridge_fold = Ridge(alpha=1.0).fit(X_tr_sc, y_tr)

        val_lgbm_preds.append(lgbm_fold.predict(pd.DataFrame(X_val, columns=FEATURE_COLS)))
        val_rf_preds.append(rf_fold.predict(X_val_imp))
        val_ridge_preds.append(ridge_fold.predict(X_val_sc))
        val_y_all.append(y_val)
```

with:

```python
    val_lgbm_preds, val_rf_preds, val_ridge_preds, val_y_all = [], [], [], []
    fold_results: list[dict] = []
    per_ticker_rows: list[dict] = []
    ticker_layer = _ticker_layer_map()

    for fold_idx, split in enumerate(splits, start=1):
        train_dates = dates_sorted[:split]
        val_dates = dates_sorted[split: split + val_window_days]

        train_mask = df["date"].is_in(train_dates)
        val_mask = df["date"].is_in(val_dates)

        df_val = df.filter(val_mask)
        X_tr = df.filter(train_mask).select(FEATURE_COLS).to_numpy().astype(float)
        y_tr = df.filter(train_mask)["label_return_1y"].to_numpy().astype(float)
        X_val = df_val.select(FEATURE_COLS).to_numpy().astype(float)
        y_val = df_val["label_return_1y"].to_numpy().astype(float)

        medians_fold = _compute_medians(X_tr)
        X_tr_imp = _impute(X_tr, medians_fold)
        X_val_imp = _impute(X_val, medians_fold)

        scaler_fold = StandardScaler().fit(X_tr_imp)
        X_tr_sc = scaler_fold.transform(X_tr_imp)
        X_val_sc = scaler_fold.transform(X_val_imp)

        lgbm_fold = lgb.LGBMRegressor(
            objective="quantile", alpha=0.50, verbose=-1, **lgbm_base
        ).fit(X_tr, y_tr, feature_name=FEATURE_COLS)
        rf_fold = RandomForestRegressor(**rf_base).fit(X_tr_imp, y_tr)
        ridge_fold = Ridge(alpha=1.0).fit(X_tr_sc, y_tr)

        lgbm_val = lgbm_fold.predict(pd.DataFrame(X_val, columns=FEATURE_COLS))
        rf_val = rf_fold.predict(X_val_imp)
        ridge_val = ridge_fold.predict(X_val_sc)

        val_lgbm_preds.append(lgbm_val)
        val_rf_preds.append(rf_val)
        val_ridge_preds.append(ridge_val)
        val_y_all.append(y_val)

        # ── Compute fold metrics ──────────────────────────────────────────────
        # Use equal weights for fold metric computation (NNLS weights not known yet)
        fold_ensemble = (lgbm_val + rf_val + ridge_val) / 3.0

        ic_val, _ = scipy.stats.pearsonr(fold_ensemble, y_val)
        hit_rate = float(np.mean(np.sign(fold_ensemble) == np.sign(y_val)))
        top_mask = fold_ensemble >= np.percentile(fold_ensemble, 90)
        top_decile_return = float(y_val[top_mask].mean()) if top_mask.any() else 0.0

        # Per-layer metrics
        val_tickers = df_val["ticker"].to_list()
        val_layers = [ticker_layer.get(t, "unknown") for t in val_tickers]
        layer_metrics: dict[str, dict] = {}
        for layer_name in set(val_layers):
            lmask = np.array([l == layer_name for l in val_layers])
            if lmask.sum() < 10:
                continue
            l_ic, _ = scipy.stats.pearsonr(fold_ensemble[lmask], y_val[lmask])
            l_hit = float(np.mean(np.sign(fold_ensemble[lmask]) == np.sign(y_val[lmask])))
            layer_metrics[layer_name] = {
                "ic": round(float(l_ic), 4),
                "hit_rate": round(l_hit, 4),
                "n": int(lmask.sum()),
            }

        test_start = str(val_dates[0])
        test_end = str(val_dates[-1])
        train_end = str(train_dates[-1])

        fold_results.append({
            "fold": fold_idx,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "n_samples": int(len(y_val)),
            "ic": round(float(ic_val), 4),
            "hit_rate": round(hit_rate, 4),
            "top_decile_return": round(top_decile_return, 4),
            "per_layer": layer_metrics,
        })

        # Per-ticker accuracy rows
        for ticker in set(val_tickers):
            tmask = np.array([t == ticker for t in val_tickers])
            if not tmask.any():
                continue
            t_pred = float(fold_ensemble[tmask].mean())
            t_actual = float(y_val[tmask].mean())
            per_ticker_rows.append({
                "ticker": ticker,
                "layer": ticker_layer.get(ticker, "unknown"),
                "fold": fold_idx,
                "test_start": test_start,
                "test_end": test_end,
                "predicted_return": t_pred,
                "actual_return": t_actual,
                "predicted_direction": int(np.sign(t_pred)),
                "actual_direction": int(np.sign(t_actual)),
                "correct": bool(np.sign(t_pred) == np.sign(t_actual)),
                "n_observations": int(tmask.sum()),
            })
```

- [ ] **Step 6: Write backtest outputs after the NNLS solve**

After the fold loop, find the NNLS solve block (lines 423–432). After `weights = raw_weights / w_sum ...`, add:

```python
    # ── Write backtest outputs ────────────────────────────────────────────────
    backtest_dir = artifacts_dir.parent.parent / "data" / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    all_ics = [f["ic"] for f in fold_results]
    all_hits = [f["hit_rate"] for f in fold_results]
    all_top = [f["top_decile_return"] for f in fold_results]
    last_per_layer = fold_results[-1]["per_layer"] if fold_results else {}

    wf_output = {
        "as_of": date.today().isoformat(),
        "feature_count": len(FEATURE_COLS),
        "folds": fold_results,
        "summary": {
            "mean_ic": round(float(np.mean(all_ics)), 4),
            "mean_hit_rate": round(float(np.mean(all_hits)), 4),
            "mean_top_decile_return": round(float(np.mean(all_top)), 4),
            "best_layer": max(last_per_layer, key=lambda k: last_per_layer[k]["ic"]) if last_per_layer else None,
            "worst_layer": min(last_per_layer, key=lambda k: last_per_layer[k]["ic"]) if last_per_layer else None,
        },
    }
    (backtest_dir / "walk_forward_results.json").write_text(json.dumps(wf_output, indent=2))

    if per_ticker_rows:
        pl.DataFrame(per_ticker_rows).write_parquet(
            backtest_dir / "per_ticker_accuracy.parquet",
            compression="snappy",
        )
    _LOG.info(
        "[Backtest] Wrote walk_forward_results.json and per_ticker_accuracy.parquet to %s",
        backtest_dir,
    )
```

Note: `artifacts_dir` in `train()` is e.g. `models/artifacts`. Its parent is `models/`, its grandparent is the project root. So `artifacts_dir.parent.parent / "data" / "backtest"` resolves to `data/backtest/`. Verify this is correct for the actual call site in `__main__` where `artifacts_dir = project_root / "models" / "artifacts"`.

- [ ] **Step 7: Verify `date` is imported in train.py**

Check the top of `models/train.py` for `from datetime import date`. If it's missing, add:

```python
from datetime import date
```

- [ ] **Step 8: Run train.py to generate the output files**

```bash
python models/train.py 2>&1 | tail -5
```

Expected output includes:
```
INFO [Backtest] Wrote walk_forward_results.json and per_ticker_accuracy.parquet to data/backtest
[Train] Artifacts → models/artifacts
```

- [ ] **Step 9: Run the backtest tests — they should now pass**

```bash
pytest tests/test_backtest_output.py -v
```

Expected: 6 PASSED (no longer skipped — files now exist).

- [ ] **Step 10: Run full test suite**

```bash
pytest tests/ -m "not integration" -q
```

Expected: All pass.

- [ ] **Step 11: Commit**

```bash
git add models/train.py tests/test_backtest_output.py
git commit -m "feat: extend walk-forward CV to write per-fold metrics and per-ticker accuracy"
```

---

## Task 2: Wire `train()` Into `__main__`

**Important:** `train()` is NOT called by `__main__` — only `train_all_layers()` is. The backtest outputs are written inside `train()`. Add a call to `train()` at the end of `__main__` so backtest outputs are generated on each training run.

**Files:**
- Modify: `models/train.py:479-491` (`__main__` block)

- [ ] **Step 1: Add `train()` call to `__main__`**

In `models/train.py`, in the `if __name__ == "__main__":` block, after `train_all_layers(...)`, add:

```python
    _LOG.info("[Train] Computing walk-forward backtest metrics (global model)...")
    train(
        ohlcv_dir=ohlcv_dir,
        fundamentals_dir=fundamentals_dir,
        artifacts_dir=artifacts_dir / "global",  # separate dir from per-layer
    )
```

Full updated `__main__` block:

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    ohlcv_dir        = project_root / "data" / "raw" / "financials" / "ohlcv"
    fundamentals_dir = project_root / "data" / "raw" / "financials" / "fundamentals"
    artifacts_dir    = project_root / "models" / "artifacts"

    _LOG.info("Training per-layer ensembles for 10 supply chain layers...")
    train_all_layers(ohlcv_dir, fundamentals_dir, artifacts_dir)
    _LOG.info("[Train] All layer artifacts → %s", artifacts_dir)
    print(f"[Train] Artifacts → {artifacts_dir}")

    _LOG.info("Computing walk-forward backtest metrics (global model)...")
    train(
        ohlcv_dir=ohlcv_dir,
        fundamentals_dir=fundamentals_dir,
        artifacts_dir=artifacts_dir / "global",
    )
```

- [ ] **Step 2: Run train.py to generate backtest output**

```bash
python models/train.py 2>&1 | grep -E "(Backtest|global|walk_forward)"
```

Expected: `[Backtest] Wrote walk_forward_results.json and per_ticker_accuracy.parquet to data/backtest`

- [ ] **Step 3: Run backtest tests**

```bash
pytest tests/test_backtest_output.py -v
```

Expected: 6 PASSED

- [ ] **Step 4: Commit**

```bash
git add models/train.py
git commit -m "feat: call global train() from __main__ to generate backtest outputs"
```

---

## Final Check

```bash
# Verify the JSON looks right
python -c "
import json
from pathlib import Path
r = json.loads(Path('data/backtest/walk_forward_results.json').read_text())
print('Folds:', len(r['folds']))
print('Summary:', r['summary'])
print('Layers in fold 1:', list(r['folds'][0]['per_layer'].keys()))
"

# Verify per-ticker parquet
python -c "
import polars as pl
df = pl.read_parquet('data/backtest/per_ticker_accuracy.parquet')
print(df.group_by('ticker').agg(pl.col('correct').mean()).sort('correct', descending=True).head(10))
"
```
