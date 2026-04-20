# Multi-Horizon Labels + Feature Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train separate models per prediction horizon (5d → 20y) with each horizon seeing only the features whose signal persists over that timescale, then produce multi-horizon prediction files at inference time.

**Architecture:** Add `build_multi_horizon_labels()` to the label builder to generate all horizon columns in one pass; add `HORIZON_CONFIGS` and `TIER_FEATURE_COLS` constants to `train.py` so that `train_all_layers` loops over `(horizon, layer)` pairs; update inference to loop over trained horizons and write `horizon={tag}/predictions.parquet` outputs, with a backward-compat alias at the flat `predictions.parquet` path for the 252d horizon.

**Tech Stack:** Polars (label shifts, DataFrame ops), LightGBM + RandomForest + Ridge (unchanged models), Python argparse (`--horizon` flag), Parquet snappy compression.

---

## File Map

| File | Change |
|---|---|
| `processing/label_builder.py` | Add `build_multi_horizon_labels()` |
| `models/train.py` | Add `HORIZON_CONFIGS`, `TIER_FEATURE_COLS`; update `_impute`, `_compute_medians`, `build_training_dataset`, `train_single_layer`, `train_all_layers`, `__main__` |
| `models/inference.py` | Update `_impute`, `_predict_layer`, `run_inference`, `__main__` |
| `tests/test_label_builder.py` | Add 4 tests for `build_multi_horizon_labels` |
| `tests/test_train.py` | Add 4 tests for horizon constants, dataset, and artifact layout |
| `tests/test_inference.py` | Add 3 tests for horizon output path, backward-compat path, horizon column |

---

### Task 1: build_multi_horizon_labels

**Files:**
- Modify: `processing/label_builder.py`
- Modify: `tests/test_label_builder.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_label_builder.py` (after the existing tests):

```python
# ── Multi-horizon label tests ─────────────────────────────────────────────────

def test_build_multi_horizon_labels_has_expected_columns(tmp_path):
    """Returns a DataFrame with one label column per requested horizon."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=400)

    from processing.label_builder import build_multi_horizon_labels
    horizons = {"5d": 5, "20d": 20, "252d": 252}
    result = build_multi_horizon_labels(ohlcv_dir=ohlcv_dir, horizons=horizons)

    assert "ticker" in result.columns
    assert "date" in result.columns
    assert "label_return_5d" in result.columns
    assert "label_return_20d" in result.columns
    assert "label_return_252d" in result.columns


def test_build_multi_horizon_labels_correct_shift_values(tmp_path):
    """label_return_5d[0] == close_price[5] / close_price[0] - 1."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=400)

    from processing.label_builder import build_multi_horizon_labels
    result = build_multi_horizon_labels(
        ohlcv_dir=ohlcv_dir, horizons={"5d": 5, "252d": 252}
    ).sort(["ticker", "date"])

    # close_price[0] = 100.0, close_price[5] = 100.0 + 5*0.1 = 100.5
    expected_5d = 100.5 / 100.0 - 1  # 0.005
    assert result["label_return_5d"][0] == pytest.approx(expected_5d, rel=1e-4)

    # close_price[0] = 100.0, close_price[252] = 100.0 + 252*0.1 = 125.2
    expected_252d = 125.2 / 100.0 - 1  # 0.252
    assert result["label_return_252d"][0] == pytest.approx(expected_252d, rel=1e-4)


def test_build_multi_horizon_labels_row_count(tmp_path):
    """With n=300 rows, 5d yields 295 non-null rows; 756d yields 0 non-null rows."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_multi_horizon_labels
    result = build_multi_horizon_labels(
        ohlcv_dir=ohlcv_dir, horizons={"5d": 5, "756d": 756}
    )

    assert result["label_return_5d"].null_count() == 5  # last 5 rows are null
    assert result["label_return_756d"].null_count() == 300  # all null (300 < 756)


def test_build_multi_horizon_labels_returns_empty_for_missing_data(tmp_path):
    """Returns empty DataFrame when no OHLCV data exists."""
    ohlcv_dir = tmp_path / "ohlcv_empty"
    ohlcv_dir.mkdir()

    from processing.label_builder import build_multi_horizon_labels
    result = build_multi_horizon_labels(
        ohlcv_dir=ohlcv_dir, horizons={"5d": 5, "252d": 252}
    )

    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/phila/Documents/AI\ Projects/AI\ Market\ Prediction/AI-PRED
pytest tests/test_label_builder.py::test_build_multi_horizon_labels_has_expected_columns \
       tests/test_label_builder.py::test_build_multi_horizon_labels_correct_shift_values \
       tests/test_label_builder.py::test_build_multi_horizon_labels_row_count \
       tests/test_label_builder.py::test_build_multi_horizon_labels_returns_empty_for_missing_data \
       -v
```

Expected: FAIL with `ImportError: cannot import name 'build_multi_horizon_labels'`

- [ ] **Step 3: Implement build_multi_horizon_labels**

Append to `processing/label_builder.py` after the existing `build_labels` function (after line 54):

```python
def build_multi_horizon_labels(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
    horizons: dict[str, int] | None = None,
) -> pl.DataFrame:
    """
    Compute multi-horizon forward returns for each ticker×date row.

    horizons: mapping of tag → shift_days, e.g. {"5d": 5, "252d": 252}.
    Defaults to the 8-horizon set from HORIZON_CONFIGS in models/train.py.

    Returns DataFrame: ticker (String), date (Date), label_return_{tag} (Float64)...
    Rows are NOT filtered — each column has nulls where the forward window is unavailable.
    Returns empty DataFrame with ticker/date schema when no data exists.
    """
    if horizons is None:
        horizons = {
            "5d": 5, "20d": 20, "65d": 65, "252d": 252,
            "756d": 756, "1260d": 1260, "2520d": 2520, "5040d": 5040,
        }

    glob = str(ohlcv_dir / "*" / "*.parquet")

    try:
        df = (
            pl.scan_parquet(glob)
            .select(["ticker", "date", "close_price"])
            .collect()
        )
    except (FileNotFoundError, pl.exceptions.ComputeError):
        schema = {"ticker": pl.String, "date": pl.Date}
        schema.update({f"label_return_{tag}": pl.Float64 for tag in horizons})
        return pl.DataFrame(schema=schema)

    if df.is_empty():
        schema = {"ticker": pl.String, "date": pl.Date}
        schema.update({f"label_return_{tag}": pl.Float64 for tag in horizons})
        return pl.DataFrame(schema=schema)

    df = df.with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])

    label_exprs = [
        (
            pl.col("close_price").shift(-shift).over("ticker")
            / pl.col("close_price") - 1
        ).alias(f"label_return_{tag}")
        for tag, shift in horizons.items()
    ]

    return df.with_columns(label_exprs).select(
        ["ticker", "date"] + [f"label_return_{tag}" for tag in horizons]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_label_builder.py::test_build_multi_horizon_labels_has_expected_columns \
       tests/test_label_builder.py::test_build_multi_horizon_labels_correct_shift_values \
       tests/test_label_builder.py::test_build_multi_horizon_labels_row_count \
       tests/test_label_builder.py::test_build_multi_horizon_labels_returns_empty_for_missing_data \
       -v
```

Expected: 4 passed

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all previously passing tests still pass

- [ ] **Step 6: Commit**

```bash
git add processing/label_builder.py tests/test_label_builder.py
git commit -m "feat: add build_multi_horizon_labels to label_builder"
```

---

### Task 2: Horizon constants + helper updates in train.py

**Files:**
- Modify: `models/train.py` (lines 105–115 to add constants; lines 255–272 to update helpers)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train.py` (after the existing imports and fixtures):

```python
def test_tier_feature_cols_medium_equals_feature_cols():
    """TIER_FEATURE_COLS['medium'] must be identical to FEATURE_COLS (48 features)."""
    from models.train import FEATURE_COLS, TIER_FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] == FEATURE_COLS


def test_horizon_configs_has_all_eight_horizons():
    """HORIZON_CONFIGS contains exactly the 8 expected horizon tags."""
    from models.train import HORIZON_CONFIGS
    expected = {"5d", "20d", "65d", "252d", "756d", "1260d", "2520d", "5040d"}
    assert set(HORIZON_CONFIGS.keys()) == expected


def test_horizon_configs_tiers_are_valid():
    """Every HORIZON_CONFIGS entry has a tier that exists in TIER_FEATURE_COLS."""
    from models.train import HORIZON_CONFIGS, TIER_FEATURE_COLS
    for tag, cfg in HORIZON_CONFIGS.items():
        assert cfg["tier"] in TIER_FEATURE_COLS, f"Invalid tier for horizon {tag}"


def test_impute_uses_feature_cols_param(tmp_path):
    """_impute fills NaNs only for the provided feature_cols, not global FEATURE_COLS."""
    from models.train import _impute
    import numpy as np

    feature_cols = ["a", "b"]
    X = np.array([[1.0, np.nan], [np.nan, 3.0]])
    medians = {"a": 10.0, "b": 20.0}
    result = _impute(X, medians, feature_cols=feature_cols)

    assert result[0, 1] == pytest.approx(20.0)
    assert result[1, 0] == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py::test_tier_feature_cols_medium_equals_feature_cols \
       tests/test_train.py::test_horizon_configs_has_all_eight_horizons \
       tests/test_train.py::test_horizon_configs_tiers_are_valid \
       tests/test_train.py::test_impute_uses_feature_cols_param \
       -v
```

Expected: FAIL — `TIER_FEATURE_COLS` and `HORIZON_CONFIGS` not yet defined; `_impute` has no `feature_cols` param.

- [ ] **Step 3: Add HORIZON_CONFIGS and TIER_FEATURE_COLS to train.py**

In `models/train.py`, after line 111 (the closing of `FEATURE_COLS`), add:

```python
# ── Horizon registry ──────────────────────────────────────────────────────────
HORIZON_CONFIGS: dict[str, dict] = {
    "5d":   {"shift": 5,    "tier": "short"},
    "20d":  {"shift": 20,   "tier": "short"},
    "65d":  {"shift": 65,   "tier": "medium"},
    "252d": {"shift": 252,  "tier": "medium"},
    "756d": {"shift": 756,  "tier": "long"},
    "1260d":{"shift": 1260, "tier": "long"},
    "2520d":{"shift": 2520, "tier": "long"},
    "5040d":{"shift": 5040, "tier": "long"},
}

TIER_FEATURE_COLS: dict[str, list[str]] = {
    "short": (
        PRICE_FEATURE_COLS
        + SENTIMENT_FEATURE_COLS
        + INSIDER_FEATURE_COLS
        + SHORT_INTEREST_FEATURE_COLS
    ),
    "medium": FEATURE_COLS,
    "long": (
        PRICE_FEATURE_COLS
        + FUND_FEATURE_COLS
        + EARNINGS_FEATURE_COLS
        + GRAPH_FEATURE_COLS
        + OWNERSHIP_FEATURE_COLS
        + ENERGY_FEATURE_COLS
        + SUPPLY_CHAIN_FEATURE_COLS
        + FX_FEATURE_COLS
    ),
}
```

- [ ] **Step 4: Update _impute to accept feature_cols param**

Replace `_impute` in `models/train.py` (lines 255–262):

```python
def _impute(
    X: np.ndarray,
    medians: dict[str, float],
    feature_cols: list[str] = FEATURE_COLS,
) -> np.ndarray:
    """Fill NaN values column-wise using pre-computed per-feature medians."""
    X = X.copy()
    for i, name in enumerate(feature_cols):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = medians.get(name, 0.0)
    return X
```

- [ ] **Step 5: Update _compute_medians to accept feature_cols param**

Replace `_compute_medians` in `models/train.py` (lines 265–272):

```python
def _compute_medians(
    X: np.ndarray,
    feature_cols: list[str] = FEATURE_COLS,
) -> dict[str, float]:
    """Compute per-feature nanmedian over the training set.
    Falls back to 0.0 for columns that are entirely NaN."""
    result = {}
    for i, name in enumerate(feature_cols):
        v = np.nanmedian(X[:, i])
        result[name] = 0.0 if np.isnan(v) else float(v)
    return result
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_train.py::test_tier_feature_cols_medium_equals_feature_cols \
       tests/test_train.py::test_horizon_configs_has_all_eight_horizons \
       tests/test_train.py::test_horizon_configs_tiers_are_valid \
       tests/test_train.py::test_impute_uses_feature_cols_param \
       -v
```

Expected: 4 passed

- [ ] **Step 7: Run full suite**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all passing

- [ ] **Step 8: Commit**

```bash
git add models/train.py tests/test_train.py
git commit -m "feat: add HORIZON_CONFIGS, TIER_FEATURE_COLS; parameterize _impute/_compute_medians"
```

---

### Task 3: Update build_training_dataset for horizon_tag

**Files:**
- Modify: `models/train.py` (lines 127–250 — `build_training_dataset`)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train.py`:

```python
def test_build_training_dataset_horizon_5d_returns_label_return(tmp_path):
    """With horizon_tag='5d', returns 'label_return' column (not label_return_1y)."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset, HORIZON_CONFIGS, TIER_FEATURE_COLS
    df = build_training_dataset(ohlcv_dir, fund_dir, horizon_tag="5d")

    assert "label_return" in df.columns
    assert "label_return_1y" not in df.columns
    assert "label_return_5d" not in df.columns
    assert df["label_return"].null_count() == 0


def test_build_training_dataset_horizon_5d_uses_short_features(tmp_path):
    """With horizon_tag='5d', returned feature columns match TIER_FEATURE_COLS['short']."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset, TIER_FEATURE_COLS
    df = build_training_dataset(ohlcv_dir, fund_dir, horizon_tag="5d")

    short_cols = TIER_FEATURE_COLS["short"]
    for col in short_cols:
        assert col in df.columns, f"Expected short-tier column {col!r} missing"
    # Verify long-tier-only columns are absent (e.g. graph features)
    assert "graph_partner_momentum_30d" not in df.columns


def test_build_training_dataset_no_horizon_tag_unchanged(tmp_path):
    """Without horizon_tag, behavior is unchanged: returns label_return_1y."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset, FEATURE_COLS
    df = build_training_dataset(ohlcv_dir, fund_dir)

    assert "label_return_1y" in df.columns
    for col in FEATURE_COLS:
        assert col in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py::test_build_training_dataset_horizon_5d_returns_label_return \
       tests/test_train.py::test_build_training_dataset_horizon_5d_uses_short_features \
       tests/test_train.py::test_build_training_dataset_no_horizon_tag_unchanged \
       -v
```

Expected: FAIL — `build_training_dataset` does not yet accept `horizon_tag`

- [ ] **Step 3: Update build_training_dataset**

In `models/train.py`, add the import at the top of the file alongside the existing `build_labels` import:

```python
from processing.label_builder import build_labels, build_multi_horizon_labels
```

Then change the `build_training_dataset` signature (line 128) from:

```python
def build_training_dataset(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    layer: str | None = None,
) -> pl.DataFrame:
```

to:

```python
def build_training_dataset(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    layer: str | None = None,
    horizon_tag: str | None = None,
) -> pl.DataFrame:
    """
    Assemble the full labeled training dataset, optionally filtered to one layer
    and one prediction horizon.

    layer:       if given, filters to tickers in that layer only.
    horizon_tag: if given, uses multi-horizon labels and returns tier-appropriate
                 feature columns plus a 'label_return' column. The column
                 'label_return_{horizon_tag}' is renamed to 'label_return', and rows
                 where that column is null are dropped.
                 If None, returns the full FEATURE_COLS plus 'label_return_1y'
                 (backward-compatible behavior).
    """
```

Then replace the final `return` statement at the bottom of the function (line 248):

```python
    # Current final select:
    return (
        df.select(["ticker", "date"] + FEATURE_COLS + ["label_return_1y"])
        .sort(["date", "ticker"])
    )
```

with:

```python
    if horizon_tag is not None:
        label_col = f"label_return_{horizon_tag}"
        tier = HORIZON_CONFIGS[horizon_tag]["tier"]
        feat_cols = TIER_FEATURE_COLS[tier]

        # Join multi-horizon labels (wide: one column per horizon)
        multi_labels = build_multi_horizon_labels(ohlcv_dir)
        if layer is not None:
            multi_labels = multi_labels.filter(pl.col("ticker").is_in(layer_tickers))

        df = df.join(
            multi_labels.select(["ticker", "date", label_col]),
            on=["ticker", "date"],
            how="inner",
        ).filter(pl.col(label_col).is_not_null()).rename({label_col: "label_return"})

        return (
            df.select(["ticker", "date"] + feat_cols + ["label_return"])
            .sort(["date", "ticker"])
        )

    return (
        df.select(["ticker", "date"] + FEATURE_COLS + ["label_return_1y"])
        .sort(["date", "ticker"])
    )
```

Note: `layer_tickers` is already defined earlier in the function body when `layer is not None`. When `layer is None`, the filter on `multi_labels` is skipped (the `if layer is not None` guard is present).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train.py::test_build_training_dataset_horizon_5d_returns_label_return \
       tests/test_train.py::test_build_training_dataset_horizon_5d_uses_short_features \
       tests/test_train.py::test_build_training_dataset_no_horizon_tag_unchanged \
       -v
```

Expected: 3 passed

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add models/train.py tests/test_train.py
git commit -m "feat: build_training_dataset accepts horizon_tag for tier-aware feature selection"
```

---

### Task 4: Update train_single_layer for feature_cols and label_col params

**Files:**
- Modify: `models/train.py` (lines 277–345 — `train_single_layer`)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train.py`:

```python
def test_train_single_layer_saves_tier_feature_names(tmp_path):
    """When feature_cols is the short tier, feature_names.json contains only those cols."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import (
        build_training_dataset, train_single_layer,
        TIER_FEATURE_COLS,
    )
    short_cols = TIER_FEATURE_COLS["short"]
    df = build_training_dataset(ohlcv_dir, fund_dir, horizon_tag="5d")
    train_single_layer(
        df, artifacts_dir,
        feature_cols=short_cols,
        label_col="label_return",
        lgbm_params=_LGBM_TEST,
        rf_params=_RF_TEST,
    )

    import json
    saved = json.loads((artifacts_dir / "feature_names.json").read_text())
    assert saved == short_cols
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_train.py::test_train_single_layer_saves_tier_feature_names -v
```

Expected: FAIL — `train_single_layer` does not accept `feature_cols`, `label_col`, `lgbm_params`, or `rf_params` params

- [ ] **Step 3: Update train_single_layer signature and body**

Replace the `train_single_layer` function signature (line 277) from:

```python
def train_single_layer(df: pl.DataFrame, artifacts_dir: Path) -> None:
```

with:

```python
def train_single_layer(
    df: pl.DataFrame,
    artifacts_dir: Path,
    feature_cols: list[str] = FEATURE_COLS,
    label_col: str = "label_return_1y",
    lgbm_params: dict | None = None,
    rf_params: dict | None = None,
) -> None:
    """Fit the ensemble on df and save all artifacts to artifacts_dir.

    df must have columns: ticker, date, feature_cols..., label_col.
    feature_cols: feature list to train on (default: all 48 FEATURE_COLS).
    label_col: target column name (default: 'label_return_1y').
    """
```

Then inside the function body, replace all hardcoded `FEATURE_COLS` references with `feature_cols`, and `"label_return_1y"` references with `label_col`:

Line 287: `X = df.select(FEATURE_COLS)` → `X = df.select(feature_cols)`

Line 290: `medians = _compute_medians(X)` → `medians = _compute_medians(X, feature_cols)`

Line 291: `X_imp = _impute(X, medians)` → `X_imp = _impute(X, medians, feature_cols)`

Line 293: `X_df = pd.DataFrame(X_imp, columns=FEATURE_COLS)` → `X_df = pd.DataFrame(X_imp, columns=feature_cols)`

Line 288: `y = df["label_return_1y"]` → `y = df[label_col]`

Add lgbm/rf param overrides (after the existing hardcoded model definitions, replace them):

```python
    lgbm_base = lgbm_params or {
        "objective": "quantile", "n_estimators": 400, "learning_rate": 0.03,
        "num_leaves": 31, "min_child_samples": 20, "random_state": 42, "verbose": -1,
    }
    rf_base = rf_params or {
        "n_estimators": 300, "max_depth": 6, "random_state": 42, "n_jobs": -1,
    }

    lgbm_q10 = lgb.LGBMRegressor(**{**lgbm_base, "alpha": 0.1})
    lgbm_q50 = lgb.LGBMRegressor(**{**lgbm_base, "alpha": 0.5})
    lgbm_q90 = lgb.LGBMRegressor(**{**lgbm_base, "alpha": 0.9})
    rf = RandomForestRegressor(**rf_base)
    ridge = Ridge(alpha=1.0)
```

Line 338: `(artifacts_dir / "feature_names.json").write_text(json.dumps(FEATURE_COLS))` → `(artifacts_dir / "feature_names.json").write_text(json.dumps(feature_cols))`

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_train.py::test_train_single_layer_saves_tier_feature_names -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add models/train.py tests/test_train.py
git commit -m "feat: train_single_layer accepts feature_cols, label_col, lgbm_params, rf_params"
```

---

### Task 5: Update train_all_layers to loop over horizons + __main__

**Files:**
- Modify: `models/train.py` (lines 348–365 — `train_all_layers`; lines 618–632 — `__main__`)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train.py`:

```python
def test_train_all_layers_creates_horizon_artifact_dirs(tmp_path):
    """train_all_layers creates horizon_5d/ subdir under each trained layer dir."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train_all_layers
    train_all_layers(
        ohlcv_dir, fund_dir, artifacts_dir,
        horizon_tag="5d",
        lgbm_params=_LGBM_TEST, rf_params=_RF_TEST,
    )

    # At least one layer dir should have been trained
    layer_dirs = list(artifacts_dir.glob("layer_*"))
    assert len(layer_dirs) > 0, "No layer dirs created"
    for layer_dir in layer_dirs:
        horizon_dir = layer_dir / "horizon_5d"
        assert horizon_dir.exists(), f"horizon_5d/ missing under {layer_dir}"
        assert (horizon_dir / "feature_names.json").exists()
        assert (horizon_dir / "lgbm_q50.pkl").exists()


def test_train_all_layers_skips_horizon_with_insufficient_labeled_rows(tmp_path):
    """With only 300 rows, 756d horizon is skipped (0 labeled rows < 100 threshold)."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    # 300 rows < 756 shift → 0 labeled rows for 756d
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, 300)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train_all_layers
    train_all_layers(
        ohlcv_dir, fund_dir, artifacts_dir,
        horizon_tag="756d",
        lgbm_params=_LGBM_TEST, rf_params=_RF_TEST,
    )

    # No horizon_756d dirs should have been created
    horizon_dirs = list(artifacts_dir.glob("layer_*/horizon_756d"))
    assert len(horizon_dirs) == 0, "756d horizon should be skipped due to insufficient data"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py::test_train_all_layers_creates_horizon_artifact_dirs \
       tests/test_train.py::test_train_all_layers_skips_horizon_with_insufficient_labeled_rows \
       -v
```

Expected: FAIL — `train_all_layers` does not accept `horizon_tag`, `lgbm_params`, `rf_params`

- [ ] **Step 3: Replace train_all_layers**

Replace the `train_all_layers` function (lines 348–365) in `models/train.py`:

```python
def train_all_layers(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    artifacts_dir: Path,
    horizon_tag: str | None = None,
    lgbm_params: dict | None = None,
    rf_params: dict | None = None,
) -> None:
    """Train one ensemble per (horizon, layer) pair and save artifacts.

    horizon_tag: if given, trains only that horizon. If None, trains all 8 horizons.
    Horizons are skipped when a layer has fewer than 100 labeled rows.
    Artifacts are saved to: artifacts_dir/layer_{id}_{name}/horizon_{tag}/
    """
    horizons_to_train = [horizon_tag] if horizon_tag else list(HORIZON_CONFIGS.keys())

    for h_tag in horizons_to_train:
        tier = HORIZON_CONFIGS[h_tag]["tier"]
        feat_cols = TIER_FEATURE_COLS[tier]
        _LOG.info("Training horizon %s (%s tier, %d features)", h_tag, tier, len(feat_cols))

        for layer in all_layers():
            layer_id = LAYER_IDS[layer]
            horizon_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}" / f"horizon_{h_tag}"

            df = build_training_dataset(
                ohlcv_dir, fundamentals_dir, layer=layer, horizon_tag=h_tag
            )
            if df.is_empty():
                _LOG.warning("No data for layer %s horizon %s — skipping", layer, h_tag)
                continue

            n_labeled = len(df)
            if n_labeled < 100:
                _LOG.warning(
                    "Horizon %s layer %s: only %d labeled rows, need ≥100 — skipping",
                    h_tag, layer, n_labeled,
                )
                continue

            _LOG.info("  layer %-20s  %d rows → %s", layer, n_labeled, horizon_dir)
            train_single_layer(
                df, horizon_dir,
                feature_cols=feat_cols,
                label_col="label_return",
                lgbm_params=lgbm_params,
                rf_params=rf_params,
            )
```

- [ ] **Step 4: Update __main__ to accept --horizon flag**

Replace the `if __name__ == "__main__":` block (lines 618–632) in `models/train.py`:

```python
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train per-layer horizon models.")
    parser.add_argument(
        "--horizon", default=None,
        help="Single horizon tag to train, e.g. '5d' or '252d'. Default: all 8 horizons.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    ohlcv_dir        = project_root / "data" / "raw" / "financials" / "ohlcv"
    fundamentals_dir = project_root / "data" / "raw" / "financials" / "fundamentals"
    artifacts_dir    = project_root / "models" / "artifacts"

    label = args.horizon or "all"
    _LOG.info("Training per-layer ensembles for horizon(s): %s", label)
    train_all_layers(ohlcv_dir, fundamentals_dir, artifacts_dir, horizon_tag=args.horizon)
    _LOG.info("[Train] All layer artifacts → %s", artifacts_dir)
    print(f"[Train] Artifacts → {artifacts_dir}")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_train.py::test_train_all_layers_creates_horizon_artifact_dirs \
       tests/test_train.py::test_train_all_layers_skips_horizon_with_insufficient_labeled_rows \
       -v
```

Expected: 2 passed

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all passing

- [ ] **Step 7: Commit**

```bash
git add models/train.py tests/test_train.py
git commit -m "feat: train_all_layers loops over (horizon, layer) pairs with coverage gate"
```

---

### Task 6: Update inference for multi-horizon output

**Files:**
- Modify: `models/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_inference.py` (after the module-level `N_DAYS = 500` line, add these constants, then add the fixture after the existing `trained_env` fixture):

```python
# Test-speed hyperparameters (after N_DAYS = 500)
_LGBM_TEST = {"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 8,
              "min_child_samples": 5, "n_jobs": 1, "random_state": 42}
_RF_TEST    = {"n_estimators": 10, "max_features": "sqrt",
               "min_samples_leaf": 2, "n_jobs": 1, "random_state": 42}


@pytest.fixture(scope="module")
def trained_env_multi(tmp_path_factory):
    """Train horizon_252d artifacts; return (base, date_str)."""
    base = tmp_path_factory.mktemp("inference_multi")
    ohlcv_dir = base / "financials" / "ohlcv"
    fund_dir = base / "financials" / "fundamentals"
    artifacts_dir = base / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train_all_layers
    train_all_layers(
        ohlcv_dir, fund_dir, artifacts_dir,
        horizon_tag="252d",
        lgbm_params=_LGBM_TEST, rf_params=_RF_TEST,
    )

    # Use the last date with a full 252-day forward window
    import datetime
    start = datetime.date(2020, 1, 1)
    date_str = (start + datetime.timedelta(days=N_DAYS - 252 - 1)).isoformat()
    return base, date_str


def test_run_inference_writes_horizon_parquet(trained_env_multi):
    """run_inference with horizon_tag='252d' writes horizon=252d/predictions.parquet."""
    base, date_str = trained_env_multi
    output_dir = base / "predictions"

    from models.inference import run_inference
    run_inference(
        date_str=date_str,
        data_dir=base,
        artifacts_dir=base / "artifacts",
        output_dir=output_dir,
        horizon_tag="252d",
    )

    expected = output_dir / f"date={date_str}" / "horizon=252d" / "predictions.parquet"
    assert expected.exists(), f"Expected output at {expected}"


def test_run_inference_backward_compat_path(trained_env_multi):
    """The 252d horizon is also written to the flat date={d}/predictions.parquet path."""
    base, date_str = trained_env_multi
    output_dir = base / "predictions_compat"

    from models.inference import run_inference
    run_inference(
        date_str=date_str,
        data_dir=base,
        artifacts_dir=base / "artifacts",
        output_dir=output_dir,
        horizon_tag="252d",
    )

    compat_path = output_dir / f"date={date_str}" / "predictions.parquet"
    assert compat_path.exists(), f"Backward-compat path missing: {compat_path}"


def test_run_inference_horizon_column_present(trained_env_multi):
    """Returned DataFrame contains a 'horizon' column with the correct tag value."""
    base, date_str = trained_env_multi

    from models.inference import run_inference
    result = run_inference(
        date_str=date_str,
        data_dir=base,
        artifacts_dir=base / "artifacts",
        output_dir=base / "predictions_col",
        horizon_tag="252d",
    )

    assert "horizon" in result.columns
    assert result["horizon"].unique().to_list() == ["252d"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inference.py::test_run_inference_writes_horizon_parquet \
       tests/test_inference.py::test_run_inference_backward_compat_path \
       tests/test_inference.py::test_run_inference_horizon_column_present \
       -v
```

Expected: FAIL — `run_inference` does not accept `horizon_tag`, output path is flat

- [ ] **Step 3: Update _impute in inference.py**

Replace `_impute` in `models/inference.py` (lines 51–57):

```python
def _impute(
    X: np.ndarray,
    medians: dict[str, float],
    feature_cols: list[str] = FEATURE_COLS,
) -> np.ndarray:
    X = X.copy()
    for i, name in enumerate(feature_cols):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = medians.get(name, 0.0)
    return X
```

- [ ] **Step 4: Update _predict_layer to accept horizon_tag**

Replace `_predict_layer` in `models/inference.py` (lines 131–196):

```python
def _predict_layer(
    feature_df: pl.DataFrame,
    layer: str,
    artifacts_dir: Path,
    horizon_tag: str = "252d",
) -> pl.DataFrame | None:
    """Run one layer model on the tickers belonging to that layer for a given horizon.

    Looks for artifacts at layer_dir/horizon_{horizon_tag}/. Falls back to layer_dir/
    directly for legacy flat-structure 252d artifacts (backward compat).

    Returns DataFrame with [ticker, layer, horizon, expected_annual_return,
    confidence_low, confidence_high, lgbm_return, rf_return, ridge_return]
    or None if artifacts missing.
    """
    layer_id = LAYER_IDS[layer]
    layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
    horizon_dir = layer_dir / f"horizon_{horizon_tag}"

    # Backward compat: fall back to flat structure for legacy 252d artifacts
    if not horizon_dir.exists() and horizon_tag == "252d":
        horizon_dir = layer_dir

    if not (horizon_dir / "feature_names.json").exists():
        return None

    feature_names_saved: list[str] = json.loads(
        (horizon_dir / "feature_names.json").read_text()
    )

    layer_tickers = tickers_in_layer(layer)
    layer_df = feature_df.filter(pl.col("ticker").is_in(layer_tickers))
    if layer_df.is_empty():
        return None

    tickers = layer_df["ticker"].to_list()
    medians = json.loads((horizon_dir / "imputation_medians.json").read_text())
    weights = json.loads((horizon_dir / "ensemble_weights.json").read_text())

    X_raw = layer_df.select(feature_names_saved).to_numpy().astype(float)
    X_imp = _impute(X_raw, medians, feature_names_saved)
    scaler = _load_pickle(horizon_dir / "feature_scaler.pkl")
    X_sc = scaler.transform(X_imp)
    X_df = pd.DataFrame(X_imp, columns=feature_names_saved)

    lgbm_q10 = _load_pickle(horizon_dir / "lgbm_q10.pkl")
    lgbm_q50 = _load_pickle(horizon_dir / "lgbm_q50.pkl")
    lgbm_q90 = _load_pickle(horizon_dir / "lgbm_q90.pkl")
    rf_model  = _load_pickle(horizon_dir / "rf_model.pkl")
    ridge_model = _load_pickle(horizon_dir / "ridge_model.pkl")

    q10_preds = lgbm_q10.predict(X_df)
    q50_preds = lgbm_q50.predict(X_df)
    q90_preds = lgbm_q90.predict(X_df)
    rf_preds    = rf_model.predict(X_imp)
    ridge_preds = ridge_model.predict(X_sc)

    expected = (
        weights["lgbm"] * q50_preds
        + weights["rf"]  * rf_preds
        + weights["ridge"] * ridge_preds
    )

    return pl.DataFrame({
        "ticker": tickers,
        "layer": [layer] * len(tickers),
        "horizon": [horizon_tag] * len(tickers),
        "expected_annual_return": expected.tolist(),
        "confidence_low":  q10_preds.tolist(),
        "confidence_high": q90_preds.tolist(),
        "lgbm_return":  q50_preds.tolist(),
        "rf_return":    rf_preds.tolist(),
        "ridge_return": ridge_preds.tolist(),
    })
```

- [ ] **Step 5: Update run_inference to accept horizon_tag and loop**

Replace the `run_inference` function signature and body in `models/inference.py` (lines 199–251):

```python
def run_inference(
    date_str: str,
    data_dir: Path = Path("data/raw"),
    artifacts_dir: Path = Path("models/artifacts"),
    output_dir: Path = Path("data/predictions"),
    horizon_tag: str | None = None,
) -> pl.DataFrame:
    """Run all trained layer models and return globally ranked predictions.

    horizon_tag: if given, runs only that horizon. If None, runs all horizons
                 that have at least one trained layer artifact.

    Returns the primary horizon's (252d if available, else first found) combined
    DataFrame for backward compatibility.

    Raises ValueError if date_str is a weekend.
    Raises RuntimeError if no price data exists for date_str.
    """
    as_of = dt.date.fromisoformat(date_str)
    if as_of.weekday() >= 5:
        raise ValueError(f"{date_str} is a weekend. Skip inference on non-trading days.")

    from models.train import HORIZON_CONFIGS
    horizons_to_run = [horizon_tag] if horizon_tag else list(HORIZON_CONFIGS.keys())

    print(f"[Inference] Running for {date_str}, horizon(s): {horizon_tag or 'all'}...")

    feature_df = _build_feature_df(date_str, data_dir)

    primary_combined: pl.DataFrame | None = None

    for h_tag in horizons_to_run:
        all_preds: list[pl.DataFrame] = []
        for layer in all_layers():
            layer_preds = _predict_layer(feature_df, layer, artifacts_dir, h_tag)
            if layer_preds is not None:
                all_preds.append(layer_preds)

        if not all_preds:
            _LOG.debug("No artifacts for horizon %s — skipping", h_tag)
            continue

        combined = pl.concat(all_preds).sort("expected_annual_return", descending=True)
        combined = combined.with_columns(
            pl.Series("rank", list(range(1, len(combined) + 1)), dtype=pl.Int32),
            pl.lit(as_of).cast(pl.Date).alias("as_of_date"),
        )

        # Write horizon-partitioned output
        out_path = output_dir / f"date={date_str}" / f"horizon={h_tag}" / "predictions.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(out_path, compression="snappy")

        # Backward-compat alias for 252d
        if h_tag == "252d":
            compat_path = output_dir / f"date={date_str}" / "predictions.parquet"
            combined.write_parquet(compat_path, compression="snappy")

        n_tickers = len(combined)
        n_layers = combined["layer"].n_unique()
        print(f"[Inference] [{h_tag}] {n_tickers} tickers × {n_layers} layers → {out_path}")

        # Track primary result (252d preferred, else first)
        if primary_combined is None or h_tag == "252d":
            primary_combined = combined

    if primary_combined is None:
        raise RuntimeError(
            f"No layer artifacts found in {artifacts_dir} for any requested horizon. "
            "Run models/train.py first."
        )

    # Enrich primary predictions with portfolio metrics
    try:
        from processing.portfolio_metrics import enrich
        enrich(date_str, predictions_dir=output_dir)
    except Exception as exc:
        _LOG.warning("Portfolio metrics enrichment failed (non-fatal): %s", exc, exc_info=True)

    return primary_combined
```

- [ ] **Step 6: Update __main__ in inference.py**

Replace the `if __name__ == "__main__":` block (lines 254–260):

```python
if __name__ == "__main__":
    import argparse
    today = dt.date.today()
    if today.weekday() >= 5:
        print(f"[Inference] {today} is a weekend — skipping.")
        import sys; sys.exit(0)

    parser = argparse.ArgumentParser(description="Run multi-horizon inference.")
    parser.add_argument(
        "--horizon", default=None,
        help="Single horizon tag, e.g. '5d' or '252d'. Default: all trained horizons.",
    )
    args = parser.parse_args()
    run_inference(date_str=today.isoformat(), horizon_tag=args.horizon)
```

- [ ] **Step 7: Run the new inference tests**

```bash
pytest tests/test_inference.py::test_run_inference_writes_horizon_parquet \
       tests/test_inference.py::test_run_inference_backward_compat_path \
       tests/test_inference.py::test_run_inference_horizon_column_present \
       -v
```

Expected: 3 passed

- [ ] **Step 8: Run full test suite**

```bash
pytest tests/ -m 'not integration' -q
```

Expected: all previously passing tests still pass, plus 3 new inference tests

- [ ] **Step 9: Commit**

```bash
git add models/inference.py tests/test_inference.py
git commit -m "feat: inference outputs per-horizon parquet with 252d backward-compat alias"
```

---

## Verification

After all 6 tasks are committed, run the full suite one final time:

```bash
pytest tests/ -m 'not integration' -v
```

Expected: all tests pass, zero failures. The key new behaviors to verify in the output:

- `test_build_multi_horizon_labels_*` (4 tests) — label builder
- `test_tier_feature_cols_*`, `test_horizon_configs_*`, `test_impute_*` (4 tests) — constants
- `test_build_training_dataset_horizon_*` (3 tests) — dataset assembly
- `test_train_single_layer_saves_tier_feature_names` (1 test) — artifact content
- `test_train_all_layers_*` (2 tests) — training loop + coverage gate
- `test_run_inference_*` (3 tests) — inference outputs
