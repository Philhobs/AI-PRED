import datetime
import json

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path


# ── Fixture helpers ──────────────────────────────────────────────────────────

TICKERS_FIXTURE = ["NVDA", "MSFT", "AMZN", "GOOGL", "META"]
N_DAYS = 500  # 500 total days → 500-252=248 labeled days per ticker


def _write_ohlcv_fixture(ohlcv_dir: Path, tickers: list[str], n_days: int) -> None:
    """Write n_days of synthetic random-walk OHLCV for each ticker."""
    start = datetime.date(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(n_days)]

    for ticker in tickers:
        path = ohlcv_dir / ticker / "all.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(hash(ticker) % (2**31))
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days))

        table = pa.table({
            "ticker": [ticker] * n_days,
            "date": pa.array(dates, type=pa.date32()),
            "open": prices.tolist(),
            "high": (prices * 1.01).tolist(),
            "low": (prices * 0.99).tolist(),
            "close_price": prices.tolist(),
            "volume": [1_000_000] * n_days,
        })
        pq.write_table(table, str(path))


def _write_fundamentals_fixture(fund_dir: Path, tickers: list[str]) -> None:
    """Write a single quarterly snapshot per ticker starting before the OHLCV range."""
    for ticker in tickers:
        path = fund_dir / ticker / "quarterly.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([{
            "ticker": ticker,
            "period_end": datetime.date(2019, 12, 31),
            "pe_ratio_trailing": 25.0,
            "price_to_sales": 8.0,
            "price_to_book": 3.0,
            "revenue_growth_yoy": 0.15,
            "gross_margin": 0.60,
            "operating_margin": 0.25,
            "capex_to_revenue": 0.08,
            "debt_to_equity": 0.5,
            "current_ratio": 1.8,
            # 5 new columns
            "net_income_margin": 0.20,
            "free_cash_flow_margin": 0.15,
            "capex_growth_yoy": 0.10,
            "revenue_growth_accel": 0.02,
            "research_to_revenue": 0.12,
        }]).write_parquet(str(path))


# Shared reduced hyperparams for test speed
_LGBM_TEST = {"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 8,
              "min_child_samples": 5, "n_jobs": 1, "random_state": 42}
_RF_TEST    = {"n_estimators": 10, "max_features": "sqrt",
               "min_samples_leaf": 2, "n_jobs": 1, "random_state": 42}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_training_dataset_returns_labeled_rows(tmp_path):
    """build_training_dataset returns (N_DAYS-252)*n_tickers rows with all feature cols."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset, FEATURE_COLS
    df = build_training_dataset(ohlcv_dir, fund_dir)

    assert len(df) == (N_DAYS - 252) * len(TICKERS_FIXTURE)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"
    assert "label_return_1y" in df.columns
    assert df["label_return_1y"].null_count() == 0


def test_train_writes_all_artifacts(tmp_path):
    """train() writes all 9 required artifact files to artifacts_dir."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train
    train(
        ohlcv_dir=ohlcv_dir,
        fundamentals_dir=fund_dir,
        artifacts_dir=artifacts_dir,
        lgbm_params=_LGBM_TEST,
        rf_params=_RF_TEST,
        val_window_days=50,
    )

    expected = [
        "lgbm_q10.pkl", "lgbm_q50.pkl", "lgbm_q90.pkl",
        "rf_model.pkl", "ridge_model.pkl", "feature_scaler.pkl",
        "imputation_medians.json", "feature_names.json", "ensemble_weights.json",
    ]
    for name in expected:
        assert (artifacts_dir / name).exists(), f"Missing artifact: {name}"


def test_ensemble_weights_sum_to_one(tmp_path):
    """ensemble_weights.json values sum to 1.0 within floating-point tolerance."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train
    train(
        ohlcv_dir=ohlcv_dir,
        fundamentals_dir=fund_dir,
        artifacts_dir=artifacts_dir,
        lgbm_params=_LGBM_TEST,
        rf_params=_RF_TEST,
        val_window_days=50,
    )

    weights = json.loads((artifacts_dir / "ensemble_weights.json").read_text())
    total = weights["lgbm"] + weights["rf"] + weights["ridge"]
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"
    assert weights["lgbm"] >= 0
    assert weights["rf"] >= 0
    assert weights["ridge"] >= 0


def test_train_raises_on_insufficient_data(tmp_path):
    """train() raises ValueError when data has fewer unique dates than 3*val_window+50."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    # 300 days → 300-252=48 labeled days; need 252*3+50=806 → should raise
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, 300)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train
    with pytest.raises(ValueError, match="Insufficient unique dates"):
        train(
            ohlcv_dir=ohlcv_dir,
            fundamentals_dir=fund_dir,
            artifacts_dir=artifacts_dir,
            val_window_days=252,  # production default — 300 days is way too few
        )


def test_feature_cols_contains_fx():
    from models.train import FEATURE_COLS, FX_FEATURE_COLS
    assert all(c in FEATURE_COLS for c in FX_FEATURE_COLS)


def test_tier_feature_cols_medium_equals_feature_cols():
    """TIER_FEATURE_COLS['medium'] must be identical to FEATURE_COLS (88 features)."""
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
    assert "label_return" not in df.columns


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


def test_build_training_dataset_invalid_horizon_tag_raises(tmp_path):
    """build_training_dataset raises ValueError for unknown horizon_tag."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset
    with pytest.raises(ValueError, match="Unknown horizon_tag"):
        build_training_dataset(ohlcv_dir, fund_dir, horizon_tag="99y")


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

    layer_dirs = list(artifacts_dir.glob("layer_*"))
    # TICKERS_FIXTURE belongs to "cloud" and/or "compute" layers — at least 1 trains
    assert len(layer_dirs) >= 1, "Expected at least one layer dir to be created"
    for layer_dir in layer_dirs:
        horizon_dir = layer_dir / "horizon_5d"
        assert horizon_dir.exists(), f"horizon_5d/ missing under {layer_dir}"
        assert (horizon_dir / "feature_names.json").exists()
        assert (horizon_dir / "lgbm_q50.pkl").exists()
    # All trained layers must have the horizon subdir — none should use the flat layout
    flat_artifacts = [
        d for d in layer_dirs
        if (d / "feature_names.json").exists()  # flat layout artifact at layer root
    ]
    assert len(flat_artifacts) == 0, "Found flat-layout artifacts; expected horizon_{tag}/ subdirs"


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


def test_build_training_dataset_short_horizon_does_not_lose_recent_rows(tmp_path):
    """5d horizon dataset must NOT be constrained to rows with valid 252d labels.

    With N_DAYS rows and a 5d shift, there should be N_DAYS - 5 labeled rows.
    With N_DAYS rows and a 252d shift (old bug), there would only be N_DAYS - 252 rows.
    """
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    n = 400
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, n)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset
    df = build_training_dataset(ohlcv_dir, fund_dir, horizon_tag="5d")

    # With 400 rows per ticker and shift=5, each ticker contributes 395 labeled rows.
    # If the 252d-constraint bug were present, only 148 rows per ticker would survive.
    rows_per_ticker = df.group_by("ticker").agg(pl.len().alias("n"))["n"].min()
    assert rows_per_ticker >= n - 5 - 10, (
        f"Expected ~{n-5} rows per ticker for 5d horizon, got {rows_per_ticker}. "
        "Likely still constrained by 252d label scope."
    )


def test_feature_cols_includes_cyber_threat():
    """FEATURE_COLS must contain all 7 CYBER_THREAT_FEATURE_COLS after integration."""
    from models.train import FEATURE_COLS
    from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS
    for col in CYBER_THREAT_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_tier_short_includes_cyber_threat_7d_features():
    """Short tier must include the 5 cyber threat _7d features, not the _30d ones."""
    from models.train import TIER_FEATURE_COLS
    short = TIER_FEATURE_COLS["short"]
    short_cyber = ["cve_critical_7d", "cve_high_7d", "cisa_kev_7d", "otx_pulse_7d", "cyber_threat_index_7d"]
    for col in short_cyber:
        assert col in short, f"{col} missing from short tier"
    # 30d features should NOT be in short tier
    assert "cve_critical_30d" not in short
    assert "cisa_kev_30d" not in short


def test_tier_long_excludes_cyber_threat():
    """Long tier must NOT include any cyber threat features."""
    from models.train import TIER_FEATURE_COLS
    from processing.cyber_threat_features import CYBER_THREAT_FEATURE_COLS
    for col in CYBER_THREAT_FEATURE_COLS:
        assert col not in TIER_FEATURE_COLS["long"], f"{col} should not be in long tier"


# ── Options signals integration ────────────────────────────────────────────────

def test_options_feature_cols_in_feature_cols():
    """All 6 OPTIONS_FEATURE_COLS must appear in FEATURE_COLS."""
    from models.train import FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    for col in OPTIONS_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_options_feature_cols_in_short_tier():
    """All 6 OPTIONS_FEATURE_COLS must be in TIER_FEATURE_COLS['short']."""
    from models.train import TIER_FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    short_cols = TIER_FEATURE_COLS["short"]
    for col in OPTIONS_FEATURE_COLS:
        assert col in short_cols, f"{col} missing from TIER_FEATURE_COLS['short']"


def test_options_feature_cols_in_medium_tier():
    """All 6 OPTIONS_FEATURE_COLS must be in TIER_FEATURE_COLS['medium'] (inherits FEATURE_COLS)."""
    from models.train import TIER_FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    medium_cols = TIER_FEATURE_COLS["medium"]
    for col in OPTIONS_FEATURE_COLS:
        assert col in medium_cols, f"{col} missing from TIER_FEATURE_COLS['medium']"


def test_options_feature_cols_not_in_long_tier():
    """OPTIONS_FEATURE_COLS must NOT appear in TIER_FEATURE_COLS['long'] (noise at year+ horizons)."""
    from models.train import TIER_FEATURE_COLS
    from processing.options_features import OPTIONS_FEATURE_COLS
    long_cols = set(TIER_FEATURE_COLS["long"])
    for col in OPTIONS_FEATURE_COLS:
        assert col not in long_cols, f"{col} must not be in TIER_FEATURE_COLS['long']"


def test_medium_tier_is_copy_of_feature_cols():
    """TIER_FEATURE_COLS['medium'] must be a separate list object from FEATURE_COLS."""
    from models.train import FEATURE_COLS, TIER_FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] is not FEATURE_COLS, (
        "medium tier must be list(FEATURE_COLS), not FEATURE_COLS itself"
    )


def test_feature_cols_includes_gov_behavioral():
    """FEATURE_COLS must contain all 6 GOV_BEHAVIORAL_FEATURE_COLS after integration."""
    from models.train import FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    assert len(GOV_BEHAVIORAL_FEATURE_COLS) == 6
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_gov_behavioral_cols_absent_from_short_tier():
    """GOV cols must not be in short tier — contract award cycles too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_gov_behavioral_cols_in_medium_tier():
    """GOV cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_gov_behavioral_cols_in_long_tier():
    """GOV cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_gov_behavioral_col_names_are_correct():
    """GOV_BEHAVIORAL_FEATURE_COLS must contain exactly the 6 expected column names."""
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    expected = {
        "gov_contract_value_90d",
        "gov_contract_count_90d",
        "gov_contract_momentum",
        "gov_ai_spend_30d",
        "ferc_queue_mw_30d",
        "ferc_grid_constraint_score",
    }
    assert set(GOV_BEHAVIORAL_FEATURE_COLS) == expected


def test_tier_medium_equals_feature_cols_after_gov_integration():
    """TIER_FEATURE_COLS['medium'] must still equal full FEATURE_COLS (now 88)."""
    from models.train import FEATURE_COLS, TIER_FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] == FEATURE_COLS


def test_feature_cols_includes_uspto_patent():
    """FEATURE_COLS must contain all 6 USPTO_PATENT_FEATURE_COLS after integration."""
    from models.train import FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    assert len(USPTO_PATENT_FEATURE_COLS) == 6
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 88


def test_uspto_patent_cols_absent_from_short_tier():
    """USPTO cols must not appear in short tier — patent cycles too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_uspto_patent_cols_in_medium_tier():
    """USPTO cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_uspto_patent_cols_in_long_tier():
    """USPTO cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_uspto_patent_col_names_correct():
    """USPTO_PATENT_FEATURE_COLS must contain exactly the 6 expected column names."""
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    expected = {
        "patent_app_count_90d",
        "patent_app_momentum",
        "patent_grant_count_365d",
        "patent_grant_rate_365d",
        "patent_ai_cpc_share_90d",
        "patent_citation_count_365d",
    }
    assert set(USPTO_PATENT_FEATURE_COLS) == expected


def test_feature_cols_includes_labor():
    """FEATURE_COLS must contain all 4 LABOR_FEATURE_COLS and total must be 88."""
    from models.train import FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    assert len(LABOR_FEATURE_COLS) == 4
    for col in LABOR_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 88, f"Expected 88 features, got {len(FEATURE_COLS)}"


def test_labor_cols_absent_from_short_tier():
    """LABOR cols must not appear in short tier — monthly data too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in LABOR_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_labor_cols_in_medium_tier():
    """LABOR cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in LABOR_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_labor_cols_in_long_tier():
    """LABOR cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in LABOR_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_labor_col_names_correct():
    """LABOR_FEATURE_COLS must contain exactly the 4 expected column names."""
    from processing.labor_features import LABOR_FEATURE_COLS
    expected = {
        "gov_ai_hiring_30d",
        "gov_ai_hiring_momentum",
        "tech_job_openings_index",
        "tech_job_openings_momentum",
    }
    assert set(LABOR_FEATURE_COLS) == expected


def test_feature_cols_includes_census():
    """FEATURE_COLS must contain all 6 CENSUS_TRADE_FEATURE_COLS and total must be 88."""
    from models.train import FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    assert len(CENSUS_TRADE_FEATURE_COLS) == 6
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 88, f"Expected 88 features, got {len(FEATURE_COLS)}"


def test_census_cols_absent_from_short_tier():
    """CENSUS cols must not appear in short tier — monthly data too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_census_cols_in_medium_tier():
    """CENSUS cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_census_cols_in_long_tier():
    """CENSUS cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in CENSUS_TRADE_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_census_col_names_correct():
    """CENSUS_TRADE_FEATURE_COLS must contain exactly the 6 expected column names."""
    from processing.census_trade_features import CENSUS_TRADE_FEATURE_COLS
    expected = {
        "semicon_import_value",
        "semicon_import_momentum",
        "dc_equipment_import_value",
        "dc_equipment_import_momentum",
        "china_semicon_export_share",
        "taiwan_semicon_import_share",
    }
    assert set(CENSUS_TRADE_FEATURE_COLS) == expected


# ── EDGAR expanded fundamentals (Task 2: 9→14 columns) ───────────────────────

def test_feature_cols_includes_edgar_expanded():
    """FEATURE_COLS must contain all 14 FUNDAMENTAL_FEATURE_COLS and total must be 88."""
    from models.train import FEATURE_COLS
    from processing.fundamental_features import FUNDAMENTAL_FEATURE_COLS
    assert len(FUNDAMENTAL_FEATURE_COLS) == 14
    for col in FUNDAMENTAL_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 88, f"Expected 88 features, got {len(FEATURE_COLS)}"


def test_edgar_expanded_cols_absent_from_short_tier():
    """New fundamental cols must not appear in short tier — quarterly cadence too slow."""
    from models.train import TIER_FEATURE_COLS
    new_cols = {
        "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
        "revenue_growth_accel", "research_to_revenue",
    }
    short = set(TIER_FEATURE_COLS["short"])
    for col in new_cols:
        assert col not in short, f"{col} must not be in short tier"


def test_edgar_expanded_cols_in_medium_tier():
    """New fundamental cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    new_cols = [
        "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
        "revenue_growth_accel", "research_to_revenue",
    ]
    medium = TIER_FEATURE_COLS["medium"]
    for col in new_cols:
        assert col in medium, f"{col} missing from medium tier"


def test_edgar_expanded_cols_in_long_tier():
    """New fundamental cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    new_cols = [
        "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
        "revenue_growth_accel", "research_to_revenue",
    ]
    long_cols = TIER_FEATURE_COLS["long"]
    for col in new_cols:
        assert col in long_cols, f"{col} missing from long tier"


def test_edgar_expanded_col_names_correct():
    """FUNDAMENTAL_FEATURE_COLS must contain exactly the 14 expected column names."""
    from processing.fundamental_features import FUNDAMENTAL_FEATURE_COLS
    expected = {
        "pe_ratio_trailing", "price_to_sales", "price_to_book",
        "revenue_growth_yoy", "gross_margin", "operating_margin",
        "capex_to_revenue", "debt_to_equity", "current_ratio",
        "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
        "revenue_growth_accel", "research_to_revenue",
    }
    assert set(FUNDAMENTAL_FEATURE_COLS) == expected
