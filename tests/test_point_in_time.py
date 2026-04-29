"""Architecture v2 / Phase A.4 — point-in-time invariant regression suite.

These tests are the safety net that catches regressions in our point-in-time
correctness work. They assert three layers of invariant:

  (1) SCHEMA PRESENCE — Every ingestion module on the PIT-corrected list emits
      an `available_date` column. Catches regressions where the column is
      accidentally removed from the output schema.

  (2) DATA-LEVEL INVARIANT — Every row in every feature parquet that has both
      `available_date` and `period_end` satisfies available_date >= period_end.
      Catches regressions in the date-stamping logic at ingest time.

  (3) JOIN-LEVEL INVARIANT — For each `join_*_features` function on the
      PIT-corrected list, building a synthetic spine + fixture and joining
      MUST NOT attach a feature value whose underlying available_date is
      after the spine date. Catches regressions that revert an asof join's
      right key from `available_date` back to `period_end`.

If you add a new feature module that touches financial / macro data with a
publication lag, register it in PIT_CORRECTED_INGESTION_MODULES /
PIT_CORRECTED_JOIN_FUNCTIONS so the regression suite covers it.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest


# ── Module registry — every PIT-corrected source must be listed here ─────────

# (module path, schema-attribute name) — the output PyArrow schema is read off
# the named attribute and must include an `available_date` field.
PIT_CORRECTED_INGESTION_MODULES: list[tuple[str, str]] = [
    ("ingestion.sec_13f_ingestion",            "_RAW_SCHEMA"),
    ("ingestion.edgar_fundamentals_ingestion", "_SCHEMA"),
]


# ── (1) SCHEMA PRESENCE ──────────────────────────────────────────────────────

@pytest.mark.parametrize("module_path,schema_attr", PIT_CORRECTED_INGESTION_MODULES)
def test_pit_module_schema_has_available_date(module_path: str, schema_attr: str):
    """Every PIT-corrected ingestion module must persist an available_date column."""
    import importlib
    module = importlib.import_module(module_path)
    schema: pa.Schema = getattr(module, schema_attr)
    field_names = [f.name for f in schema]
    assert "available_date" in field_names, (
        f"{module_path}.{schema_attr} is missing 'available_date' — point-in-time "
        f"correctness regression. Schema fields: {field_names}"
    )
    available_field = schema.field("available_date")
    assert available_field.type == pa.date32(), (
        f"{module_path}.{schema_attr}.available_date must be date32, got {available_field.type}"
    )


# ── (2) DATA-LEVEL INVARIANT ─────────────────────────────────────────────────

# Real-data parquet directories that should carry both period_end and
# available_date (i.e. quarterly aggregates that hold the invariant
# available_date >= period_end). Skipped when the directory is missing —
# the test is a no-op if the user hasn't run ingestion yet.
PIT_FEATURE_DIRS_QUARTERLY: list[Path] = [
    Path("data/raw/financials/13f_holdings/features"),
    Path("data/raw/financials/fundamentals"),
]


@pytest.mark.parametrize("features_dir", PIT_FEATURE_DIRS_QUARTERLY)
def test_pit_real_data_available_date_after_period_end(features_dir: Path):
    """Every real feature parquet row must satisfy available_date >= period_end."""
    parquets = list(features_dir.glob("*/*.parquet"))
    if not parquets:
        pytest.skip(f"no feature parquets under {features_dir} — ingestion not run")

    bad_rows: list[tuple[Path, int]] = []
    rows_checked = 0
    for path in parquets:
        df = pl.read_parquet(path)
        if "available_date" not in df.columns or "period_end" not in df.columns:
            # Legacy parquet without available_date — skip silently (the join-side
            # fallback handles these). Once they're rebuilt this branch becomes dead.
            continue
        eligible = df.filter(
            pl.col("available_date").is_not_null() & pl.col("period_end").is_not_null()
        )
        violations = eligible.filter(pl.col("available_date") < pl.col("period_end"))
        rows_checked += eligible.height
        if violations.height > 0:
            bad_rows.append((path, violations.height))

    assert not bad_rows, (
        f"{sum(n for _, n in bad_rows)} rows across {len(bad_rows)} parquets violate "
        f"available_date >= period_end. Sample: {bad_rows[:3]}"
    )
    if rows_checked == 0:
        pytest.skip(f"no rows with both columns under {features_dir} — legacy data only")


# ── (3) JOIN-LEVEL INVARIANT ─────────────────────────────────────────────────

def _ymd(y: int, m: int, d: int) -> datetime.date:
    return datetime.date(y, m, d)


def test_pit_ownership_join_no_lookahead(tmp_path):
    """join_ownership_features must not attach a value where available_date > spine_date."""
    from processing.ownership_features import save_ownership_features, join_ownership_features

    # Q4 13F: period 2023-12-31, but the slowest filer reports 2024-02-28 → that's
    # the available_date. Spine 2024-01-15 must NOT see this row.
    features = pl.DataFrame({
        "ticker":                  ["NVDA"],
        "quarter":                 ["2023Q4"],
        "period_end":              [_ymd(2023, 12, 31)],
        "available_date":          [_ymd(2024, 2, 28)],
        "inst_ownership_pct":      [80.0],
        "inst_net_shares_qoq":     [None],
        "inst_holder_count":       [1],
        "inst_concentration_top10":[1.0],
        "inst_momentum_2q":        [None],
    })
    save_ownership_features(features, tmp_path)

    spine = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   [_ymd(2024, 1, 15), _ymd(2024, 3, 15)],  # before / after available_date
    })
    result = join_ownership_features(spine, tmp_path).sort("date")

    pre  = result.filter(pl.col("date") == _ymd(2024, 1, 15))
    post = result.filter(pl.col("date") == _ymd(2024, 3, 15))
    assert pre["inst_ownership_pct"][0] is None, (
        "regression: ownership join attached a value before its available_date"
    )
    assert post["inst_ownership_pct"][0] == pytest.approx(80.0)


def test_pit_fundamentals_join_no_lookahead(tmp_path):
    """join_fundamentals must not attach a value where available_date > spine_date."""
    from processing.fundamental_features import join_fundamentals

    quarter = {
        "ticker": "NVDA",
        "period_end": _ymd(2023, 12, 31),
        "available_date": _ymd(2024, 2, 14),  # 10-Q filed
        "pe_ratio_trailing": 30.0, "price_to_sales": 10.0, "price_to_book": 5.0,
        "revenue_growth_yoy": 0.20, "gross_margin": 0.65, "operating_margin": 0.40,
        "capex_to_revenue": 0.10, "debt_to_equity": 0.30, "current_ratio": 2.0,
        "net_income_margin": 0.20, "free_cash_flow_margin": 0.15,
        "capex_growth_yoy": 0.10, "revenue_growth_accel": 0.02,
        "research_to_revenue": 0.12,
    }
    out = tmp_path / "NVDA" / "quarterly.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([quarter]).write_parquet(str(out))

    price = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   pl.Series([_ymd(2024, 1, 20), _ymd(2024, 2, 28)], dtype=pl.Date),
        "close_price": [500.0, 600.0],
    })
    result = join_fundamentals(price, fundamentals_dir=tmp_path).sort("date")

    pre  = result.filter(pl.col("date") == _ymd(2024, 1, 20))
    post = result.filter(pl.col("date") == _ymd(2024, 2, 28))
    assert pre["gross_margin"][0] is None, (
        "regression: fundamentals join attached a value before its available_date"
    )
    assert post["gross_margin"][0] == pytest.approx(0.65)


def test_pit_physical_ai_join_respects_publication_lag(tmp_path):
    """join_physical_ai_features must not attach a FRED level before publication_lag elapses."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "fred"
    fred_dir.mkdir()
    # NEWORDER: period 2024-01-01 → publishes 2024-02-15 (45-day lag)
    pl.DataFrame({
        "date":  [_ymd(2024, 1, 1)],
        "value": [100.0],
    }).write_parquet(str(fred_dir / "NEWORDER.parquet"))

    jolts_dir = tmp_path / "jolts"  # left empty — no JOLTS data
    patents_dir = tmp_path / "patents"  # left empty

    spine = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   pl.Series([_ymd(2024, 1, 20),  # PRE-publication
                             _ymd(2024, 3, 15)], # POST-publication
                            dtype=pl.Date),
    })
    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir).sort("date")

    pre  = out.filter(pl.col("date") == _ymd(2024, 1, 20))
    post = out.filter(pl.col("date") == _ymd(2024, 3, 15))
    assert pre["phys_ai_capgoods_orders_level"][0] is None, (
        "regression: physical_ai FRED join attached a value before publication_lag elapsed"
    )
    assert post["phys_ai_capgoods_orders_level"][0] == pytest.approx(100.0)


# ── Documentation guardrail ──────────────────────────────────────────────────

def test_pit_planning_seed_exists():
    """The architecture v2 / Phase A planning seed must exist as documentation."""
    seed = Path("docs/superpowers/specs/_planning_seed_point_in_time.md")
    assert seed.exists(), (
        f"missing {seed} — architecture v2 / Phase A documentation should be tracked"
    )
    text = seed.read_text()
    for marker in ["13f", "fundamental", "BLS", "Census", "FRED"]:
        assert marker.lower() in text.lower(), (
            f"planning seed missing reference to {marker}"
        )
