# EDGAR Fundamentals Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `edgar_fundamentals_ingestion.py` into `run_refresh.sh` and expand the fundamental feature set from 9 → 14 columns (FEATURE_COLS 83 → 88) by adding 5 EDGAR-derived TTM metrics.

**Architecture:** Add R&D XBRL fetch to `_build_income_df`; add 5 new TTM-based metrics to `_compute_derived`; expand `_SCHEMA` 11→16; rename `_FUND_COLS` → `FUNDAMENTAL_FEATURE_COLS` (public) in `fundamental_features.py`; expand `FUND_FEATURE_COLS` in `train.py`; insert EDGAR as step 2/16 in `run_refresh.sh`.

**Tech Stack:** Polars, PyArrow, SEC EDGAR XBRL API, pytest

---

### Task 1: EDGAR Ingestion Expansion — R&D fetch, 5 new TTM metrics, _SCHEMA 11→16

**Files:**
- Modify: `ingestion/edgar_fundamentals_ingestion.py`
- Modify: `tests/test_edgar_fundamentals.py` (add `rd_expense` to existing fixtures)
- Create: `tests/test_edgar_fundamentals_ingestion.py` (5 new tests)

- [ ] **Step 1: Write 5 failing tests in a new file**

Create `tests/test_edgar_fundamentals_ingestion.py`:

```python
"""Tests for the 5 new EDGAR-derived TTM metrics in edgar_fundamentals_ingestion.py."""
import datetime
import pyarrow as pa
import polars as pl
import pytest

from ingestion.edgar_fundamentals_ingestion import _SCHEMA, _compute_derived


# ── 8-quarter fixtures (enough history for capex_growth_yoy) ──────────────────

_INCOME_8Q = pl.DataFrame({
    "period_end": pl.Series([
        datetime.date(2021, 3, 31), datetime.date(2021, 6, 30),
        datetime.date(2021, 9, 30), datetime.date(2021, 12, 31),
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
    ], dtype=pl.Date),
    "revenue":          [10_000.0] * 8,
    "gross_profit":     [ 6_000.0] * 8,
    "operating_income": [ 3_000.0] * 8,
    "net_income":       [ 2_000.0] * 8,
    "capex":            [500.0, 500.0, 500.0, 500.0,   # prior TTM = 2000
                         600.0, 600.0, 600.0, 600.0],  # current TTM = 2400
    "rd_expense":       [1_000.0] * 8,                 # TTM R&D = 4000
})

_BALANCE_8Q = pl.DataFrame({
    "period_end": pl.Series([
        datetime.date(2021, 3, 31), datetime.date(2021, 6, 30),
        datetime.date(2021, 9, 30), datetime.date(2021, 12, 31),
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
    ], dtype=pl.Date),
    "equity":              [50_000.0] * 8,
    "long_term_debt":      [10_000.0] * 8,
    "current_assets":      [20_000.0] * 8,
    "current_liabilities": [ 8_000.0] * 8,
    "shares_outstanding":  [ 1_000.0] * 8,
})


def test_schema_has_16_columns():
    """_SCHEMA must have 16 fields: ticker + period_end + 14 ratio columns."""
    assert len(_SCHEMA) == 16
    assert _SCHEMA.field("net_income_margin").type == pa.float64()
    assert _SCHEMA.field("free_cash_flow_margin").type == pa.float64()
    assert _SCHEMA.field("capex_growth_yoy").type == pa.float64()
    assert _SCHEMA.field("revenue_growth_accel").type == pa.float64()
    assert _SCHEMA.field("research_to_revenue").type == pa.float64()


def test_net_income_margin():
    """net_income_margin = TTM net income / TTM revenue."""
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    # TTM net_income = 2000*4 = 8000; TTM revenue = 10000*4 = 40000 → 0.20
    assert abs(row["net_income_margin"] - 0.20) < 1e-6


def test_free_cash_flow_margin():
    """free_cash_flow_margin = (TTM op income - TTM capex) / TTM revenue."""
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    # TTM op income = 3000*4=12000; TTM capex = 600*4=2400; TTM rev = 40000
    # fcf_margin = (12000 - 2400) / 40000 = 0.24
    assert abs(row["free_cash_flow_margin"] - 0.24) < 1e-6


def test_capex_growth_yoy():
    """capex_growth_yoy = (TTM capex[t] / TTM capex[t-4q]) - 1; null when <8q."""
    df = _compute_derived(_INCOME_8Q, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    # current TTM capex = 2400; prior TTM capex = 2000 → 0.20
    assert abs(row["capex_growth_yoy"] - 0.20) < 1e-6

    # With only 4 quarters of history, capex_growth_yoy must be null (no prior TTM)
    df_4q = _compute_derived(_INCOME_8Q.tail(4), _BALANCE_8Q.tail(4))
    assert df_4q["capex_growth_yoy"].is_null().all()


def test_research_to_revenue_zero_when_no_rd():
    """research_to_revenue = 0.0 when rd_expense is 0 (R&D concept unavailable)."""
    income_no_rd = _INCOME_8Q.with_columns(pl.lit(0.0).alias("rd_expense"))
    df = _compute_derived(income_no_rd, _BALANCE_8Q)
    q = df.filter(pl.col("period_end") == datetime.date(2022, 12, 31))
    row = q.row(0, named=True)
    assert row["research_to_revenue"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/phila/Documents/AI\ Projects/AI\ Market\ Prediction/AI-PRED
pytest tests/test_edgar_fundamentals_ingestion.py -v 2>&1 | head -40
```

Expected: ImportError or AttributeError (columns don't exist yet).

- [ ] **Step 3: Update `_INCOME_FIXTURE` in `tests/test_edgar_fundamentals.py`**

Add `rd_expense` to `_INCOME_FIXTURE` (line 112–123 in the existing file). Replace the existing `_INCOME_FIXTURE` definition:

```python
_INCOME_FIXTURE = pl.DataFrame({
    "period_end":       pl.Series([
        datetime.date(2022, 3, 31), datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30), datetime.date(2022, 12, 31),
        datetime.date(2023, 3, 31),
    ], dtype=pl.Date),
    "revenue":          [10_000.0, 10_500.0, 11_000.0, 11_500.0, 12_000.0],
    "gross_profit":     [ 6_000.0,  6_300.0,  6_600.0,  6_900.0,  7_200.0],
    "operating_income": [ 3_000.0,  3_150.0,  3_300.0,  3_450.0,  3_600.0],
    "net_income":       [ 2_500.0,  2_625.0,  2_750.0,  2_875.0,  3_000.0],
    "capex":            [   500.0,    525.0,    550.0,    575.0,    600.0],
    "rd_expense":       [ 1_000.0,  1_050.0,  1_100.0,  1_150.0,  1_200.0],
})
```

Also add `"rd_expense": [1_000.0, 1_050.0, 1_150.0, 1_200.0]` to `income_with_gap` in `test_compute_derived_yoy_handles_missing_quarter` (the 4-row fixture without Q3 2022):

```python
    income_with_gap = pl.DataFrame({
        "period_end": pl.Series([
            datetime.date(2022, 3, 31),   # Q1 2022
            datetime.date(2022, 6, 30),   # Q2 2022
            # Q3 2022 MISSING
            datetime.date(2022, 12, 31),  # Q4 2022
            datetime.date(2023, 3, 31),   # Q1 2023
        ], dtype=pl.Date),
        "revenue":          [10_000.0, 10_500.0, 11_500.0, 12_000.0],
        "gross_profit":     [ 6_000.0,  6_300.0,  6_900.0,  7_200.0],
        "operating_income": [ 3_000.0,  3_150.0,  3_450.0,  3_600.0],
        "net_income":       [ 2_500.0,  2_625.0,  2_875.0,  3_000.0],
        "capex":            [   500.0,    525.0,    575.0,    600.0],
        "rd_expense":       [ 1_000.0,  1_050.0,  1_150.0,  1_200.0],
    })
```

- [ ] **Step 4: Expand `_SCHEMA` in `ingestion/edgar_fundamentals_ingestion.py`**

Replace the existing `_SCHEMA` definition (lines 149–161):

```python
_SCHEMA = pa.schema([
    pa.field("ticker",                pa.string()),
    pa.field("period_end",            pa.date32()),
    pa.field("pe_ratio_trailing",     pa.float64()),
    pa.field("price_to_sales",        pa.float64()),
    pa.field("price_to_book",         pa.float64()),
    pa.field("revenue_growth_yoy",    pa.float64()),
    pa.field("gross_margin",          pa.float64()),
    pa.field("operating_margin",      pa.float64()),
    pa.field("capex_to_revenue",      pa.float64()),
    pa.field("debt_to_equity",        pa.float64()),
    pa.field("current_ratio",         pa.float64()),
    # 5 new TTM-based metrics
    pa.field("net_income_margin",     pa.float64()),
    pa.field("free_cash_flow_margin", pa.float64()),
    pa.field("capex_growth_yoy",      pa.float64()),
    pa.field("revenue_growth_accel",  pa.float64()),
    pa.field("research_to_revenue",   pa.float64()),
])
```

- [ ] **Step 5: Add R&D fetch to `_build_income_df`**

In `ingestion/edgar_fundamentals_ingestion.py`, update `_build_income_df`. Replace the section from `revenue_records` through `capex_records` (lines 293–307):

```python
    revenue_records        = _try_concepts(cik, REVENUE_CONCEPTS)
    gross_profit_records   = _fetch_xbrl(cik, "GrossProfit");                   time.sleep(0.15)
    op_income_records      = _fetch_xbrl(cik, "OperatingIncomeLoss");           time.sleep(0.15)
    net_income_records     = _try_concepts(cik, NET_INCOME_CONCEPTS)
    capex_records          = _try_concepts(cik, CAPEX_CONCEPTS)
    rd_records             = _fetch_xbrl(cik, "ResearchAndDevelopmentExpense"); time.sleep(0.15)

    revenue = _to_period_series(revenue_records, "revenue", annual)
    if revenue.is_empty():
        print(f"[EDGAR] {ticker}: no revenue data found — skipping")
        return pl.DataFrame()

    gross_profit      = _to_period_series(gross_profit_records,  "gross_profit",      annual)
    operating_income  = _to_period_series(op_income_records,     "operating_income",  annual)
    net_income        = _to_period_series(net_income_records,     "net_income",        annual)
    capex             = _to_period_series(capex_records,          "capex",             annual)
    rd_expense        = _to_period_series(rd_records,             "rd_expense",        annual)
```

Then update the join loop and the "Ensure all expected columns exist" block. Replace lines 310–328:

```python
    # Outer-join all series on period_end; missing quarters become null
    df = revenue
    for other in [gross_profit, operating_income, net_income, capex, rd_expense]:
        if not other.is_empty():
            df = df.join(other, on="period_end", how="left")
        else:
            col_name = [c for c in other.columns if c != "period_end"][0] if other.columns else None
            if col_name:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

    # Ensure all expected columns exist
    for col in ["gross_profit", "operating_income", "net_income", "capex", "rd_expense"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Capex is a cash outflow (negative in XBRL) — store as positive
    if "capex" in df.columns:
        df = df.with_columns(pl.col("capex").abs())
```

Also update the docstring for `_build_income_df`:

```python
def _build_income_df(cik: str, ticker: str, annual: bool) -> pl.DataFrame:
    """
    Fetch income statement + capex + R&D from EDGAR for one ticker.
    Returns DataFrame: [period_end, revenue, gross_profit, operating_income,
                        net_income, capex, rd_expense]
    Missing concepts produce null columns (rd_expense = null when concept returns 404).
    Returns empty DataFrame if revenue data is unavailable.
    """
```

- [ ] **Step 6: Add 5 new TTM metrics to `_compute_derived`**

In `_compute_derived`, insert the following block immediately before `return df.drop("revenue_4q_prior")` (currently line 473):

```python
    # ── 5 new TTM-based metrics ───────────────────────────────────────────────
    df = df.sort("period_end").with_columns([
        pl.col("net_income").rolling_sum(window_size=4, min_samples=4).alias("_ttm_net_income"),
        pl.col("operating_income").rolling_sum(window_size=4, min_samples=4).alias("_ttm_op_income"),
        pl.col("capex").rolling_sum(window_size=4, min_samples=4).alias("_ttm_capex"),
        pl.col("revenue").rolling_sum(window_size=4, min_samples=4).alias("_ttm_revenue"),
        pl.col("rd_expense").fill_null(0.0).rolling_sum(window_size=4, min_samples=4).alias("_ttm_rd"),
    ])

    df = df.with_columns(
        pl.col("_ttm_capex").shift(4).alias("_prior_ttm_capex")
    )

    df = df.with_columns([
        # net_income_margin: 0.0 when TTM revenue unavailable
        pl.when(
            pl.col("_ttm_revenue").is_not_null() & (pl.col("_ttm_revenue") != 0)
            & pl.col("_ttm_net_income").is_not_null()
        )
        .then(pl.col("_ttm_net_income") / pl.col("_ttm_revenue"))
        .otherwise(0.0)
        .alias("net_income_margin"),

        # free_cash_flow_margin: 0.0 when TTM revenue unavailable
        pl.when(
            pl.col("_ttm_revenue").is_not_null() & (pl.col("_ttm_revenue") != 0)
            & pl.col("_ttm_op_income").is_not_null()
            & pl.col("_ttm_capex").is_not_null()
        )
        .then((pl.col("_ttm_op_income") - pl.col("_ttm_capex")) / pl.col("_ttm_revenue"))
        .otherwise(0.0)
        .alias("free_cash_flow_margin"),

        # capex_growth_yoy: null when <8 quarters of capex history
        pl.when(
            pl.col("_prior_ttm_capex").is_not_null() & (pl.col("_prior_ttm_capex") > 0)
            & pl.col("_ttm_capex").is_not_null()
        )
        .then((pl.col("_ttm_capex") / pl.col("_prior_ttm_capex")) - 1.0)
        .otherwise(None)
        .alias("capex_growth_yoy"),

        # research_to_revenue: 0.0 when R&D concept unavailable (rd_expense = 0/null)
        pl.when(pl.col("_ttm_revenue").is_not_null() & (pl.col("_ttm_revenue") != 0))
        .then(pl.col("_ttm_rd") / pl.col("_ttm_revenue"))
        .otherwise(0.0)
        .alias("research_to_revenue"),
    ])

    # revenue_growth_accel: second derivative of YoY growth
    df = df.with_columns(
        pl.col("revenue_growth_yoy").shift(1).alias("_prior_yoy")
    )
    df = df.with_columns(
        pl.when(
            pl.col("revenue_growth_yoy").is_not_null()
            & pl.col("_prior_yoy").is_not_null()
        )
        .then(pl.col("revenue_growth_yoy") - pl.col("_prior_yoy"))
        .otherwise(0.0)
        .alias("revenue_growth_accel")
    )

    df = df.drop(["_ttm_net_income", "_ttm_op_income", "_ttm_capex", "_ttm_revenue",
                  "_ttm_rd", "_prior_ttm_capex", "_prior_yoy"])
```

- [ ] **Step 7: Expand `output_cols` in `fetch_edgar_fundamentals`**

Replace `output_cols` (lines 600–604):

```python
    output_cols = [
        "period_end", "pe_ratio_trailing", "price_to_sales", "price_to_book",
        "revenue_growth_yoy", "gross_margin", "operating_margin",
        "capex_to_revenue", "debt_to_equity", "current_ratio",
        # 5 new TTM metrics
        "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
        "revenue_growth_accel", "research_to_revenue",
    ]
```

- [ ] **Step 8: Run all tests**

```bash
pytest tests/test_edgar_fundamentals_ingestion.py tests/test_edgar_fundamentals.py -v
```

Expected: all tests pass. Fix any failures before continuing.

- [ ] **Step 9: Commit**

```bash
git add ingestion/edgar_fundamentals_ingestion.py \
        tests/test_edgar_fundamentals.py \
        tests/test_edgar_fundamentals_ingestion.py
git commit -m "feat: expand EDGAR ingestion — R&D fetch + 5 new TTM metrics, _SCHEMA 11→16"
```

---

### Task 2: Feature Module + Model Integration — FUNDAMENTAL_FEATURE_COLS 9→14, FEATURE_COLS 83→88

**Files:**
- Modify: `processing/fundamental_features.py`
- Modify: `models/train.py`
- Modify: `tools/run_refresh.sh`
- Modify: `tests/test_train.py`
- Create: `tests/test_fundamental_features.py`

- [ ] **Step 1: Write 5 failing feature tests**

Create `tests/test_fundamental_features.py`:

```python
"""Tests for the 5 new fundamental feature columns in fundamental_features.py."""
import datetime
import polars as pl
import pytest
from pathlib import Path


def _write_fund_parquet(fund_dir: Path, ticker: str, rows: list[dict]) -> None:
    """Helper: write quarterly.parquet for a ticker into fund_dir/<ticker>/."""
    path = fund_dir / ticker / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(str(path))


_ALL_FUND_COLS = [
    "pe_ratio_trailing", "price_to_sales", "price_to_book",
    "revenue_growth_yoy", "gross_margin", "operating_margin",
    "capex_to_revenue", "debt_to_equity", "current_ratio",
    "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
    "revenue_growth_accel", "research_to_revenue",
]


def _make_fund_row(ticker: str, period_end: datetime.date, **overrides) -> dict:
    base = {
        "ticker": ticker,
        "period_end": period_end,
        "pe_ratio_trailing": 25.0, "price_to_sales": 8.0, "price_to_book": 3.0,
        "revenue_growth_yoy": 0.15, "gross_margin": 0.60, "operating_margin": 0.25,
        "capex_to_revenue": 0.08, "debt_to_equity": 0.5, "current_ratio": 1.8,
        "net_income_margin": 0.20, "free_cash_flow_margin": 0.15,
        "capex_growth_yoy": 0.10, "revenue_growth_accel": 0.02,
        "research_to_revenue": 0.12,
    }
    base.update(overrides)
    return base


def test_net_income_margin_asof_picks_most_recent_past_quarter(tmp_path):
    """net_income_margin backward asof join selects the most recent quarter ≤ query date."""
    from processing.fundamental_features import join_fundamentals
    _write_fund_parquet(tmp_path, "MSFT", [
        _make_fund_row("MSFT", datetime.date(2022, 9, 30), net_income_margin=0.30),
        _make_fund_row("MSFT", datetime.date(2022, 12, 31), net_income_margin=0.35),
        # Future quarter — must NOT be picked for a date before 2023-03-31
        _make_fund_row("MSFT", datetime.date(2023, 3, 31), net_income_margin=0.40),
    ])
    price_df = pl.DataFrame({
        "ticker": ["MSFT"],
        "date": pl.Series([datetime.date(2023, 1, 15)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["net_income_margin"][0] == pytest.approx(0.35)


def test_free_cash_flow_margin_via_asof_join(tmp_path):
    """free_cash_flow_margin is correctly joined backward by date."""
    from processing.fundamental_features import join_fundamentals
    _write_fund_parquet(tmp_path, "NVDA", [
        _make_fund_row("NVDA", datetime.date(2022, 12, 31), free_cash_flow_margin=0.25),
    ])
    price_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date": pl.Series([datetime.date(2023, 2, 1)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["free_cash_flow_margin"][0] == pytest.approx(0.25)


def test_capex_growth_yoy_positive_when_accelerating(tmp_path):
    """capex_growth_yoy reflects positive growth from the most recent quarter."""
    from processing.fundamental_features import join_fundamentals
    _write_fund_parquet(tmp_path, "AMD", [
        _make_fund_row("AMD", datetime.date(2022, 12, 31), capex_growth_yoy=0.20),
    ])
    price_df = pl.DataFrame({
        "ticker": ["AMD"],
        "date": pl.Series([datetime.date(2023, 3, 1)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["capex_growth_yoy"][0] == pytest.approx(0.20)


def test_revenue_growth_accel_second_derivative(tmp_path):
    """revenue_growth_accel = current YoY growth minus prior quarter YoY growth."""
    from processing.fundamental_features import join_fundamentals
    _write_fund_parquet(tmp_path, "GOOGL", [
        # Q3 2022: YoY growth 0.10; Q4 2022: YoY growth 0.15 → accel = 0.05
        _make_fund_row("GOOGL", datetime.date(2022, 9, 30),  revenue_growth_accel=0.0),
        _make_fund_row("GOOGL", datetime.date(2022, 12, 31), revenue_growth_accel=0.05),
    ])
    price_df = pl.DataFrame({
        "ticker": ["GOOGL"],
        "date": pl.Series([datetime.date(2023, 1, 20)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path)
    assert result["revenue_growth_accel"][0] == pytest.approx(0.05)


def test_missing_fundamentals_dir_all_14_cols_present(tmp_path):
    """When fundamentals directory is missing, all 14 columns are present in output."""
    from processing.fundamental_features import join_fundamentals, FUNDAMENTAL_FEATURE_COLS
    price_df = pl.DataFrame({
        "ticker": ["MSFT"],
        "date": pl.Series([datetime.date(2023, 1, 1)], dtype=pl.Date),
    })
    result = join_fundamentals(price_df, tmp_path / "nonexistent")
    assert len(FUNDAMENTAL_FEATURE_COLS) == 14
    for col in FUNDAMENTAL_FEATURE_COLS:
        assert col in result.columns, f"{col} missing from result"
    assert len(result) == 1
```

- [ ] **Step 2: Write 5 failing train tests and update count assertions**

Append to `tests/test_train.py`:

```python
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
```

Also update the existing count assertions in `tests/test_train.py`. Find and update these 3 lines (each currently asserts `== 83`):

- In `test_feature_cols_includes_uspto_patent` (line ~499): `assert len(FEATURE_COLS) == 83` → `assert len(FEATURE_COLS) == 88`
- In `test_feature_cols_includes_labor` (line ~550): `assert len(FEATURE_COLS) == 83` → `assert len(FEATURE_COLS) == 88`
- In `test_feature_cols_includes_census` (line ~599): `assert len(FEATURE_COLS) == 83` → `assert len(FEATURE_COLS) == 88`

Update docstrings on two tests:
- `test_tier_feature_cols_medium_equals_feature_cols` docstring: `83 features` → `88 features`
- `test_tier_medium_equals_feature_cols_after_gov_integration` docstring: `now 83` → `now 88`

Also update `_write_fundamentals_fixture` (starting at line ~42) to write all 14 fundamental columns:

```python
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
```

- [ ] **Step 3: Run new tests to confirm they fail**

```bash
pytest tests/test_fundamental_features.py tests/test_train.py::test_feature_cols_includes_edgar_expanded -v 2>&1 | tail -20
```

Expected: ImportError (FUNDAMENTAL_FEATURE_COLS not exported yet) and count failures.

- [ ] **Step 4: Update `processing/fundamental_features.py`**

Replace the entire file contents:

```python
from pathlib import Path

import polars as pl

FUNDAMENTAL_FEATURE_COLS: list[str] = [
    # Existing 9
    "pe_ratio_trailing",
    "price_to_sales",
    "price_to_book",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "capex_to_revenue",
    "debt_to_equity",
    "current_ratio",
    # 5 new TTM-based metrics
    "net_income_margin",
    "free_cash_flow_margin",
    "capex_growth_yoy",
    "revenue_growth_accel",
    "research_to_revenue",
]

_FUND_SCHEMA = {
    "ticker": pl.Utf8,
    "period_end": pl.Date,
    "pe_ratio_trailing": pl.Float64,
    "price_to_sales": pl.Float64,
    "price_to_book": pl.Float64,
    "revenue_growth_yoy": pl.Float64,
    "gross_margin": pl.Float64,
    "operating_margin": pl.Float64,
    "capex_to_revenue": pl.Float64,
    "debt_to_equity": pl.Float64,
    "current_ratio": pl.Float64,
    "net_income_margin": pl.Float64,
    "free_cash_flow_margin": pl.Float64,
    "capex_growth_yoy": pl.Float64,
    "revenue_growth_accel": pl.Float64,
    "research_to_revenue": pl.Float64,
}


def join_fundamentals(
    price_df: pl.DataFrame,
    fundamentals_dir: Path = Path("data/raw/financials/fundamentals"),
) -> pl.DataFrame:
    """
    Join the most recent available quarterly fundamental snapshot onto each
    price_df row using a backward asof join on date/period_end, per ticker.

    For each (ticker, date) row: selects the fundamental row where
    period_end <= date and period_end is maximised (most recent quarter).

    If no fundamentals exist for a ticker (or at all), fundamental columns
    are null — LightGBM handles nulls natively; Ridge/RF use saved imputation
    medians at training time.

    Args:
        price_df: DataFrame with at minimum columns [ticker, date].
        fundamentals_dir: Root directory containing <TICKER>/quarterly.parquet files.

    Returns:
        price_df with 14 additional fundamental columns appended.
    """
    null_fund_cols = [pl.lit(None).cast(pl.Float64).alias(c) for c in FUNDAMENTAL_FEATURE_COLS]

    glob = str(fundamentals_dir / "*" / "quarterly.parquet")

    try:
        fund_df = (
            pl.scan_parquet(glob, schema=_FUND_SCHEMA)
            .with_columns(pl.col("period_end").cast(pl.Date))
            .collect()
            .sort(["ticker", "period_end"])
        )
    except FileNotFoundError:
        return price_df.with_columns(null_fund_cols)

    if fund_df.is_empty():
        return price_df.with_columns(null_fund_cols)

    price_sorted = (
        price_df
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["ticker", "date"])
    )

    joined = price_sorted.join_asof(
        fund_df.select(["ticker", "period_end"] + FUNDAMENTAL_FEATURE_COLS),
        left_on="date",
        right_on="period_end",
        by="ticker",
        strategy="backward",
        check_sortedness=False,
    )

    # Drop the right join key (period_end) — keep output schema consistent with null-fallback path
    return joined.drop("period_end").sort(["ticker", "date"])
```

- [ ] **Step 5: Update `FUND_FEATURE_COLS` in `models/train.py`**

Replace the existing `FUND_FEATURE_COLS` definition (lines 57–61):

```python
FUND_FEATURE_COLS = [
    "pe_ratio_trailing", "price_to_sales", "price_to_book",
    "revenue_growth_yoy", "gross_margin", "operating_margin",
    "capex_to_revenue", "debt_to_equity", "current_ratio",
    # 5 new TTM-based metrics (medium + long tiers only — quarterly cadence too slow for 5d/20d)
    "net_income_margin", "free_cash_flow_margin", "capex_growth_yoy",
    "revenue_growth_accel", "research_to_revenue",
]
```

Update the medium tier comment (line ~150):

```python
    "medium": list(FEATURE_COLS),    # all 88 features (copy to avoid shared mutable reference)
```

- [ ] **Step 6: Update `tools/run_refresh.sh` — insert EDGAR as step 2/16, renumber 2–15 → 3–16**

Replace the entire file:

```bash
#!/bin/zsh
# tools/run_refresh.sh
# Full pipeline refresh — run from the project root.
# Each step must succeed before the next runs (set -e).
set -e

cd "$(dirname "$0")/.."
echo "Starting full pipeline refresh at $(date)"

echo ""
echo "=== 1/16  OHLCV price data ==="
python ingestion/ohlcv_ingestion.py

echo ""
echo "=== 2/16  EDGAR fundamentals (reads OHLCV for valuation ratios) ==="
python ingestion/edgar_fundamentals_ingestion.py

echo ""
echo "=== 3/16  Short interest (FINRA) ==="
python ingestion/short_interest_ingestion.py

echo ""
echo "=== 4/16  Earnings surprises ==="
python ingestion/earnings_ingestion.py

echo ""
echo "=== 5/16  News articles (GDELT + RSS) ==="
python ingestion/news_ingestion.py

echo ""
echo "=== 6/16  NLP sentiment scoring (FinBERT) ==="
python processing/nlp_pipeline.py

echo ""
echo "=== 7/16  Sentiment features ==="
python processing/sentiment_features.py

echo ""
echo "=== 8/16  Graph features ==="
python processing/graph_features.py

echo ""
echo "=== 9/16  13F institutional holdings (incremental) ==="
python ingestion/sec_13f_ingestion.py

echo ""
echo "=== 10/16  Ownership features ==="
python processing/ownership_features.py

echo ""
echo "=== 11/16  SAM.gov government contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 12/16  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== 13/16  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== 14/16  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 15/16  BLS JOLTS tech sector job openings ==="
python ingestion/bls_jolts_ingestion.py

echo ""
echo "=== 16/16  Census international trade (semiconductors + DC equipment) ==="
python ingestion/census_trade_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
echo "Run: python models/train.py  (to retrain with fresh data)"
```

- [ ] **Step 7: Run all tests**

```bash
pytest tests/test_fundamental_features.py tests/test_edgar_fundamentals_ingestion.py \
       tests/test_edgar_fundamentals.py tests/test_train.py -v 2>&1 | tail -40
```

Expected: all tests pass. Fix any failures before continuing.

- [ ] **Step 8: Run full test suite**

```bash
pytest tests/ -m 'not integration' -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add processing/fundamental_features.py \
        models/train.py \
        tools/run_refresh.sh \
        tests/test_fundamental_features.py \
        tests/test_train.py
git commit -m "feat: FUNDAMENTAL_FEATURE_COLS 9→14, FEATURE_COLS 83→88, EDGAR wired into run_refresh.sh step 2/16"
```

---

## Self-Review

**Spec coverage:**
- ✅ 5 new metrics: net_income_margin, free_cash_flow_margin, capex_growth_yoy, revenue_growth_accel, research_to_revenue
- ✅ R&D XBRL fetch added to `_build_income_df` with 404 fallback (null → 0.0 via fill_null)
- ✅ `_SCHEMA` 11→16
- ✅ `FUNDAMENTAL_FEATURE_COLS` renamed (public) and expanded 9→14
- ✅ `FUND_FEATURE_COLS` in train.py expanded 9→14 → FEATURE_COLS 83→88
- ✅ medium tier comment updated (83→88)
- ✅ long tier: `FUND_FEATURE_COLS` already present; expanding it picks up the 5 new cols automatically
- ✅ `run_refresh.sh` step 2/16 inserted after OHLCV, all others renumbered 2–15 → 3–16
- ✅ `capex_growth_yoy` = null when <8 quarters (shift(4) on TTM series gives null for early rows)
- ✅ `revenue_growth_accel` = 0.0 when only one YoY data point (prior_yoy is null → otherwise 0.0)
- ✅ `research_to_revenue` = 0.0 when R&D concept unavailable (`fill_null(0.0)` on rd_expense before rolling sum)
- ✅ 5 ingestion tests + 5 feature tests + 5 train tests
- ✅ `_write_fundamentals_fixture` in test_train.py updated to write all 14 fundamental columns
- ✅ Existing test fixtures updated to include `rd_expense` column

**Placeholder scan:** No TBD or TODO found.

**Type consistency:** `_SCHEMA` uses `pa.float64()`, `_FUND_SCHEMA` uses `pl.Float64`, `FUND_FEATURE_COLS` lists strings — consistent throughout.
