# Phase 2a: Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the data pipeline with quarterly fundamental ingestion (yfinance), 1-year forward return labels, and a fundamentals joiner — providing everything `models/train.py` needs to train the ensemble predictor.

**Architecture:** Three new modules feed the model training pipeline. `fundamental_ingestion.py` pulls quarterly income-statement and balance-sheet metrics from yfinance and writes per-ticker Parquet. `label_builder.py` reads OHLCV and computes 1-year forward annualized returns, excluding rows with incomplete forward windows. `fundamental_features.py` joins the most recent available fundamental snapshot onto each price-feature row using Polars `join_asof`. `ohlcv_ingestion.py` gains a `--bootstrap` flag for pulling max-available history.

**Tech Stack:** Python 3.11+, yfinance>=0.2, lightgbm>=4.0, polars>=1.0, pyarrow>=17.0, pytest

---

## File Structure

**Modify:**
- `pyproject.toml` — add `lightgbm>=4.0` to core `[project.dependencies]`
- `ingestion/ohlcv_ingestion.py:79-89` — add `--bootstrap` argparse flag; `period="max"` when set, `period="5d"` default

**Create:**
- `ingestion/fundamental_ingestion.py` — fetch quarterly fundamentals from yfinance, write to `data/raw/financials/fundamentals/<TICKER>/quarterly.parquet`
- `processing/label_builder.py` — compute 1-year forward return label for each ticker×date; drop incomplete windows
- `processing/fundamental_features.py` — join most recent quarter's fundamentals onto a price DataFrame using `join_asof`

**Tests:**
- `tests/test_fundamental_ingestion.py`
- `tests/test_label_builder.py`
- `tests/test_fundamental_features.py`

---

### Task 1: Add lightgbm to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

No tests — verify by install and existing suite.

- [ ] **Step 1: Read pyproject.toml**

Run: `cat pyproject.toml`

Confirm `lightgbm` is not already present in `[project.dependencies]`.

- [ ] **Step 2: Add lightgbm to core dependencies**

In `pyproject.toml`, add `"lightgbm>=4.0",` to the `[project.dependencies]` list, after `"yfinance>=0.2"`:

```toml
[project]
name = "ai-infra-predictor"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "polars>=1.0",
  "pandas>=2.0",
  "duckdb>=1.1",
  "pyarrow>=17.0",
  "requests>=2.32",
  "httpx>=0.27",
  "aiohttp>=3.10",
  "websockets>=13.0",
  "feedparser>=6.0",
  "transformers>=4.45",
  "torch>=2.4",
  "yfinance>=0.2",
  "lightgbm>=4.0",
  "python-dotenv>=1.0",
]
```

- [ ] **Step 3: Install lightgbm into the venv**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pip install "lightgbm>=4.0" --quiet
```

Expected: installs successfully (shows installed version, e.g. `lightgbm-4.x.x`).

- [ ] **Step 4: Verify existing tests still pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/ -m "not integration" -q
```

Expected: `38 passed, 4 deselected`

- [ ] **Step 5: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && git add pyproject.toml && git commit -m "chore: add lightgbm>=4.0 to core dependencies"
```

---

### Task 2: OHLCV Bootstrap Flag

**Files:**
- Modify: `ingestion/ohlcv_ingestion.py:79-89`

The `__main__` block currently hard-codes `period="2y"`. The bootstrap flag enables `period="max"` (~20y) for initial history download; daily scheduler updates use `period="5d"`.

No new unit tests — `fetch_ohlcv` and `save_ohlcv` are already tested. Verify the flag parses correctly by running the help text.

- [ ] **Step 1: Replace the `__main__` block**

Open `ingestion/ohlcv_ingestion.py`. Replace lines 79-89 (the entire `if __name__ == "__main__":` block) with:

```python
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download OHLCV price data for all watchlist tickers.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Download maximum available history (~20y). Omit for daily 5-day refresh.",
    )
    args = parser.parse_args()

    period = "max" if args.bootstrap else "5d"
    output_dir = Path("data/raw")

    for ticker in TICKERS:
        print(f"[OHLCV] Downloading {ticker} (period={period})...")
        records = fetch_ohlcv(ticker, period=period)
        save_ohlcv(records, ticker, output_dir)
        time.sleep(1)  # Rate limit — Yahoo Finance fair use
    print(f"[OHLCV] Done. {len(TICKERS)} tickers written to {output_dir}/financials/ohlcv/")
```

- [ ] **Step 2: Verify help text**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/python ingestion/ohlcv_ingestion.py --help
```

Expected output contains:
```
  --bootstrap  Download maximum available history (~20y)...
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/ -m "not integration" -q
```

Expected: `38 passed, 4 deselected`

- [ ] **Step 4: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && git add ingestion/ohlcv_ingestion.py && git commit -m "feat: add --bootstrap flag to ohlcv_ingestion for max-history download"
```

---

### Task 3: Fundamental Ingestion

**Files:**
- Create: `tests/test_fundamental_ingestion.py`
- Create: `ingestion/fundamental_ingestion.py`

Fetches quarterly income-statement and balance-sheet metrics from yfinance. Valuation ratios (P/E, P/S, P/B) come from `ticker.info` and are only populated for the most recent quarter (yfinance does not provide historical ratios). Per-quarter metrics (margins, capex/revenue, D/E, current ratio) are computed from quarterly financial statements.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamental_ingestion.py`:

```python
import datetime
import pytest
import pandas as pd
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock
from pathlib import Path


def _make_mock_ticker():
    """
    Mock yf.Ticker with two quarters of financial data.
    quarterly_financials: rows=metric names, cols=period-end Timestamps.
    """
    mock = MagicMock()

    mock.quarterly_financials = pd.DataFrame({
        pd.Timestamp("2024-03-31"): {
            "Total Revenue":       44_000_000_000.0,
            "Gross Profit":        32_000_000_000.0,
            "Operating Income":    20_000_000_000.0,
            "Capital Expenditure": -6_000_000_000.0,
        },
        pd.Timestamp("2023-03-31"): {
            "Total Revenue":       36_000_000_000.0,
            "Gross Profit":        27_000_000_000.0,
            "Operating Income":    16_000_000_000.0,
            "Capital Expenditure": -5_000_000_000.0,
        },
    })

    mock.quarterly_balance_sheet = pd.DataFrame({
        pd.Timestamp("2024-03-31"): {
            "Stockholders Equity": 100_000_000_000.0,
            "Total Debt":           50_000_000_000.0,
            "Current Assets":       80_000_000_000.0,
            "Current Liabilities":  30_000_000_000.0,
        },
        pd.Timestamp("2023-03-31"): {
            "Stockholders Equity":  90_000_000_000.0,
            "Total Debt":           45_000_000_000.0,
            "Current Assets":       70_000_000_000.0,
            "Current Liabilities":  28_000_000_000.0,
        },
    })

    mock.info = {
        "trailingPE": 35.0,
        "priceToSalesTrailing12Months": 12.5,
        "priceToBook": 10.2,
    }

    return mock


def test_fetch_fundamentals_returns_one_record_per_quarter():
    """fetch_fundamentals returns one dict per quarter in quarterly_financials."""
    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=_make_mock_ticker()):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        records = fetch_fundamentals("NVDA")

    assert len(records) == 2
    assert all(r["ticker"] == "NVDA" for r in records)
    assert all(isinstance(r["period_end"], datetime.date) for r in records)


def test_fetch_fundamentals_computes_gross_margin():
    """gross_margin = Gross Profit / Total Revenue for each quarter."""
    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=_make_mock_ticker()):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        records = fetch_fundamentals("NVDA")

    # Sort by period_end descending (most recent first)
    records.sort(key=lambda r: r["period_end"], reverse=True)
    # 2024-Q1: 32B / 44B ≈ 0.7273
    assert records[0]["gross_margin"] == pytest.approx(32 / 44, rel=1e-4)
    # 2023-Q1: 27B / 36B = 0.75
    assert records[1]["gross_margin"] == pytest.approx(27 / 36, rel=1e-4)


def test_fetch_fundamentals_valuation_ratios_only_on_most_recent():
    """pe_ratio_trailing, price_to_sales, price_to_book are non-null only for most recent quarter."""
    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=_make_mock_ticker()):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        records = fetch_fundamentals("NVDA")

    records.sort(key=lambda r: r["period_end"], reverse=True)
    assert records[0]["pe_ratio_trailing"] == pytest.approx(35.0)
    assert records[0]["price_to_sales"] == pytest.approx(12.5)
    assert records[0]["price_to_book"] == pytest.approx(10.2)
    assert records[1]["pe_ratio_trailing"] is None
    assert records[1]["price_to_sales"] is None
    assert records[1]["price_to_book"] is None


def test_fetch_fundamentals_returns_empty_for_empty_financials():
    """fetch_fundamentals returns [] when quarterly_financials is empty."""
    mock = MagicMock()
    mock.quarterly_financials = pd.DataFrame()
    mock.quarterly_balance_sheet = pd.DataFrame()
    mock.info = {}

    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=mock):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        assert fetch_fundamentals("UNKNOWN") == []


def test_save_fundamentals_writes_parquet_with_correct_schema(tmp_path):
    """save_fundamentals writes snappy Parquet at the expected path with correct columns."""
    records = [
        {
            "ticker": "NVDA",
            "period_end": datetime.date(2024, 3, 31),
            "pe_ratio_trailing": 35.0,
            "price_to_sales": 12.5,
            "price_to_book": 10.2,
            "revenue_growth_yoy": 0.22,
            "gross_margin": 0.727,
            "operating_margin": 0.455,
            "capex_to_revenue": 0.136,
            "debt_to_equity": 0.5,
            "current_ratio": 2.67,
        }
    ]

    from ingestion.fundamental_ingestion import save_fundamentals
    save_fundamentals(records, "NVDA", tmp_path)

    path = tmp_path / "financials" / "fundamentals" / "NVDA" / "quarterly.parquet"
    assert path.exists()

    table = pq.read_table(str(path))
    assert "ticker" in table.schema.names
    assert "period_end" in table.schema.names
    assert "gross_margin" in table.schema.names
    assert table.num_rows == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/test_fundamental_ingestion.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'ingestion.fundamental_ingestion'`

- [ ] **Step 3: Implement ingestion/fundamental_ingestion.py**

Create `ingestion/fundamental_ingestion.py`:

```python
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

TICKERS = [
    "MSFT", "AMZN", "GOOGL", "META",           # Hyperscalers
    "NVDA", "AMD", "AVGO", "MRVL", "TSM",      # AI chips
    "ASML", "AMAT", "LRCX", "KLAC",            # Foundry equipment
    "VRT", "SMCI", "DELL", "HPE",              # AI infrastructure
    "EQIX", "DLR", "AMT",                      # Data center REITs
    "CEG", "VST", "NRG", "TLN",               # Power / nuclear
]

SCHEMA = pa.schema([
    pa.field("ticker", pa.string()),
    pa.field("period_end", pa.date32()),
    pa.field("pe_ratio_trailing", pa.float64()),
    pa.field("price_to_sales", pa.float64()),
    pa.field("price_to_book", pa.float64()),
    pa.field("revenue_growth_yoy", pa.float64()),
    pa.field("gross_margin", pa.float64()),
    pa.field("operating_margin", pa.float64()),
    pa.field("capex_to_revenue", pa.float64()),
    pa.field("debt_to_equity", pa.float64()),
    pa.field("current_ratio", pa.float64()),
])


def _safe_get(df: pd.DataFrame, period_col, *row_names) -> float | None:
    """Try multiple row-name variants; return float value or None if missing/NaN."""
    for name in row_names:
        if name in df.index:
            try:
                v = df.loc[name, period_col]
                return float(v) if pd.notna(v) else None
            except (KeyError, TypeError, ValueError):
                continue
    return None


def fetch_fundamentals(ticker: str) -> list[dict]:
    """
    Fetch quarterly fundamentals from yfinance for one ticker.

    Income-statement metrics (margins, capex/revenue, revenue growth) are
    available per quarter. Valuation ratios (P/E, P/S, P/B) come from
    ticker.info (current snapshot) and are stored only for the most recent
    quarter; all older quarters have null for those fields.

    Returns [] when no quarterly financials are available.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    try:
        qf = t.quarterly_financials
    except Exception:
        qf = pd.DataFrame()

    try:
        qbs = t.quarterly_balance_sheet
    except Exception:
        qbs = pd.DataFrame()

    if qf.empty:
        return []

    # Most-recent period first
    periods = sorted(qf.columns, reverse=True)
    most_recent = periods[0]

    def float_info(key: str) -> float | None:
        v = info.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    records = []
    for i, period_col in enumerate(periods):
        revenue = _safe_get(qf, period_col, "Total Revenue")
        gross_profit = _safe_get(qf, period_col, "Gross Profit")
        operating_income = _safe_get(qf, period_col, "Operating Income", "EBIT")
        capex = _safe_get(qf, period_col, "Capital Expenditure")

        gross_margin = gross_profit / revenue if (gross_profit is not None and revenue) else None
        operating_margin = operating_income / revenue if (operating_income is not None and revenue) else None
        capex_to_revenue = abs(capex) / revenue if (capex is not None and revenue) else None

        # Revenue YoY: same quarter 4 periods back
        revenue_growth_yoy = None
        if revenue is not None and i + 4 < len(periods):
            prior_rev = _safe_get(qf, periods[i + 4], "Total Revenue")
            if prior_rev and prior_rev != 0:
                revenue_growth_yoy = (revenue - prior_rev) / abs(prior_rev)

        # Balance sheet
        total_equity = _safe_get(qbs, period_col, "Stockholders Equity", "Total Stockholder Equity") if not qbs.empty else None
        total_debt = _safe_get(qbs, period_col, "Total Debt", "Long Term Debt") if not qbs.empty else None
        current_assets = _safe_get(qbs, period_col, "Current Assets", "Total Current Assets") if not qbs.empty else None
        current_liabilities = _safe_get(qbs, period_col, "Current Liabilities", "Total Current Liabilities") if not qbs.empty else None

        debt_to_equity = total_debt / total_equity if (total_debt is not None and total_equity) else None
        current_ratio = current_assets / current_liabilities if (current_assets is not None and current_liabilities) else None

        is_most_recent = period_col == most_recent

        records.append({
            "ticker": ticker,
            "period_end": period_col.date() if hasattr(period_col, "date") else period_col,
            "pe_ratio_trailing": float_info("trailingPE") if is_most_recent else None,
            "price_to_sales": float_info("priceToSalesTrailing12Months") if is_most_recent else None,
            "price_to_book": float_info("priceToBook") if is_most_recent else None,
            "revenue_growth_yoy": revenue_growth_yoy,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "capex_to_revenue": capex_to_revenue,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
        })

    return records


def save_fundamentals(records: list[dict], ticker: str, output_dir: Path) -> None:
    """
    Write fundamental records to:
    <output_dir>/financials/fundamentals/<TICKER>/quarterly.parquet
    Overwrites any existing file for the ticker.
    """
    if not records:
        print(f"[Fundamentals] {ticker}: no data available")
        return

    path = output_dir / "financials" / "fundamentals" / ticker / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    pq.write_table(table, path, compression="snappy")
    print(f"[Fundamentals] {ticker}: {len(records)} quarters → {path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    output_dir = Path("data/raw")
    for ticker in TICKERS:
        print(f"[Fundamentals] Fetching {ticker}...")
        records = fetch_fundamentals(ticker)
        save_fundamentals(records, ticker, output_dir)
        time.sleep(2)  # yfinance info calls are heavier — be conservative
    print(f"[Fundamentals] Done. {len(TICKERS)} tickers written.")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/test_fundamental_ingestion.py -v
```

Expected: `5 passed`

**Troubleshooting:**
- If `_safe_get` raises `KeyError` on `df.loc[name, period_col]`: the mock's DataFrame may have the wrong axis orientation. Confirm `qf.index` contains metric names (e.g. `"Total Revenue"`) and `qf.columns` contains Timestamps. If reversed, swap `df.loc[name, period_col]` to `df.loc[period_col, name]` and update the mock accordingly.
- If `period_col.date()` raises `AttributeError`: `period_col` is already a `datetime.date`. Add `if hasattr(period_col, "date")` guard (already present in the implementation above).

- [ ] **Step 5: Run full test suite**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/ -m "not integration" -q
```

Expected: `43 passed, 4 deselected` (38 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && git add ingestion/fundamental_ingestion.py tests/test_fundamental_ingestion.py && git commit -m "feat: quarterly fundamental ingestion via yfinance — margins, D/E, capex-to-revenue"
```

---

### Task 4: Label Builder

**Files:**
- Create: `tests/test_label_builder.py`
- Create: `processing/label_builder.py`

Reads all OHLCV Parquet files and computes `label_return_1y = close_price[t+252] / close_price[t] - 1` for each ticker×date. Rows where the 252-day forward window is incomplete are excluded — this is the core look-ahead leakage guard. Uses Polars `shift(-252).over("ticker")` to shift within each ticker's time series without mixing tickers.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_label_builder.py`:

```python
import datetime
import pytest
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def _write_ohlcv_fixture(ohlcv_dir: Path, ticker: str = "NVDA", n: int = 300) -> None:
    """Write n consecutive trading days of OHLCV for one ticker."""
    path = ohlcv_dir / ticker / "2024.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    prices = [100.0 + i * 0.1 for i in range(n)]  # linearly increasing
    table = pa.table({
        "ticker": [ticker] * n,
        "date": pa.array(dates, type=pa.date32()),
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low":  [p - 1.0 for p in prices],
        "close_price": prices,
        "volume": [1_000_000] * n,
    })
    pq.write_table(table, str(path))


def test_build_labels_returns_n_minus_252_rows(tmp_path):
    """With 300 rows, exactly 300-252=48 rows have complete forward windows."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir)

    assert len(result) == 300 - 252


def test_build_labels_computes_correct_return(tmp_path):
    """label_return_1y[0] == close_price[252] / close_price[0] - 1."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir).sort(["ticker", "date"])

    # close_price[0] = 100.0, close_price[252] = 100.0 + 252*0.1 = 125.2
    expected_label = 125.2 / 100.0 - 1  # 0.252
    assert result["label_return_1y"][0] == pytest.approx(expected_label, rel=1e-4)


def test_build_labels_has_no_null_labels(tmp_path):
    """All returned rows have non-null label_return_1y."""
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv_fixture(ohlcv_dir, "NVDA", n=300)

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir)

    assert result["label_return_1y"].null_count() == 0


def test_build_labels_returns_empty_dataframe_for_missing_data(tmp_path):
    """Returns empty DataFrame with correct schema when no OHLCV data exists."""
    ohlcv_dir = tmp_path / "ohlcv_empty"
    ohlcv_dir.mkdir()

    from processing.label_builder import build_labels
    result = build_labels(ohlcv_dir=ohlcv_dir)

    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()
    assert "ticker" in result.columns
    assert "date" in result.columns
    assert "label_return_1y" in result.columns
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/test_label_builder.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'processing.label_builder'`

- [ ] **Step 3: Implement processing/label_builder.py**

Create `processing/label_builder.py`:

```python
from pathlib import Path

import polars as pl

_EMPTY_SCHEMA = {"ticker": pl.String, "date": pl.Date, "label_return_1y": pl.Float64}


def build_labels(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
) -> pl.DataFrame:
    """
    Compute 1-year forward annualized return for each ticker×date row.

    label_return_1y = close_price[t+252] / close_price[t] - 1

    Uses row-offset shift within each ticker partition (Polars shift(-252).over("ticker")).
    Rows where the 252-day forward price is unavailable are dropped — this prevents
    look-ahead leakage in model training.

    Returns DataFrame with columns: ticker (String), date (Date), label_return_1y (Float64).
    Returns empty DataFrame with the same schema when no data exists.
    """
    glob = str(ohlcv_dir / "*" / "*.parquet")

    try:
        df = (
            pl.scan_parquet(glob)
            .select(["ticker", "date", "close_price"])
            .collect()
        )
    except Exception:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    if df.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    result = (
        df
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["ticker", "date"])
        .with_columns(
            pl.col("close_price")
            .shift(-252)
            .over("ticker")
            .alias("future_price")
        )
        .filter(pl.col("future_price").is_not_null())
        .with_columns(
            (pl.col("future_price") / pl.col("close_price") - 1).alias("label_return_1y")
        )
        .select(["ticker", "date", "label_return_1y"])
    )

    return result


if __name__ == "__main__":
    import pyarrow.parquet as pq
    from dotenv import load_dotenv
    load_dotenv()

    ohlcv_dir = Path("data/raw/financials/ohlcv")
    print("[Labels] Building 1-year forward return labels...")
    df = build_labels(ohlcv_dir=ohlcv_dir)

    if df.is_empty():
        print("[Labels] No OHLCV data found. Run ingestion/ohlcv_ingestion.py --bootstrap first.")
    else:
        out_path = Path("data/raw/financials/labels.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(out_path), compression="snappy")
        print(f"[Labels] {len(df)} labeled rows → {out_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/test_label_builder.py -v
```

Expected: `4 passed`

**Troubleshooting:**
- If `pl.scan_parquet(glob)` raises `FileNotFoundError` when the directory exists but is empty: wrap in try/except returning the empty schema DataFrame (already done above).
- If test `test_build_labels_computes_correct_return` fails with a small float error: the linearly increasing fixture has close_price[252] = 100.0 + 252 * 0.1 = 125.2. Verify with `result.sort(["ticker","date"])["label_return_1y"][0]` directly in a Python shell.

- [ ] **Step 5: Run full test suite**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/ -m "not integration" -q
```

Expected: `47 passed, 4 deselected`

- [ ] **Step 6: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && git add processing/label_builder.py tests/test_label_builder.py && git commit -m "feat: 1-year forward return label builder — look-ahead-safe via shift(-252).over(ticker)"
```

---

### Task 5: Fundamental Features Joiner

**Files:**
- Create: `tests/test_fundamental_features.py`
- Create: `processing/fundamental_features.py`

Joins the most recent available quarterly fundamental snapshot onto each price-feature row using Polars `join_asof(strategy="backward")`. For a price row on date `d`, this selects the most recent quarter with `period_end <= d` for the same ticker. If no fundamentals exist for a ticker, all fundamental columns are null (handled gracefully — LightGBM supports nulls, Ridge/RF use imputation at training time).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamental_features.py`:

```python
import datetime
import pytest
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path


def _write_fundamentals_fixture(fund_dir: Path, ticker: str, quarters: list[dict]) -> None:
    """Write fundamental fixture Parquet for one ticker."""
    path = fund_dir / ticker / "quarterly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(quarters).write_parquet(str(path))


def test_join_fundamentals_picks_most_recent_quarter_before_date(tmp_path):
    """join_fundamentals selects period_end <= date (backward asof join)."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "date":   [datetime.date(2024, 5, 1), datetime.date(2024, 8, 1)],
        "close_price": [900.0, 950.0],
    })

    quarters = [
        {"ticker": "NVDA", "period_end": datetime.date(2024, 3, 31),
         "pe_ratio_trailing": 30.0, "price_to_sales": 10.0, "price_to_book": 5.0,
         "revenue_growth_yoy": 0.2, "gross_margin": 0.70, "operating_margin": 0.45,
         "capex_to_revenue": 0.10, "debt_to_equity": 0.30, "current_ratio": 2.0},
        {"ticker": "NVDA", "period_end": datetime.date(2024, 6, 30),
         "pe_ratio_trailing": 35.0, "price_to_sales": 12.0, "price_to_book": 6.0,
         "revenue_growth_yoy": 0.3, "gross_margin": 0.72, "operating_margin": 0.50,
         "capex_to_revenue": 0.12, "debt_to_equity": 0.25, "current_ratio": 2.1},
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters)

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    # 2024-05-01: only 2024-03-31 is available (2024-06-30 is in the future)
    row_may = result.filter(pl.col("date") == datetime.date(2024, 5, 1))
    assert row_may["pe_ratio_trailing"][0] == pytest.approx(30.0)
    assert row_may["gross_margin"][0] == pytest.approx(0.70)

    # 2024-08-01: 2024-06-30 is now the most recent
    row_aug = result.filter(pl.col("date") == datetime.date(2024, 8, 1))
    assert row_aug["pe_ratio_trailing"][0] == pytest.approx(35.0)
    assert row_aug["gross_margin"][0] == pytest.approx(0.72)


def test_join_fundamentals_returns_null_columns_when_no_data(tmp_path):
    """When fundamentals directory is empty, all fundamental columns are null."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA"],
        "date":   [datetime.date(2024, 5, 1)],
        "close_price": [900.0],
    })

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=empty_dir)

    assert "gross_margin" in result.columns
    assert result["gross_margin"][0] is None


def test_join_fundamentals_preserves_all_price_rows(tmp_path):
    """join_fundamentals returns same number of rows as input price_df."""
    price_df = pl.DataFrame({
        "ticker": ["NVDA", "NVDA", "AMZN"],
        "date":   [datetime.date(2024, 1, 1), datetime.date(2024, 6, 1), datetime.date(2024, 6, 1)],
        "close_price": [500.0, 900.0, 180.0],
    })

    quarters_nvda = [
        {"ticker": "NVDA", "period_end": datetime.date(2023, 12, 31),
         "pe_ratio_trailing": None, "price_to_sales": None, "price_to_book": None,
         "revenue_growth_yoy": 0.1, "gross_margin": 0.65, "operating_margin": 0.4,
         "capex_to_revenue": 0.08, "debt_to_equity": 0.4, "current_ratio": 1.8},
    ]
    _write_fundamentals_fixture(tmp_path, "NVDA", quarters_nvda)
    # No fundamentals for AMZN → AMZN rows get all-null fundamental cols

    from processing.fundamental_features import join_fundamentals
    result = join_fundamentals(price_df, fundamentals_dir=tmp_path)

    assert len(result) == 3
    # NVDA 2024-01-01 gets 2023-12-31 quarter
    nvda_jan = result.filter((pl.col("ticker") == "NVDA") & (pl.col("date") == datetime.date(2024, 1, 1)))
    assert nvda_jan["gross_margin"][0] == pytest.approx(0.65)
    # AMZN has no fundamentals → null
    amzn = result.filter(pl.col("ticker") == "AMZN")
    assert amzn["gross_margin"][0] is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/test_fundamental_features.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'processing.fundamental_features'`

- [ ] **Step 3: Implement processing/fundamental_features.py**

Create `processing/fundamental_features.py`:

```python
from pathlib import Path

import polars as pl

_FUND_COLS = [
    "pe_ratio_trailing",
    "price_to_sales",
    "price_to_book",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "capex_to_revenue",
    "debt_to_equity",
    "current_ratio",
]


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
        price_df with 9 additional fundamental columns appended.
    """
    null_fund_cols = [pl.lit(None).cast(pl.Float64).alias(c) for c in _FUND_COLS]

    glob = str(fundamentals_dir / "*" / "quarterly.parquet")

    try:
        fund_df = (
            pl.scan_parquet(glob)
            .with_columns(pl.col("period_end").cast(pl.Date))
            .collect()
            .sort(["ticker", "period_end"])
        )
    except Exception:
        return price_df.with_columns(null_fund_cols)

    if fund_df.is_empty():
        return price_df.with_columns(null_fund_cols)

    price_sorted = (
        price_df
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["ticker", "date"])
    )

    joined = price_sorted.join_asof(
        fund_df.select(["ticker", "period_end"] + _FUND_COLS),
        left_on="date",
        right_on="period_end",
        by="ticker",
        strategy="backward",
    )

    # Restore original row order (sort by ticker then date)
    return joined.sort(["ticker", "date"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/test_fundamental_features.py -v
```

Expected: `3 passed`

**Troubleshooting:**
- If `join_asof` raises `InvalidOperationError: join_asof requires sorted data`: confirm `fund_df` is sorted by `["ticker", "period_end"]` and `price_sorted` by `["ticker", "date"]` (both are sorted in the implementation above).
- If AMZN rows show non-null fundamentals when only NVDA data exists: `join_asof` with `by="ticker"` should produce null when no matching ticker exists in `fund_df`. If it bleeds across tickers, check that `by="ticker"` is correctly specified.
- If `pl.scan_parquet(glob)` errors on an empty directory (no `.parquet` files): the `except Exception` block returns early with null columns — this is the intended behaviour.

- [ ] **Step 5: Run full test suite**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && .venv/bin/pytest tests/ -m "not integration" -q
```

Expected: `50 passed, 4 deselected`

- [ ] **Step 6: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/ai-infra-predictor" && git add processing/fundamental_features.py tests/test_fundamental_features.py && git commit -m "feat: fundamental features joiner — backward asof join on quarterly snapshots"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| OHLCV max history via `--bootstrap` flag | Task 2 |
| `lightgbm>=4.0` in core deps | Task 1 |
| Fundamental ingestion: 9 features, quarterly.parquet per ticker | Task 3 |
| Valuation ratios only on most recent quarter | Task 3 |
| `_safe_get` fallback row names for yfinance version differences | Task 3 |
| Label: 1-year forward return, look-ahead guard | Task 4 |
| `shift(-252).over("ticker")` — no cross-ticker leakage | Task 4 |
| Empty schema returned when no OHLCV data | Task 4 |
| Backward asof join — most recent period_end <= date | Task 5 |
| Null fundamentals when no data for a ticker | Task 5 |
| All 38 existing tests still pass after each task | Every task Step 5 |

**Placeholder scan:** No TBD, TODO, or vague steps found.

**Type consistency:**
- `fetch_fundamentals` returns `list[dict]` with `"period_end": datetime.date` → matches `SCHEMA pa.date32()` ✓
- `save_fundamentals` writes to `<output_dir>/financials/fundamentals/<TICKER>/quarterly.parquet` → matches `join_fundamentals` glob `<fundamentals_dir>/**/quarterly.parquet` ✓
- `build_labels` returns `pl.DataFrame` with `[ticker: String, date: Date, label_return_1y: Float64]` ✓
- `join_fundamentals` takes `price_df: pl.DataFrame` with `[ticker, date]` columns → matches output of `build_daily_feature_matrix` in `feature_engineering.py` ✓
- `_FUND_COLS` list in `fundamental_features.py` matches exactly the 9 fields in `SCHEMA` in `fundamental_ingestion.py` ✓
