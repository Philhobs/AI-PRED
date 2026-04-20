# Sub-project A — Multi-Exchange Ticker Registry + FX Pipeline

**Goal:** Expand the model from 83 US-only tickers to 127 tickers across 11 layers (10 existing + new `robotics` layer), adding European and Asian market champions. Introduce a typed `TickerInfo` registry and a first-class FX pipeline that produces both local-currency and USD-normalized returns. FEATURE_COLS grows from 47 → 48.

**Architecture:** One registry refactor (`ticker_registry.py` → `TickerInfo` dataclass), two new modules (`ingestion/fx_ingestion.py`, `processing/fx_features.py`), and small extensions to `models/train.py`, `models/inference.py`, and `processing/supply_chain_features.py`.

---

## Context: What Already Exists

- `ingestion/ticker_registry.py` — flat `TICKER_LAYERS: dict[str, str]` mapping 83 US tickers to 10 layers. `tickers_in_layer()`, `layers()`, `LAYER_IDS`, `HYPERSCALERS` are all used across the pipeline.
- `ingestion/ohlcv_ingestion.py` — fetches via yfinance, saves to `data/raw/financials/ohlcv/{ticker}/{year}.parquet`. Already works for international ticker symbols (e.g., `6954.T`, `ABBN.SW`) — no changes needed.
- `processing/supply_chain_features.py` — builds wide close matrix from OHLCV, computes peer momentum and correlation. Uses `_LAYER_MAP` and `_ALL_TICKERS` derived from the registry at module load.
- `models/train.py` / `models/inference.py` — import from `ticker_registry` and call all join functions.

---

## 1. Registry Refactor (`ingestion/ticker_registry.py`)

### 1a. TickerInfo dataclass

Replace the flat `TICKER_LAYERS` dict with a typed list of `TickerInfo` entries:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TickerInfo:
    symbol: str    # yfinance-compatible: "NVDA", "ABBN.SW", "6954.T"
    layer: str     # layer name (one of LAYER_IDS keys)
    exchange: str  # "US", "DE", "PA", "SW", "MI", "CO", "ST", "OL", "L", "AS", "BR", "MC", "T"
    currency: str  # ISO 4217: "USD", "EUR", "CHF", "JPY", "DKK", "SEK", "NOK", "GBP"
    country: str   # ISO 3166-1 alpha-2: "US", "DE", "FR", "CH", "IT", "DK", "SE", "NO", "GB", "JP"
```

### 1b. TICKERS_INFO list (128 entries)

All 83 existing tickers carry `exchange="US", currency="USD", country="US"`. New entries below.

**cloud** (+3, total 9):
- `SAP.DE` — SAP SE, DE, EUR, DE
- `CAP.PA` — Capgemini, PA, EUR, FR
- `OVH.PA` — OVHcloud, PA, EUR, FR

**compute** (+2, total 13):
- `IFX.DE` — Infineon Technologies, DE, EUR, DE
- `STM` — STMicroelectronics (NYSE-listed), US, USD, CH

**semi_equipment** (+4, total 15):
- `8035.T` — Tokyo Electron, T, JPY, JP
- `6920.T` — Lasertec, T, JPY, JP
- `ASMI.AS` — ASM International, AS, EUR, NL
- `BESI.AS` — BE Semiconductor Industries, AS, EUR, NL

**networking** (+3, total 11):
- `ERIC` — Ericsson (NASDAQ ADR), US, USD, SE
- `JNPR` — Juniper Networks, US, USD, US
- `SPT.L` — Spirent Communications, L, GBP, GB

**servers** (+2, total 9):
- `6702.T` — Fujitsu, T, JPY, JP
- `KTN.DE` — Kontron, DE, EUR, DE

**datacenter** (+2, total 8):
- `9432.T` — NTT Corp, T, JPY, JP
- `CLNX.MC` — Cellnex Telecom, MC, EUR, ES

**power** (+6, total 19):
- `ENR.DE` — Siemens Energy, DE, EUR, DE
- `VWS.CO` — Vestas Wind Systems, CO, DKK, DK
- `RWE.DE` — RWE AG, DE, EUR, DE
- `ENEL.MI` — Enel SpA, MI, EUR, IT
- `ORSTED.CO` — Ørsted, CO, DKK, DK
- `ENGI.PA` — Engie, PA, EUR, FR

**cooling** (+3, total 10):
- `ALFA.ST` — Alfa Laval, ST, SEK, SE
- `ASETEK.OL` — Asetek, OL, NOK, NO
- `SU.PA` — Schneider Electric, PA, EUR, FR

**grid** (+4, total 10):
- `PRY.MI` — Prysmian, MI, EUR, IT
- `NEX.PA` — Nexans, PA, EUR, FR
- `NG.L` — National Grid, L, GBP, GB
- `TRN.MI` — Terna, MI, EUR, IT

**metals** (+4, total 12):
- `UMI.BR` — Umicore, BR, EUR, BE
- `GLEN.L` — Glencore, L, GBP, GB
- `RIO.L` — Rio Tinto, L, GBP, GB
- `5713.T` — Sumitomo Metal Mining, T, JPY, JP

**robotics** (new layer, 11 tickers):
- `ISRG` — Intuitive Surgical, US, USD, US
- `ROK` — Rockwell Automation, US, USD, US
- `ZBRA` — Zebra Technologies, US, USD, US
- `CGNX` — Cognex, US, USD, US
- `SYM` — Symbotic, US, USD, US
- `ABBN.SW` — ABB, SW, CHF, CH
- `KGX.DE` — KION Group, DE, EUR, DE
- `HEXA-B.ST` — Hexagon, ST, SEK, SE
- `6954.T` — FANUC, T, JPY, JP
- `6506.T` — YASKAWA Electric, T, JPY, JP
- `6861.T` — Keyence, T, JPY, JP

### 1c. Generated backwards-compatible lookups

All existing call sites remain unchanged:

```python
TICKERS_INFO: list[TickerInfo] = [...]  # 128 entries

# Backwards-compatible — regenerated from TICKERS_INFO
TICKER_LAYERS: dict[str, str]  = {t.symbol: t.layer     for t in TICKERS_INFO}
TICKERS:       list[str]        = sorted(t.symbol for t in TICKERS_INFO)
HYPERSCALERS:  frozenset[str]   = frozenset({"MSFT", "AMZN", "GOOGL", "META"})

# New lookups
TICKER_CURRENCY: dict[str, str] = {t.symbol: t.currency for t in TICKERS_INFO}
TICKER_EXCHANGE: dict[str, str] = {t.symbol: t.exchange for t in TICKERS_INFO}
TICKER_COUNTRY:  dict[str, str] = {t.symbol: t.country  for t in TICKERS_INFO}

def non_usd_tickers() -> list[str]:
    return sorted(t.symbol for t in TICKERS_INFO if t.currency != "USD")

def ticker_currency(symbol: str) -> str:
    return TICKER_CURRENCY[symbol]
```

`tickers_in_layer()` and `layers()` work identically — `layers()` derives from `LAYER_IDS` keys as before. `LAYER_IDS` gains `"robotics": 11`.

---

## 2. FX Ingestion (`ingestion/fx_ingestion.py`)

Fetches 7 daily FX rate series from yfinance. Saves to `data/raw/financials/fx/{pair}.parquet`.

**Schema:** `date` (pl.Date), `rate` (pl.Float64). One file per pair.

**Pairs and yfinance symbols:**

| Pair | yfinance symbol | Covers currencies |
|---|---|---|
| EURUSD | EURUSD=X | EUR (DE, PA, MI, AS, BR, MC) |
| CHFUSD | CHFUSD=X | CHF (SW) |
| JPYUSD | JPYUSD=X | JPY (T) |
| DKKUSD | DKKUSD=X | DKK (CO) |
| SEKUSD | SEKUSD=X | SEK (ST) |
| NOKUSD | NOKUSD=X | NOK (OL) |
| GBPUSD | GBPUSD=X | GBP (L) |

**Interface:**

```python
def fetch_fx_rates(pairs: list[str] | None = None, years: int = 5) -> dict[str, pl.DataFrame]:
    """Fetch daily FX rates for all 7 pairs. Returns {pair: DataFrame}."""

def save_fx_rates(fx_dir: Path, pairs: dict[str, pl.DataFrame]) -> None:
    """Append-and-deduplicate into data/raw/financials/fx/{pair}.parquet."""
```

Follows same failure handling as `eia_ingestion.py`: HTTP errors are caught, logged as warnings, and return empty DataFrames (never crash the pipeline).

---

## 3. FX Features (`processing/fx_features.py`)

### 3a. USD close matrix

```python
def build_usd_close_matrix(
    close_wide: pl.DataFrame,
    fx_dir: Path,
) -> pl.DataFrame:
    """
    Convert non-USD ticker columns to USD using daily FX rates.
    - USD tickers: pass through unchanged.
    - Non-USD tickers: close_price * rate_on_date (left join on date; missing rate → null).
    Returns same shape as close_wide.
    """
```

Currency-to-pair mapping is derived from `TICKER_CURRENCY` in the registry.

### 3b. FX-adjusted 20d return feature

```python
def join_fx_features(
    df: pl.DataFrame,
    fx_dir: Path | None = None,
    ohlcv_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Add fx_adjusted_return_20d: 20-day cumulative return on USD-normalized close prices.
    - Non-USD tickers: populated Float64.
    - USD tickers: null (their price features already capture this).

    Returns df with 1 new Float64 column.
    """
```

### 3c. Effect on supply_chain_features.py

`join_supply_chain_features()` gains an optional `fx_dir: Path | None = None` parameter. When provided, `compute_supply_chain_correlation()` calls `build_usd_close_matrix(close_wide, fx_dir)` to build its 60-day returns window — preventing spurious correlations from shared EUR or JPY exposure between tickers in the same currency zone. When `fx_dir=None` (e.g., missing data), falls back silently to local-currency returns (current behavior, fully backwards-compatible).

`compute_layer_momentum()` always uses raw local-currency close prices — correct for measuring each ticker's performance in its home market context. No change needed there.

Updated signature:
```python
def join_supply_chain_features(
    df: pl.DataFrame,
    ohlcv_dir: Path | None = None,
    earnings_dir: Path | None = None,
    fx_dir: Path | None = None,   # new — enables USD correlation matrix
) -> pl.DataFrame:
```

---

## 4. Model Integration (`models/train.py` + `models/inference.py`)

Add to both files:

```python
FX_FEATURE_COLS = ["fx_adjusted_return_20d"]
```

Append to `FEATURE_COLS`:
```python
FEATURE_COLS = (
    ... + SUPPLY_CHAIN_FEATURE_COLS + FX_FEATURE_COLS  # 47 → 48 features total
)
```

Add import and call in both `build_training_dataset()` and `_build_feature_df()`:
```python
from processing.fx_features import join_fx_features
df = join_fx_features(df, ohlcv_dir=ohlcv_dir)
```

Fallback (when `fx_dir` doesn't exist):
```python
df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("fx_adjusted_return_20d"))
```

---

## 5. Testing

### `tests/test_ticker_registry.py` (3 new tests)

- **`test_tickerinfo_fields_complete`** — every `TickerInfo` entry has non-empty symbol, layer, exchange, currency, country; no duplicate symbols
- **`test_robotics_layer_in_layer_ids`** — `tickers_in_layer("robotics")` returns 11 tickers without raising
- **`test_127_tickers_total`** — `len(TICKERS) == 127`; backwards-compatible `TICKER_LAYERS` has 127 entries

### `tests/test_fx_features.py` (7 tests)

- **`test_usd_ticker_passthrough`** — USD ticker close prices unchanged after `build_usd_close_matrix`
- **`test_eur_ticker_converted_to_usd`** — EUR ticker × EURUSD rate = expected USD close (within 0.01%)
- **`test_missing_fx_rate_produces_null`** — date with no FX data → null for that ticker, no crash
- **`test_fx_adjusted_return_20d_correct`** — synthetic prices + known FX rates → correct 20d USD return
- **`test_usd_ticker_gets_null_feature`** — `fx_adjusted_return_20d` is null for USD tickers
- **`test_join_fx_features_adds_column`** — public join adds exactly 1 Float64 column named `fx_adjusted_return_20d`
- **`test_registry_coverage`** — every non-USD currency in `TICKERS_INFO` has a corresponding FX pair file supported by `fx_ingestion.py`

---

## 6. Null Handling

| Feature | Null condition |
|---|---|
| `fx_adjusted_return_20d` | USD tickers (always null); non-USD tickers with missing FX rate on a date |

LightGBM handles nulls natively — no imputation needed (consistent with existing pipeline).

---

## 7. File Map

| File | Action |
|---|---|
| `ingestion/ticker_registry.py` | Refactor (TickerInfo dataclass, 83 → 128 tickers) |
| `ingestion/fx_ingestion.py` | Create |
| `processing/fx_features.py` | Create |
| `processing/supply_chain_features.py` | Extend (`join_supply_chain_features` gains `fx_dir` param; `compute_supply_chain_correlation` uses USD matrix when available) |
| `models/train.py` | Extend (FX_FEATURE_COLS, 47 → 48) |
| `models/inference.py` | Extend (mirror train.py) |
| `tests/test_ticker_registry.py` | Extend (3 new tests) |
| `tests/test_fx_features.py` | Create (7 tests) |
