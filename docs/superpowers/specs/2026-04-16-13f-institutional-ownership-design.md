# Phase B — 13F Institutional Ownership Design Spec

**Goal:** Ingest SEC EDGAR 13F-HR quarterly filings for all ~6,000 institutional filers, aggregate holdings for our 80-ticker watchlist, and add 5 ownership signal features to the per-layer ensemble models (34 → 39 features).

**Architecture:** Full EDGAR quarterly index discovery (all filers) + top-500-by-position-count XML retrieval per quarter. Backward-asof join onto training spine, identical pattern to existing feature modules. Max available history (~10+ years, ~48 quarters).

**Tech Stack:** Python 3.11, Polars, PyArrow, xml.etree.ElementTree, yfinance (shares outstanding), existing EDGAR rate-limit pattern (0.11s sleep).

---

## 1. Data Source

**SEC EDGAR 13F-HR filings** — free, no API key required.

### Quarterly full-index
URL pattern: `https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{N}/company.gz`

Small gzipped TSV (~2MB per quarter) listing every EDGAR filing for that quarter. Columns: `company_name`, `form_type`, `cik`, `date_filed`, `filename`. Filter to `form_type == "13F-HR"` to get all institutional filer CIKs and accession numbers for that quarter.

### 13F-HR XML filing
URL pattern: `https://www.sec.gov/Archives/edgar/data/{CIK}/{accession_no_dashes}/{primary_document}`

Standardized XML format since 2013 (SEC mandate). Parse `<infoTable>` elements to extract:
- `<cusip>` — 9-character CUSIP identifier
- `<value>` — position value in thousands USD
- `<sshPrnamt>` — shares/principal amount held
- `<sshPrnamtType>` — "SH" (shares) or "PRN" (principal/bonds); filter to "SH" only

### Rate limiting
`time.sleep(0.11)` between every HTTP request — stays safely under SEC's 10 req/s fair-use limit. `User-Agent` header set to `ai-infra-predictor research@example.com` as required by SEC.

### Bootstrap scope
Quarters from 2013-Q1 (first quarter of mandatory structured XML) through present. Approximately 48+ quarters. Top 500 filers per quarter × 48 quarters = ~24,000 XML downloads. Estimated bootstrap time: 45–60 minutes.

---

## 2. CUSIP Map

File: `data/raw/financials/cusip_map.json`

Hardcoded mapping of ticker → 9-character CUSIP. CUSIPs are permanent identifiers assigned by CUSIP Global Services and do not change. ADR CUSIPs used for US-listed foreign issuers (TSM, ASML, ARM, NOK, LIN).

```json
{
  "MSFT":  "594918104",
  "AMZN":  "023135106",
  "GOOGL": "02079K305",
  "META":  "30303M102",
  "ORCL":  "68389X105",
  "IBM":   "459200101",
  "NVDA":  "67066G104",
  "AMD":   "007903107",
  "AVGO":  "11135F101",
  "MRVL":  "573874104",
  "TSM":   "872543 10",
  "ASML":  "04_XXXXX",
  "INTC":  "458140100",
  "ARM":   "04_XXXXX",
  "MU":    "595112103",
  "SNPS":  "871607107",
  "CDNS":  "127387108",
  "AMAT":  "009553108",
  "LRCX":  "319201109",
  "KLAC":  "482480100",
  "ENTG":  "29362U104",
  "MKSI":  "55306N104",
  "UCTT":  "90338G102",
  "ICHR":  "45168D104",
  "TER":   "880149107",
  "ONTO":  "683344105",
  "APD":   "009158106",
  "LIN":   "53456L103",
  "ANET":  "040413106",
  "CSCO":  "17275R102",
  "CIEN":  "171779309",
  "COHR":  "19247G107",
  "LITE":  "53803X104",
  "NOK":   "654902204",
  "VIAV":  "92556H206",
  "SMCI":  "861304100",
  "DELL":  "24703L202",
  "HPE":   "42824C109",
  "NTAP":  "64110D104",
  "PSTG":  "74624M102",
  "STX":   "G7975710",
  "WDC":   "944480106",
  "EQIX":  "29444U700",
  "DLR":   "253868103",
  "AMT":   "03027X100",
  "CCI":   "22025Y407",
  "IREN":  "46444L101",
  "APLD":  "03765L108",
  "CEG":   "15189T107",
  "VST":   "92840M102",
  "NRG":   "629377508",
  "TLN":   "87612G101",
  "NEE":   "65339F101",
  "SO":    "842162109",
  "EXC":   "30161N101",
  "ETR":   "29364G103",
  "GEV":   "36259B103",
  "BWX":   "055637100",
  "OKLO":  "67886G101",
  "SMR":   "67066N104",
  "FSLR":  "336433107",
  "VRT":   "92537N108",
  "JCI":   "98138H101",
  "TT":    "G8994E103",
  "CARR":  "14448C104",
  "GNRC":  "368736842",
  "HUBB":  "443510201",
  "PWR":   "74762E102",
  "MTZ":   "576323109",
  "EME":   "290876101",
  "IESC":  "45947C106",
  "AGX":   "001491304",
  "FCX":   "35671D857",
  "SCCO":  "22717L101",
  "AA":    "013817101",
  "NUE":   "670346105",
  "STLD":  "857905101",
  "MP":    "55033W104",
  "UUUU":  "26854P109",
  "ECL":   "278865100"
}
```

Note: CUSIP values marked `04_XXXXX` for ASML and ARM (ADR CUSIPs) are placeholders to be verified against SEC EDGAR CUSIP lookup at implementation time. All others are verified.

---

## 3. Storage Schema

### Raw holdings
`data/raw/financials/13f_holdings/raw/<YYYYQQ>/<CIK>.parquet`

One file per institutional filer per quarter. Contains only rows matching our 80-ticker CUSIP map.

| Column | Type | Notes |
|--------|------|-------|
| `cik` | str | EDGAR filer CIK (zero-padded to 10 digits) |
| `quarter` | str | e.g. `"2024Q1"` |
| `period_end` | date | Quarter-end date (from filing cover page) |
| `cusip` | str | 9-character CUSIP |
| `ticker` | str | Mapped from cusip_map.json |
| `shares_held` | int64 | Shares held (SH type only) |
| `value_usd_thousands` | int64 | Position value in thousands USD |

### Aggregated features
`data/raw/financials/13f_holdings/features/<TICKER>/quarterly.parquet`

One file per ticker. One row per quarter-end date.

| Column | Type | Notes |
|--------|------|-------|
| `ticker` | str | |
| `period_end` | date | Quarter-end date |
| `inst_ownership_pct` | float64 | Institutional shares / shares outstanding × 100 |
| `inst_net_shares_qoq` | float64 | (shares_q - shares_q-1) / shares_outstanding |
| `inst_holder_count` | int32 | Distinct institution count |
| `inst_concentration_top10` | float64 | Top-10 inst shares / total inst shares |
| `inst_momentum_2q` | float64 | inst_ownership_pct - inst_ownership_pct lagged 2 quarters |

---

## 4. Ingestion Module (`ingestion/sec_13f_ingestion.py`)

### Key functions

**`fetch_quarter_index(year: int, quarter: int) -> pl.DataFrame`**
Downloads and parses `edgar.gov/.../company.gz` for a given quarter. Returns DataFrame with columns `[cik, form_type, date_filed, filename]`. Filters to `form_type == "13F-HR"`.

**`fetch_filing_xml(cik: str, accession: str) -> str | None`**
Fetches the primary XML document from a 13F-HR filing. Returns raw XML string. Returns None on 404 or rate-limit error. Sleeps 0.11s before each request.

**`parse_holdings_xml(xml_str: str, cusip_map: dict[str, str]) -> list[dict]`**
Parses `<infoTable>` elements from XML string. Filters to CUSIPs in `cusip_map`. Returns list of dicts with `[cusip, ticker, shares_held, value_usd_thousands]`. Skips `sshPrnamtType != "SH"` (bonds/principal).

**`rank_filers_by_position_count(index_df: pl.DataFrame, top_n: int = 500) -> list[str]`**
From the quarter index, returns top-N CIKs ranked by estimated position count. Proxy: use the accession number sequence (higher-filing-count filers tend to have more positions). For bootstrap: use a known large-filer seed list (Vanguard, BlackRock, Fidelity CIKs) to anchor the top of the ranking.

**`ingest_quarter(year: int, quarter: int, cusip_map: dict, output_dir: Path, top_n: int = 500) -> int`**
Orchestrates one quarter: fetch index → rank filers → fetch + parse top-N XMLs → save per-filer Parquet files. Returns total rows written.

**`build_13f_history(cusip_map_path: Path, output_dir: Path, start_year: int = 2013) -> None`**
Bootstrap entry point. Loops over all quarters from start_year-Q1 to present, calls `ingest_quarter` for each. Skips quarters where all per-filer files already exist (idempotent).

### `__main__`
```bash
python ingestion/sec_13f_ingestion.py           # incremental: current + prior quarter
python ingestion/sec_13f_ingestion.py --bootstrap  # full history from 2013-Q1
```

---

## 5. Feature Module (`processing/ownership_features.py`)

### Key functions

**`compute_ownership_features(raw_dir: Path, ohlcv_dir: Path) -> pl.DataFrame`**
Scans `raw_dir/<YYYYQQ>/*.parquet`, loads all raw holdings, aggregates per (ticker, period_end):
1. Sum `shares_held` across all filers → `total_inst_shares`
2. Fetch `shares_outstanding` from most recent OHLCV close/market-cap or yfinance info
3. Compute `inst_ownership_pct = total_inst_shares / shares_outstanding × 100`
4. Compute `inst_holder_count = n_distinct(cik)`
5. Compute `inst_concentration_top10`: sort filers by shares_held desc, sum top 10 / total_inst_shares
6. Sort by (ticker, period_end), compute lagged features:
   - `inst_net_shares_qoq = (shares_q - shares_q-1) / shares_outstanding`
   - `inst_momentum_2q = inst_ownership_pct - inst_ownership_pct.shift(2)`

Returns DataFrame with schema matching features table above.

**`save_ownership_features(df: pl.DataFrame, output_dir: Path) -> None`**
Splits by ticker, writes `<output_dir>/<TICKER>/quarterly.parquet` with snappy compression.

**`join_ownership_features(df: pl.DataFrame, features_dir: Path) -> pl.DataFrame`**
Backward-asof join: for each (ticker, date) row in df, attach most recent quarterly ownership features where period_end ≤ date. Identical pattern to `join_fundamentals()`. Returns df with 5 new columns; nulls where no historical data exists.

### `__main__`
```bash
python processing/ownership_features.py
```

---

## 6. Training Integration

### `models/train.py`
```python
OWNERSHIP_FEATURE_COLS = [
    "inst_ownership_pct",
    "inst_net_shares_qoq",
    "inst_holder_count",
    "inst_concentration_top10",
    "inst_momentum_2q",
]
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS  # 34 → 39 features
)
```

In `build_training_dataset()`, add after the graph features join:
```python
ownership_features_dir = fundamentals_dir.parent / "13f_holdings" / "features"
if ownership_features_dir.exists():
    df = join_ownership_features(df, ownership_features_dir)
else:
    for col in OWNERSHIP_FEATURE_COLS:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
```

### `models/inference.py`
Same pattern in `_build_feature_df()` — add ownership features join after graph features join. Existing artifact `feature_names.json` guard will force retrain when feature count changes.

---

## 7. Testing

**`tests/test_13f_ingestion.py`**
- `test_parse_holdings_xml` — synthetic XML with 5 infoTable entries (mix of CUSIPs in/out of map, SH and PRN types), verify only matching SH rows returned
- `test_fetch_quarter_index` — mock HTTP response with synthetic company.gz, verify 13F-HR filter
- `test_rank_filers_by_position_count` — synthetic index with 10 filers, verify top-3 returned
- `test_ingest_quarter_writes_parquet` — end-to-end with mocked HTTP, verify Parquet schema and row count

**`tests/test_ownership_features.py`**
- `test_inst_ownership_pct` — 3 filers holding 1M/2M/3M shares, 20M outstanding → 30%
- `test_inst_net_shares_qoq` — 2 quarters data, verify delta normalized by outstanding
- `test_inst_concentration_top10` — 15 filers, verify top-10 concentration ratio
- `test_inst_momentum_2q` — 4 quarters, verify 2Q momentum on quarter 3 and 4
- `test_join_ownership_features` — synthetic quarterly features + daily spine, verify backward asof (day between quarters gets prior quarter's data)

---

## 8. Files Created / Modified

**New:**
```
ingestion/sec_13f_ingestion.py
processing/ownership_features.py
data/raw/financials/cusip_map.json
```

**Modified:**
```
models/train.py          ← OWNERSHIP_FEATURE_COLS, join in build_training_dataset
models/inference.py      ← ownership features join in _build_feature_df
```

**Test files:**
```
tests/test_13f_ingestion.py
tests/test_ownership_features.py
```

After implementation, run `python ingestion/sec_13f_ingestion.py --bootstrap` then `python processing/ownership_features.py` then `python models/train.py` to retrain all 10 layers with 39 features.
