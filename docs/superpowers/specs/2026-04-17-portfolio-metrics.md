# Phase C-C — Portfolio Metrics Post-Processor

**Goal:** Add a post-processing step after inference that enriches predictions with liquidity data, model agreement scores, and peer correlation warnings — making the output actionable rather than just ranked.

**Architecture:** New standalone module `processing/portfolio_metrics.py`. Reads raw predictions parquet, adds 4 new columns, writes enriched parquet alongside original. Called from `models/inference.py` automatically after prediction.

**Tech Stack:** Python 3.11, Polars, yfinance (already a dep), NumPy

---

## 1. New Columns Added

| Column | Type | Description |
|--------|------|-------------|
| `market_cap_b` | float64 | Market cap in billions USD (from yfinance, cached daily) |
| `is_liquid` | bool | True if market_cap_b ≥ 1.0 |
| `model_agreement` | float64 | Fraction of sub-models agreeing with ensemble direction. 1.0 = all three point same way; 0.33 = only one of three agrees |
| `peer_correlation_90d` | float64 | Average pairwise 90-day return correlation of this ticker with the other top-9 tickers by rank. High value (>0.7) means the pick is redundant with others already in the basket |

---

## 2. Market Cap Cache

**Path:** `data/raw/financials/market_caps.parquet`

Schema: `[ticker (str), date (date), market_cap_b (float64)]`

Refreshed daily. On each call to `portfolio_metrics.enrich()`, if today's market cap isn't cached, fetch from yfinance for all 80 tickers in a single batch, append to parquet.

```python
def _get_market_caps(tickers: list[str], as_of: date) -> dict[str, float]:
    """Returns {ticker: market_cap_billions}. Uses cache; fetches missing."""
    cache_path = _DATA_DIR / "raw" / "financials" / "market_caps.parquet"
    if cache_path.exists():
        cached = pl.read_parquet(cache_path).filter(pl.col("date") == as_of)
        if len(cached) == len(tickers):
            return dict(zip(cached["ticker"], cached["market_cap_b"]))

    caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            caps[ticker] = round(info.market_cap / 1e9, 3)
            time.sleep(0.1)
        except Exception:
            caps[ticker] = None

    # Append to cache
    new_rows = pl.DataFrame({
        "ticker": list(caps.keys()),
        "date": [as_of] * len(caps),
        "market_cap_b": list(caps.values()),
    })
    if cache_path.exists():
        existing = pl.read_parquet(cache_path).filter(pl.col("date") != as_of)
        pl.concat([existing, new_rows]).write_parquet(cache_path, compression="snappy")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        new_rows.write_parquet(cache_path, compression="snappy")

    return caps
```

---

## 3. Model Agreement

Uses the existing sub-model return columns already in predictions.parquet (`lgbm_return`, `rf_return`, `ridge_return`):

```python
def _model_agreement(df: pl.DataFrame) -> pl.Series:
    """Fraction of sub-models whose sign matches the ensemble sign."""
    ensemble_sign = pl.Series(np.sign(df["expected_annual_return"].to_numpy()))
    agreements = []
    for col in ["lgbm_return", "rf_return", "ridge_return"]:
        sub_sign = pl.Series(np.sign(df[col].to_numpy()))
        agreements.append((sub_sign == ensemble_sign).cast(pl.Float64))
    # Mean agreement across 3 models
    return (agreements[0] + agreements[1] + agreements[2]) / 3.0
```

---

## 4. Peer Correlation

Uses the OHLCV data already in `data/raw/financials/ohlcv/`:

```python
def _peer_correlation(
    df: pl.DataFrame,
    ohlcv_dir: Path,
    as_of: date,
    top_n: int = 10,
) -> pl.Series:
    """
    For each ticker in df, compute its average pairwise 90-day return correlation
    with the other top-(top_n-1) tickers by rank.
    Returns null for tickers ranked > top_n.
    """
    top_tickers = df.sort("rank").head(top_n)["ticker"].to_list()
    window_start = as_of - timedelta(days=90)

    # Load 90d returns for top-N tickers
    prices = {}
    for ticker in top_tickers:
        parquets = list(ohlcv_dir.glob(f"{ticker}/*.parquet"))
        if not parquets:
            continue
        p = (
            pl.scan_parquet([str(x) for x in parquets])
            .filter((pl.col("date") >= window_start) & (pl.col("date") <= as_of))
            .select(["date", "close_price"])
            .collect()
            .sort("date")
        )
        if len(p) > 5:
            prices[ticker] = p["close_price"].pct_change().drop_nulls().to_numpy()

    # Build correlation matrix
    corr_map = {}
    for ticker in top_tickers:
        if ticker not in prices:
            corr_map[ticker] = None
            continue
        peers = [t for t in top_tickers if t != ticker and t in prices]
        if not peers:
            corr_map[ticker] = 0.0
            continue
        peer_corrs = []
        for peer in peers:
            min_len = min(len(prices[ticker]), len(prices[peer]))
            if min_len < 10:
                continue
            c = float(np.corrcoef(prices[ticker][-min_len:], prices[peer][-min_len:])[0, 1])
            peer_corrs.append(c)
        corr_map[ticker] = round(float(np.mean(peer_corrs)), 4) if peer_corrs else None

    return pl.Series([corr_map.get(t) for t in df["ticker"].to_list()])
```

---

## 5. Main Entry Point

```python
def enrich(date_str: str, predictions_dir: Path | None = None) -> pl.DataFrame:
    """
    Enrich predictions for a given date with liquidity and portfolio metrics.
    Writes predictions_enriched.parquet alongside predictions.parquet.
    Returns enriched DataFrame.
    """
    if predictions_dir is None:
        predictions_dir = Path("data/predictions")

    pred_path = predictions_dir / f"date={date_str}" / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions found at {pred_path}")

    df = pl.read_parquet(pred_path)
    as_of = date.fromisoformat(date_str)

    # Market cap + liquidity
    caps = _get_market_caps(df["ticker"].to_list(), as_of)
    df = df.with_columns([
        pl.col("ticker").map_elements(lambda t: caps.get(t), return_dtype=pl.Float64).alias("market_cap_b"),
    ])
    df = df.with_columns(
        (pl.col("market_cap_b") >= 1.0).alias("is_liquid")
    )

    # Model agreement
    df = df.with_columns(_model_agreement(df).alias("model_agreement"))

    # Peer correlation
    ohlcv_dir = Path("data/raw/financials/ohlcv")
    df = df.with_columns(
        _peer_correlation(df, ohlcv_dir, as_of).alias("peer_correlation_90d")
    )

    # Write enriched output
    out_path = predictions_dir / f"date={date_str}" / "predictions_enriched.parquet"
    df.write_parquet(out_path, compression="snappy")

    # Print actionable summary
    liquid = df.filter(pl.col("is_liquid")).sort("rank")
    _LOG.info("[PortfolioMetrics] %d liquid picks (≥$1B market cap)", len(liquid))
    print(liquid.select(["rank", "ticker", "layer", "expected_annual_return",
                         "market_cap_b", "model_agreement", "peer_correlation_90d"]))

    return df


if __name__ == "__main__":
    import sys
    date_str = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    enrich(date_str)
```

---

## 6. Wire Into Inference

In `models/inference.py`, at the end of `run_inference()`:

```python
# After writing predictions.parquet
try:
    from processing.portfolio_metrics import enrich
    enrich(date_str)
except Exception as exc:
    _LOG.warning("Portfolio metrics enrichment failed (non-fatal): %s", exc)
```

The try/except ensures inference never breaks if enrichment fails (e.g. no internet for yfinance).

---

## 7. CLI

```bash
python processing/portfolio_metrics.py 2026-04-15   # enrich specific date
python processing/portfolio_metrics.py              # enrich today
```

---

## 8. Testing

**`tests/test_portfolio_metrics.py`** (new):

- `test_model_agreement_all_agree` — when lgbm/rf/ridge all positive and ensemble positive → agreement = 1.0
- `test_model_agreement_one_disagrees` — two positive, one negative → agreement = 0.667
- `test_is_liquid_threshold` — ticker with market_cap_b=0.5 → is_liquid=False; 1.5 → True
- `test_enrich_writes_enriched_parquet` — mock predictions parquet + mock market cap fetch → verify enriched parquet written with correct columns
- `test_peer_correlation_returns_null_for_no_data` — ticker with no OHLCV data → peer_correlation_90d is null

---

## 9. Output Example

```
Liquid picks for 2026-04-15 (market cap ≥ $1B):

rank ticker layer            exp_return  mkt_cap_b  agreement  peer_corr
   3    TSM  compute          2.14       650.2      1.00       0.31
   6    WDC  servers          1.46        16.8      0.67       0.44
   7   CIEN  networking       1.43         7.2      1.00       0.38
   8   LRCX  semi_equipment   1.42        94.3      1.00       0.62  ⚠
  10   AVGO  compute          1.10       820.1      1.00       0.55
  11   NVDA  compute          0.94      2800.0      0.67       0.71  ⚠
...

⚠ = peer_correlation_90d > 0.65 (consider position sizing)
Note: APLD ($0.3B), IREN ($0.4B) excluded — below $1B liquidity threshold
```

---

## 10. Files Created / Modified

**New:**
```
processing/portfolio_metrics.py
tests/test_portfolio_metrics.py
data/raw/financials/market_caps.parquet   ← generated at runtime
data/predictions/date=*/predictions_enriched.parquet  ← generated at runtime
```

**Modified:**
```
models/inference.py   ← call enrich() at end of run_inference()
```
