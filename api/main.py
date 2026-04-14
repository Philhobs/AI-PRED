"""
FastAPI serving layer for AI Infrastructure stock predictions.

Endpoints:
  GET /health                  — liveness + last prediction date
  GET /predictions/latest      — all rows from most recent date=* dir, sorted by rank
  GET /predictions/{ticker}    — history for one ticker across all date=* dirs
  GET /features/{ticker}       — OHLCV rows for one ticker
"""
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException

# ── Module-level constants (patchable by tests) ────────────────────────────────
PREDICTIONS_DIR = Path("data/predictions")
OHLCV_DIR = Path("data/raw/financials/ohlcv")

# ── Watchlist ──────────────────────────────────────────────────────────────────
TICKERS = [
    "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AMD", "AVGO", "MRVL", "TSM",
    "ASML", "AMAT", "LRCX", "KLAC",
    "VRT", "SMCI", "DELL", "HPE",
    "EQIX", "DLR", "AMT",
    "CEG", "VST", "NRG", "TLN",
]

_NO_PREDICTIONS_MSG = (
    "No predictions available yet. "
    "Run python models/inference.py to generate."
)

app = FastAPI(title="AI Infra Predictor", version="1.0.0")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _date_dirs() -> list[Path]:
    """Return sorted list of date=* subdirectories from PREDICTIONS_DIR."""
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(
        d for d in PREDICTIONS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("date=")
    )


def _latest_predictions() -> pl.DataFrame:
    """Read predictions from the most recent date=* directory."""
    dirs = _date_dirs()
    if not dirs:
        raise HTTPException(status_code=503, detail=_NO_PREDICTIONS_MSG)
    latest_dir = dirs[-1]
    parquet_path = latest_dir / "predictions.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=503, detail=_NO_PREDICTIONS_MSG)
    return pl.read_parquet(str(parquet_path))


def _all_predictions() -> pl.DataFrame:
    """Read and concatenate predictions across all date=* directories."""
    dirs = _date_dirs()
    if not dirs:
        raise HTTPException(status_code=503, detail=_NO_PREDICTIONS_MSG)
    frames = []
    for d in dirs:
        p = d / "predictions.parquet"
        if p.exists():
            frames.append(pl.read_parquet(str(p)))
    if not frames:
        raise HTTPException(status_code=503, detail=_NO_PREDICTIONS_MSG)
    return pl.concat(frames)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Return service health and the most recent prediction date."""
    df = _latest_predictions()
    last_date = _date_dirs()[-1].name[len("date="):]
    return {
        "status": "ok",
        "last_prediction_date": last_date,
        "ticker_count": len(df),
    }


# NOTE: must be registered before /predictions/{ticker} — FastAPI resolves static paths first
@app.get("/predictions/latest")
def predictions_latest():
    """Return all rows from the most recent prediction run, sorted by rank."""
    df = _latest_predictions()
    return df.sort("rank").to_dicts()


@app.get("/predictions/{ticker}")
def predictions_ticker(ticker: str):
    """Return prediction history for a single ticker across all dates."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {ticker} not in watchlist",
        )
    df = _all_predictions()
    result = df.filter(pl.col("ticker") == ticker).sort("as_of_date")
    return result.to_dicts()


@app.get("/features/{ticker}")
def features_ticker(ticker: str):
    """Return OHLCV feature rows for a single ticker, sorted by date."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {ticker} not in watchlist",
        )
    ticker_dir = OHLCV_DIR / ticker
    parquet_files = list(ticker_dir.glob("*.parquet")) if ticker_dir.exists() else []
    if not parquet_files:
        raise HTTPException(
            status_code=503,
            detail=f"No OHLCV data for {ticker}. Run ingestion first.",
        )
    df = pl.concat([pl.read_parquet(str(f)) for f in parquet_files])
    return df.sort("date").to_dicts()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
