"""
Streamlit dashboard for AI Infrastructure Predictor.

Two pages:
  1. Top Picks — ranked table + top-5 metric cards
  2. Stock Drill-Down — price chart, prediction history, fundamentals table
"""
import datetime
import streamlit as st
import plotly.graph_objects as go
import polars as pl
from pathlib import Path

# ── Module-level constants ─────────────────────────────────────────────────────

PREDICTIONS_DIR = Path("data/predictions")
OHLCV_DIR = Path("data/raw/financials/ohlcv")
FUNDAMENTALS_DIR = Path("data/raw/financials/fundamentals")
LGBM_ARTIFACTS_DIR = Path("models/artifacts")

TICKERS = [
    "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AMD", "AVGO", "MRVL", "TSM",
    "ASML", "AMAT", "LRCX", "KLAC",
    "VRT", "SMCI", "DELL", "HPE",
    "EQIX", "DLR", "AMT",
    "CEG", "VST", "NRG", "TLN",
]

SECTOR = {
    "MSFT": "Hyperscaler", "AMZN": "Hyperscaler",
    "GOOGL": "Hyperscaler", "META": "Hyperscaler",
    "NVDA": "AI Chips", "AMD": "AI Chips",
    "AVGO": "AI Chips", "MRVL": "AI Chips", "TSM": "AI Chips",
    "ASML": "Foundry Equipment", "AMAT": "Foundry Equipment",
    "LRCX": "Foundry Equipment", "KLAC": "Foundry Equipment",
    "VRT": "AI Infrastructure", "SMCI": "AI Infrastructure",
    "DELL": "AI Infrastructure", "HPE": "AI Infrastructure",
    "EQIX": "Data Center REIT", "DLR": "Data Center REIT", "AMT": "Data Center REIT",
    "CEG": "Power", "VST": "Power", "NRG": "Power", "TLN": "Power",
}

# ── Data loading functions (testable — no Streamlit calls) ────────────────────


def load_latest_predictions() -> pl.DataFrame:
    """Load most recent predictions Parquet, sorted by rank. Returns empty DataFrame if none."""
    date_dirs = sorted(PREDICTIONS_DIR.glob("date=*"))
    if not date_dirs:
        return pl.DataFrame()
    parquet_path = date_dirs[-1] / "predictions.parquet"
    if not parquet_path.exists():
        return pl.DataFrame()
    return pl.read_parquet(str(parquet_path)).sort("rank")


def load_ticker_predictions(ticker: str) -> pl.DataFrame:
    """Load all historical predictions for a single ticker, sorted by as_of_date."""
    date_dirs = sorted(PREDICTIONS_DIR.glob("date=*"))
    if not date_dirs:
        return pl.DataFrame()
    frames = []
    for d in date_dirs:
        try:
            df = pl.read_parquet(str(d / "predictions.parquet"))
            filtered = df.filter(pl.col("ticker") == ticker)
            if not filtered.is_empty():
                frames.append(filtered)
        except (FileNotFoundError, OSError):
            continue
    return pl.concat(frames) if frames else pl.DataFrame()


def load_ohlcv(ticker: str, days: int = 504) -> pl.DataFrame:
    """Load last `days` rows of OHLCV for a ticker. Returns empty DataFrame if missing."""
    ticker_dir = OHLCV_DIR / ticker
    parquet_files = list(ticker_dir.glob("*.parquet")) if ticker_dir.exists() else []
    if not parquet_files:
        return pl.DataFrame()
    try:
        return pl.concat([pl.read_parquet(str(f)) for f in parquet_files]).sort("date").tail(days)
    except (OSError, Exception):
        return pl.DataFrame()


def load_fundamentals(ticker: str) -> pl.DataFrame:
    """Load latest 4 quarterly fundamentals for a ticker. Returns empty DataFrame if missing."""
    path = FUNDAMENTALS_DIR / ticker / "quarterly.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(str(path)).sort("period_end").tail(4)


def load_lgbm_feature_importances() -> dict[str, float]:
    """Load LightGBM feature importances from lgbm_q50.pkl + feature_names.json.
    Returns empty dict if artifacts missing."""
    import json
    import pickle
    q50_path = LGBM_ARTIFACTS_DIR / "lgbm_q50.pkl"
    names_path = LGBM_ARTIFACTS_DIR / "feature_names.json"
    if not q50_path.exists() or not names_path.exists():
        return {}
    with open(names_path) as f:
        feature_names = json.load(f)
    with open(q50_path, "rb") as f:
        model = pickle.load(f)
    importances = model.feature_importances_
    return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1]))


# ── Page 1 — Top Picks ────────────────────────────────────────────────────────


def page_top_picks():
    st.title("AI Infrastructure Predictor — Top Picks")
    df = load_latest_predictions()
    if df.is_empty():
        st.error("No predictions available yet. Run `python models/inference.py` to generate.")
        return

    # Staleness warning: if predictions are >2 days old
    date_dirs = sorted(PREDICTIONS_DIR.glob("date=*"))
    last_date_str = date_dirs[-1].name.replace("date=", "")
    last_date = datetime.date.fromisoformat(last_date_str)
    days_old = (datetime.date.today() - last_date).days
    if days_old > 2:
        st.warning(f"Predictions are {days_old} days old (last updated {last_date_str}). Run inference to refresh.")
    else:
        st.caption(f"Last updated: {last_date_str}")

    # Add sector column
    df = df.with_columns(
        pl.col("ticker").map_elements(lambda t: SECTOR.get(t, "Unknown"), return_dtype=pl.Utf8).alias("sector")
    )

    # Ranked table with percentage columns
    display_df = df.select([
        "rank", "ticker", "sector",
        "expected_annual_return", "confidence_low", "confidence_high",
    ]).with_columns([
        (pl.col("expected_annual_return") * 100).round(1).alias("Expected Return (%)"),
        (pl.col("confidence_low") * 100).round(1).alias("Confidence Low (%)"),
        (pl.col("confidence_high") * 100).round(1).alias("Confidence High (%)"),
        ((pl.col("confidence_high") - pl.col("confidence_low")) * 100).round(1).alias("Width (%)"),
    ]).select(["rank", "ticker", "sector", "Expected Return (%)", "Confidence Low (%)", "Confidence High (%)", "Width (%)"])

    st.subheader("All Tickers Ranked by Expected 1-Year Return")
    st.dataframe(display_df.to_pandas(), use_container_width=True, hide_index=True)

    # Top 5 buy list
    st.subheader("Top 5 Picks")
    importances = load_lgbm_feature_importances()
    top_features = list(importances.keys())[:3] if importances else []
    top5 = df.head(5)

    cols = st.columns(5)
    for col, row in zip(cols, top5.iter_rows(named=True)):
        with col:
            ret_pct = row["expected_annual_return"] * 100
            low_pct = row["confidence_low"] * 100
            high_pct = row["confidence_high"] * 100
            st.metric(
                label=f"#{row['rank']} {row['ticker']}",
                value=f"{ret_pct:.1f}%",
            )
            st.caption(f"{row['sector']} | Range: {low_pct:.1f}%–{high_pct:.1f}%")
            if top_features:
                st.caption("Key signals: " + ", ".join(top_features))


# ── Page 2 — Stock Drill-Down ─────────────────────────────────────────────────


def page_drill_down():
    st.title("AI Infrastructure Predictor — Stock Drill-Down")
    ticker = st.sidebar.selectbox("Select ticker", TICKERS)

    # Price chart (2 years, with 20-day SMA)
    st.subheader(f"{ticker} — Price Chart (2 years)")
    ohlcv = load_ohlcv(ticker, days=504)
    if ohlcv.is_empty():
        st.info(f"No OHLCV data for {ticker}. Run ingestion first.")
    else:
        ohlcv_pd = ohlcv.to_pandas()
        ohlcv_pd["sma_20"] = ohlcv_pd["close_price"].rolling(20).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlcv_pd["date"], y=ohlcv_pd["close_price"],
                                 name="Close Price", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=ohlcv_pd["date"], y=ohlcv_pd["sma_20"],
                                 name="SMA 20", line=dict(color="#FF9800", dash="dash")))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Prediction history (expected return + confidence band)
    st.subheader(f"{ticker} — Prediction History")
    hist = load_ticker_predictions(ticker)
    if hist.is_empty():
        st.info(f"No prediction history for {ticker}.")
    else:
        hist_pd = hist.to_pandas()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist_pd["as_of_date"], y=hist_pd["confidence_high"] * 100,
            name="Confidence High", line=dict(color="#81C784", dash="dot"),
        ))
        fig2.add_trace(go.Scatter(
            x=hist_pd["as_of_date"], y=hist_pd["expected_annual_return"] * 100,
            name="Expected Return (%)", line=dict(color="#4CAF50"),
            fill="tonexty",
        ))
        fig2.add_trace(go.Scatter(
            x=hist_pd["as_of_date"], y=hist_pd["confidence_low"] * 100,
            name="Confidence Low", line=dict(color="#E57373", dash="dot"),
        ))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Expected Return (%)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Fundamentals table
    st.subheader(f"{ticker} — Recent Fundamentals")
    fund = load_fundamentals(ticker)
    if fund.is_empty():
        st.info(f"No fundamentals data for {ticker}. Run `python ingestion/fundamental_ingestion.py`.")
    else:
        st.dataframe(fund.to_pandas(), use_container_width=True, hide_index=True)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="AI Infra Predictor", layout="wide")
    page = st.sidebar.radio("Page", ["Top Picks", "Stock Drill-Down"])
    if page == "Top Picks":
        page_top_picks()
    else:
        page_drill_down()


if __name__ == "__main__":
    main()
