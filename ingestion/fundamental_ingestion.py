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
