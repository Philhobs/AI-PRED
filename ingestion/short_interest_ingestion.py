"""
Short interest / short sale volume ingestion from FINRA daily files.

Source: FINRA Consolidated NMS (CNMS) daily short sale volume files
URL: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
Frequency: Daily (trading days only), updated next morning
Free, no API key required.

Feature: short_vol_ratio = ShortVolume / TotalVolume (fraction sold short each day)
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research/academic)"}


def _fetch_day(trading_date: date, watchlist: set[str]) -> list[dict]:
    """Download one FINRA daily file and return rows for watchlist tickers."""
    url = _FINRA_URL.format(date=trading_date.strftime("%Y%m%d"))
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            return []  # weekend / holiday
        resp.raise_for_status()
    except requests.RequestException as exc:
        _LOG.debug("FINRA %s: %s", trading_date, exc)
        return []

    rows: list[dict] = []
    for line in resp.text.splitlines()[1:]:  # skip header
        parts = line.split("|")
        if len(parts) < 5:
            continue
        symbol = parts[1]
        if symbol not in watchlist:
            continue
        try:
            short_vol = float(parts[2])
            total_vol = float(parts[4])
        except ValueError:
            continue
        if total_vol == 0:
            continue
        rows.append({
            "date": trading_date,
            "ticker": symbol,
            "short_volume": short_vol,
            "total_volume": total_vol,
            "short_vol_ratio": short_vol / total_vol,
        })
    return rows


def fetch_short_interest(watchlist: list[str], days_back: int = 365) -> pl.DataFrame:
    """Fetch FINRA daily short sale volume for watchlist tickers.

    Downloads one file per calendar day (FINRA returns 404 for non-trading days).
    Returns DataFrame with [date, ticker, short_volume, total_volume, short_vol_ratio].
    """
    watch_set = set(watchlist)
    end = date.today()
    start = end - timedelta(days=days_back)

    all_rows: list[dict] = []
    current = start
    fetched = 0
    while current <= end:
        day_rows = _fetch_day(current, watch_set)
        all_rows.extend(day_rows)
        if day_rows:
            fetched += 1
        current += timedelta(days=1)
        time.sleep(0.2)  # polite rate limit

    _LOG.info("FINRA: fetched %d trading days, %d ticker-day rows", fetched, len(all_rows))
    if not all_rows:
        return pl.DataFrame(schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "short_volume": pl.Float64,
            "total_volume": pl.Float64,
            "short_vol_ratio": pl.Float64,
        })

    return pl.DataFrame(all_rows).with_columns(pl.col("date").cast(pl.Date))


def save_short_interest(df: pl.DataFrame, output_dir: Path) -> None:
    """Write to output_dir/short_interest_daily.parquet."""
    out_path = output_dir / "short_interest_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="snappy")
    _LOG.info("Saved %d short interest rows to %s", len(df), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from ingestion.ticker_registry import us_listed_tickers

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "raw" / "financials" / "short_interest"

    # FINRA only covers US-listed equities; foreign-listed symbols are skipped.
    tickers = us_listed_tickers()
    _LOG.info("Fetching FINRA short sale volume for %d tickers...", len(tickers))
    df = fetch_short_interest(tickers, days_back=365)
    _LOG.info("Fetched %d rows for %d tickers", len(df), df["ticker"].n_unique() if len(df) > 0 else 0)
    save_short_interest(df, output_dir)
