# tests/test_13f_ingestion.py
import json
from pathlib import Path
import pytest


def test_cusip_map_exists_and_covers_major_tickers():
    """cusip_map.json must exist after running build_cusip_map.py."""
    path = Path("data/raw/financials/cusip_map.json")
    assert path.exists(), "Run: python ingestion/build_cusip_map.py"
    cusip_map = json.loads(path.read_text())
    # Major tickers with known CUSIPs
    assert cusip_map.get("NVDA") == "67066G104", f"NVDA CUSIP wrong: {cusip_map.get('NVDA')}"
    assert cusip_map.get("MSFT") == "594918104", f"MSFT CUSIP wrong: {cusip_map.get('MSFT')}"
    assert cusip_map.get("AMZN") == "023135106", f"AMZN CUSIP wrong: {cusip_map.get('AMZN')}"
    # All entries must be 9-char strings
    for ticker, cusip in cusip_map.items():
        assert isinstance(cusip, str) and len(cusip) == 9, (
            f"{ticker} CUSIP {cusip!r} is not 9 chars"
        )
    # Must cover most of our watchlist (some foreign tickers may be absent)
    assert len(cusip_map) >= 60, f"Expected ≥60 entries, got {len(cusip_map)}"
