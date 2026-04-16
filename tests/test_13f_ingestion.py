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


def test_parse_holdings_xml_filters_cusip_and_type():
    """parse_holdings_xml returns only SH-type rows for CUSIPs in the map."""
    from ingestion.sec_13f_ingestion import parse_holdings_xml

    xml_str = """<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/cgi-bin/browse-edgar">
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <cusip>67066G104</cusip>
    <value>5000000</value>
    <shrsOrPrnAmt><sshPrnamt>1000000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>SOME BOND</nameOfIssuer>
    <cusip>67066G104</cusip>
    <value>1000</value>
    <shrsOrPrnAmt><sshPrnamt>500</sshPrnamt><sshPrnamtType>PRN</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>NOT IN MAP</nameOfIssuer>
    <cusip>999999999</cusip>
    <value>9999</value>
    <shrsOrPrnAmt><sshPrnamt>100</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
</informationTable>"""

    cusip_map = {"NVDA": "67066G104"}
    result = parse_holdings_xml(xml_str, cusip_map)

    assert len(result) == 1, f"Expected 1 row (SH type, in map), got {len(result)}"
    assert result[0]["ticker"] == "NVDA"
    assert result[0]["shares_held"] == 1_000_000
    assert result[0]["value_usd_thousands"] == 5_000_000


def test_parse_holdings_xml_returns_empty_on_bad_xml():
    from ingestion.sec_13f_ingestion import parse_holdings_xml
    result = parse_holdings_xml("NOT XML AT ALL <<<>>>", {"NVDA": "67066G104"})
    assert result == []


def test_parse_quarter_index_filters_13f_hr():
    """fetch_quarter_index parses pipe-separated lines, returns only 13F-HR rows."""
    from ingestion.sec_13f_ingestion import _parse_index_content

    content = (
        "Company Name|Form Type|CIK|Date Filed|Filename\n"
        "----------------------------------------\n"
        "VANGUARD GROUP INC|13F-HR|0000102909|2024-02-14|"
        "edgar/data/102909/0000102909-24-000010-index.htm\n"
        "SOME OTHER CO|10-K|0000012345|2024-01-15|"
        "edgar/data/12345/0000012345-24-000001-index.htm\n"
        "BLACKROCK INC|13F-HR|0001086364|2024-02-13|"
        "edgar/data/1086364/0001086364-24-000005-index.htm\n"
    )
    df = _parse_index_content(content)
    assert len(df) == 2
    assert set(df["cik"].to_list()) == {"0000102909", "0001086364"}
    assert all(f.endswith("-index.htm") for f in df["filename"].to_list())


import gzip
from unittest.mock import patch, MagicMock
import polars as pl


def test_ingest_quarter_writes_parquet(tmp_path):
    """ingest_quarter saves per-filer Parquet with correct schema."""
    from ingestion.sec_13f_ingestion import ingest_quarter

    # Synthetic quarter index content (one 13F-HR filer)
    index_gz_content = gzip.compress(
        b"Company Name|Form Type|CIK|Date Filed|Filename\n"
        b"---\n"
        b"VANGUARD GROUP|13F-HR|0000102909|2024-02-14|"
        b"edgar/data/102909/0000102909-24-000001-index.htm\n"
    )

    # Synthetic filing index HTML (points to infotable.xml)
    index_html = b"""<html><body>
<table><tr><td><a href="infotable.xml">infotable.xml</a></td></tr></table>
</body></html>"""

    # Synthetic XML with one NVDA holding
    xml_content = b"""<?xml version="1.0"?>
<informationTable>
  <infoTable>
    <cusip>67066G104</cusip>
    <value>9999999</value>
    <shrsOrPrnAmt><sshPrnamt>5000000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
</informationTable>"""

    cusip_map = {"NVDA": "67066G104"}

    def mock_get(url, **kwargs):
        m = MagicMock()
        m.raise_for_status = MagicMock()
        if "company.gz" in url:
            m.content = index_gz_content
            m.status_code = 200
        elif "index.htm" in url:
            m.content = index_html
            m.text = index_html.decode()
            m.status_code = 200
        elif "infotable.xml" in url or url.endswith(".xml"):
            m.content = xml_content
            m.text = xml_content.decode()
            m.status_code = 200
        else:
            m.status_code = 404
        return m

    with patch("ingestion.sec_13f_ingestion.requests.get", side_effect=mock_get):
        with patch("ingestion.sec_13f_ingestion.time.sleep"):  # skip actual sleep
            rows = ingest_quarter(2024, 1, cusip_map, tmp_path, top_n=1)

    assert rows > 0, "Expected at least one row written"
    parquets = list(tmp_path.glob("2024Q1/*.parquet"))
    assert len(parquets) == 1
    df = pl.read_parquet(parquets[0])
    assert "ticker" in df.columns
    assert "shares_held" in df.columns
    assert df["ticker"][0] == "NVDA"
    assert df["shares_held"][0] == 5_000_000
