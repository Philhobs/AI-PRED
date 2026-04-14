import pytest
import polars as pl
from ingestion.insider_trading_ingestion import _parse_form4_xml

PURCHASE_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2024-01-15</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>John Smith</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-01-14</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>500.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

GRANT_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2024-01-15</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Jane Doe</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-01-14</value></transactionDate>
      <transactionCoding>
        <transactionCode>A</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>0.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_xml_purchase():
    rows = _parse_form4_xml(PURCHASE_XML, ticker="NVDA", filed_date="2024-01-15")
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "NVDA"
    assert r["transaction_code"] == "P"
    assert r["shares"] == pytest.approx(1000.0)
    assert r["price_per_share"] == pytest.approx(500.0)
    assert r["value"] == pytest.approx(500000.0)
    assert r["insider_name"] == "John Smith"
    assert r["insider_title"] == "CEO"
    assert r["transaction_date"] == "2024-01-14"
    assert r["filed_date"] == "2024-01-15"


def test_parse_form4_xml_excludes_grants():
    rows = _parse_form4_xml(GRANT_XML, ticker="NVDA", filed_date="2024-01-15")
    assert rows == []


def test_parse_form4_xml_sale():
    """Code S (sale) is included in output."""
    sale_xml = PURCHASE_XML.replace("<transactionCode>P</transactionCode>",
                                    "<transactionCode>S</transactionCode>")
    rows = _parse_form4_xml(sale_xml, ticker="NVDA", filed_date="2024-01-15")
    assert len(rows) == 1
    assert rows[0]["transaction_code"] == "S"


def test_parse_form4_xml_parse_error():
    """Malformed XML returns empty list (no exception raised)."""
    rows = _parse_form4_xml("not valid xml <<<", ticker="NVDA", filed_date="2024-01-15")
    assert rows == []


def test_parse_form4_xml_empty_value_tag():
    """Empty <value/> in transactionDate is handled gracefully (row skipped or date is empty string)."""
    xml = PURCHASE_XML.replace("<value>2024-01-14</value>", "<value/>")
    # Replacing date value element — should either skip the transaction or return empty string date
    rows = _parse_form4_xml(xml, ticker="NVDA", filed_date="2024-01-15")
    # Either 0 rows (if we want to skip) or 1 row with empty date — either is acceptable
    # The important thing is it does not raise AttributeError
    assert isinstance(rows, list)


def test_parse_form4_xml_with_namespace():
    """XML with namespace declaration is parsed correctly (namespace stripped)."""
    namespaced_xml = PURCHASE_XML.replace(
        "<ownershipDocument>",
        '<ownershipDocument xmlns="http://www.sec.gov/cgi-bin/browse-edgar">'
    )
    rows = _parse_form4_xml(namespaced_xml, ticker="NVDA", filed_date="2024-01-15")
    assert len(rows) == 1
    assert rows[0]["transaction_code"] == "P"


from ingestion.insider_trading_ingestion import _parse_house_json_records, _parse_amount_band
from ingestion.insider_trading_ingestion import CIK_MAP

HOUSE_FIXTURE = [
    {
        "ticker": "NVDA",
        "transaction_date": "2024-01-10",
        "disclosure_date": "2024-01-20",
        "representative": "John Representative",
        "party": "democrat",
        "type": "purchase",
        "amount": "$15,001 - $50,000",
        "asset_description": "NVIDIA Corp",
    },
    {
        "ticker": "XYZ_NOT_IN_WATCHLIST",
        "transaction_date": "2024-01-10",
        "disclosure_date": "2024-01-20",
        "representative": "Jane Rep",
        "party": "republican",
        "type": "sale",
        "amount": "$1,001 - $15,000",
        "asset_description": "XYZ Corp",
    },
    {
        "ticker": "MSFT",
        "transaction_date": "2024-01-12",
        "disclosure_date": "2024-01-22",
        "representative": "Bob Senator",
        "party": "republican",
        "type": "sale",
        "amount": "$50,001 - $100,000",
        "asset_description": "Microsoft Corp",
    },
]


def test_parse_congressional_house_json():
    """Fixture House JSON → correct ticker, amount_mid, chamber."""
    watchlist = set(CIK_MAP.keys())
    df = _parse_house_json_records(HOUSE_FIXTURE, watchlist=watchlist)
    assert set(df["ticker"].to_list()) == {"NVDA", "MSFT"}
    nvda_row = df.filter(pl.col("ticker") == "NVDA").row(0, named=True)
    assert nvda_row["transaction_type"] == "purchase"
    assert nvda_row["chamber"] == "house"
    assert nvda_row["amount_mid"] == pytest.approx(32500.5)
    msft_row = df.filter(pl.col("ticker") == "MSFT").row(0, named=True)
    assert msft_row["amount_mid"] == pytest.approx(75000.5)


def test_congressional_filters_to_watchlist():
    """Tickers outside 24-ticker watchlist are excluded."""
    watchlist = set(CIK_MAP.keys())
    df = _parse_house_json_records(HOUSE_FIXTURE, watchlist=watchlist)
    assert "XYZ_NOT_IN_WATCHLIST" not in df["ticker"].to_list()


def test_congressional_amount_parsing():
    """Amount band strings parse to correct midpoints."""
    assert _parse_amount_band("$1,001 - $15,000") == pytest.approx(8000.5)
    assert _parse_amount_band("$15,001 - $50,000") == pytest.approx(32500.5)
    assert _parse_amount_band("$50,001 - $100,000") == pytest.approx(75000.5)
    assert _parse_amount_band("$100,001 - $250,000") == pytest.approx(175000.5)
    assert _parse_amount_band("$250,001 - $500,000") == pytest.approx(375000.5)
    assert _parse_amount_band("$500,001 - $1,000,000") == pytest.approx(750000.5)
    assert _parse_amount_band("over $1,000,000") == pytest.approx(1500000.0)
    assert _parse_amount_band("unknown format") is None


# ── insider feature helpers ───────────────────────────────────────────────────
from datetime import date as _date


def _make_insider_df(rows: list[dict]) -> pl.DataFrame:
    """Build minimal insider trades DataFrame for testing."""
    if not rows:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "transaction_date": pl.Date,
            "insider_name": pl.Utf8,
            "transaction_code": pl.Utf8,
            "value": pl.Float64,
        })
    return pl.DataFrame({
        "ticker": [r["ticker"] for r in rows],
        "transaction_date": [r["date"] for r in rows],
        "insider_name": [r.get("name", "Person") for r in rows],
        "transaction_code": [r["code"] for r in rows],
        "value": [float(r["value"]) for r in rows],
    }).with_columns(pl.col("transaction_date").cast(pl.Date))


def _make_congress_df(rows: list[dict]) -> pl.DataFrame:
    """Build minimal congressional trades DataFrame for testing."""
    if not rows:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "trade_date": pl.Date,
            "transaction_type": pl.Utf8,
            "amount_mid": pl.Float64,
        })
    return pl.DataFrame({
        "ticker": [r["ticker"] for r in rows],
        "trade_date": [r["date"] for r in rows],
        "transaction_type": [r["type"] for r in rows],
        "amount_mid": [float(r["amount"]) for r in rows],
    }).with_columns(pl.col("trade_date").cast(pl.Date))


def test_compute_cluster_buy_90d():
    """3 distinct insiders buying in 90-day window → cluster_buy = 3 (not 4)."""
    from processing.insider_features import _compute_cluster_buy_90d
    trades = _make_insider_df([
        {"ticker": "NVDA", "date": _date(2024, 1, 10), "code": "P", "value": 1e6, "name": "Alice"},
        {"ticker": "NVDA", "date": _date(2024, 1, 15), "code": "P", "value": 2e6, "name": "Bob"},
        {"ticker": "NVDA", "date": _date(2024, 1, 20), "code": "P", "value": 3e6, "name": "Carol"},
        {"ticker": "NVDA", "date": _date(2024, 1, 25), "code": "P", "value": 5e5, "name": "Alice"},  # Alice buys again
    ])
    result = _compute_cluster_buy_90d(trades, ticker="NVDA", as_of=_date(2024, 3, 1), window_days=90)
    assert result == 3  # distinct insiders, not transaction count


def test_compute_net_value_30d():
    """$5M purchases, $2M sales in window → net = 3.0 millions."""
    from processing.insider_features import _compute_net_value_30d
    trades = _make_insider_df([
        {"ticker": "NVDA", "date": _date(2024, 1, 20), "code": "P", "value": 5_000_000},
        {"ticker": "NVDA", "date": _date(2024, 1, 22), "code": "S", "value": 2_000_000},
    ])
    result = _compute_net_value_30d(trades, ticker="NVDA", as_of=_date(2024, 2, 15), window_days=30)
    assert result == pytest.approx(3.0)


def test_compute_buy_sell_ratio_no_trades():
    """Empty window → None (not zero)."""
    from processing.insider_features import _compute_buy_sell_ratio_90d
    trades = _make_insider_df([])
    result = _compute_buy_sell_ratio_90d(trades, ticker="NVDA", as_of=_date(2024, 3, 1), window_days=90)
    assert result is None


def test_compute_congress_net_buy_90d():
    """$3M purchases - $1M sales → net = 2.0 millions."""
    from processing.insider_features import _compute_congress_net_buy_90d
    congress = _make_congress_df([
        {"ticker": "NVDA", "date": _date(2024, 1, 10), "type": "purchase", "amount": 3_000_000},
        {"ticker": "NVDA", "date": _date(2024, 1, 15), "type": "sale", "amount": 1_000_000},
    ])
    result = _compute_congress_net_buy_90d(congress, ticker="NVDA", as_of=_date(2024, 3, 1), window_days=90)
    assert result == pytest.approx(2.0)
