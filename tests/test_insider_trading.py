import pytest
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
