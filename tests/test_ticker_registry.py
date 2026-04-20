# tests/test_ticker_registry.py


def test_ticker_count():
    from ingestion.ticker_registry import TICKERS, TICKER_LAYERS
    assert len(TICKERS) == 127
    assert len(TICKER_LAYERS) == 127


def test_all_layers_present():
    from ingestion.ticker_registry import TICKER_LAYERS, LAYER_IDS
    layers_used = set(TICKER_LAYERS.values())
    assert layers_used == set(LAYER_IDS.keys())


def test_tickers_in_layer():
    from ingestion.ticker_registry import tickers_in_layer
    cloud = tickers_in_layer("cloud")
    assert "MSFT" in cloud and "AMZN" in cloud
    assert len(cloud) == 9  # was 6; +SAP.DE, CAP.PA, OVH.PA


def test_hyperscalers_are_cloud():
    from ingestion.ticker_registry import HYPERSCALERS, TICKER_LAYERS
    for t in HYPERSCALERS:
        assert TICKER_LAYERS[t] == "cloud"


def test_layers_returns_11():
    from ingestion.ticker_registry import layers
    assert len(layers()) == 11


def test_layers_order():
    from ingestion.ticker_registry import layers
    result = layers()
    assert result[0] == "cloud"
    assert result[-1] == "robotics"  # robotics=11, metals=10


def test_cik_map_covers_domestic_tickers():
    """CIK_MAP must have entries for original 83 domestic US tickers."""
    from ingestion.edgar_fundamentals_ingestion import CIK_MAP
    from ingestion.ticker_registry import TICKER_EXCHANGE
    # Only US-listed tickers could have SEC CIK entries
    us_listed = [t for t in [
        "MSFT", "AMZN", "GOOGL", "META", "ORCL", "IBM",
        "NVDA", "AMD", "AVGO", "MRVL", "TSM", "ASML", "INTC", "ARM",
        "MU", "SNPS", "CDNS",
        "AMAT", "LRCX", "KLAC", "ENTG", "MKSI", "UCTT", "ICHR",
        "TER", "ONTO", "APD", "LIN",
        "ANET", "CSCO", "CIEN", "COHR", "LITE", "INFN", "NOK", "VIAV",
        "SMCI", "DELL", "HPE", "NTAP", "PSTG", "STX", "WDC",
        "EQIX", "DLR", "AMT", "CCI", "IREN", "APLD",
        "CEG", "VST", "NRG", "TLN", "NEE", "SO", "EXC", "ETR",
        "GEV", "BWX", "OKLO", "SMR", "FSLR",
        "VRT", "NVENT", "JCI", "TT", "CARR", "GNRC", "HUBB",
        "PWR", "MTZ", "EME", "MYR", "IESC", "AGX",
        "FCX", "SCCO", "AA", "NUE", "STLD", "MP", "UUUU", "ECL",
    ] if TICKER_EXCHANGE.get(t, "US") == "US"]
    # Foreign private issuers / non-SEC-registrants — excluded by design
    foreign = {"TSM", "ASML", "ARM", "NOK", "IREN", "STM", "ERIC"}
    domestic = [t for t in us_listed if t not in foreign]
    missing = [t for t in domestic if t not in CIK_MAP]
    assert missing == [], f"Missing CIKs for: {missing}"


# ── New tests ──────────────────────────────────────────────────────────────────

def test_tickerinfo_fields_complete():
    """Every TickerInfo entry has non-empty fields and no duplicate symbols."""
    from ingestion.ticker_registry import TICKERS_INFO
    for t in TICKERS_INFO:
        assert t.symbol,   f"Empty symbol in entry: {t}"
        assert t.layer,    f"Empty layer for {t.symbol}"
        assert t.exchange, f"Empty exchange for {t.symbol}"
        assert t.currency, f"Empty currency for {t.symbol}"
        assert t.country,  f"Empty country for {t.symbol}"
    symbols = [t.symbol for t in TICKERS_INFO]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_robotics_layer_populated():
    """robotics layer exists in LAYER_IDS and contains expected tickers."""
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics" in LAYER_IDS
    assert LAYER_IDS["robotics"] == 11
    robotics = tickers_in_layer("robotics")
    assert len(robotics) == 11
    assert "ABBN.SW" in robotics
    assert "6954.T"  in robotics
    assert "ISRG"    in robotics


def test_non_usd_tickers():
    """non_usd_tickers() returns only tickers with non-USD currency."""
    from ingestion.ticker_registry import non_usd_tickers, TICKER_CURRENCY
    result = non_usd_tickers()
    assert len(result) == 36
    for t in result:
        assert TICKER_CURRENCY[t] != "USD", f"{t} is USD but in non_usd_tickers()"
    # NVDA is USD — must not appear
    assert "NVDA" not in result
    # ABBN.SW is CHF — must appear
    assert "ABBN.SW" in result
