# tests/test_ticker_registry.py


def test_ticker_count():
    from ingestion.ticker_registry import TICKERS, TICKER_LAYERS
    # 141 + 8 new robotics-pillar tickers (EMR, TSLA, 1683.HK, 005380.KS,
    # TXN, MCHP, 6723.T, ADI) = 149.
    assert len(TICKERS) == 149
    assert len(TICKER_LAYERS) == 149


def test_all_layers_present():
    from ingestion.ticker_registry import TICKER_LAYERS, LAYER_IDS
    layers_used = set(TICKER_LAYERS.values())
    assert layers_used == set(LAYER_IDS.keys())


def test_tickers_in_layer():
    from ingestion.ticker_registry import tickers_in_layer
    cloud = tickers_in_layer("cloud")
    assert "MSFT" in cloud and "AMZN" in cloud
    assert len(cloud) == 9


def test_hyperscalers_are_cloud():
    from ingestion.ticker_registry import HYPERSCALERS, TICKER_LAYERS
    for t in HYPERSCALERS:
        assert TICKER_LAYERS[t] == "cloud"


def test_layers_returns_15():
    from ingestion.ticker_registry import layers
    assert len(layers()) == 15


def test_layers_order():
    from ingestion.ticker_registry import layers
    result = layers()
    assert result[0] == "cloud"
    # robotics pillar (ids 11/12/13) comes before cyber pillar (14/15)
    assert result[-5] == "robotics_industrial"
    assert result[-4] == "robotics_medical_humanoid"
    assert result[-3] == "robotics_mcu_chips"
    assert result[-2] == "cyber_pureplay"
    assert result[-1] == "cyber_platform"


def test_cyber_pureplay_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "cyber_pureplay" in LAYER_IDS
    assert LAYER_IDS["cyber_pureplay"] == 14   # was 12, shifted by robotics split
    tickers = tickers_in_layer("cyber_pureplay")
    assert len(tickers) == 5
    assert "CRWD" in tickers
    assert "ZS" in tickers
    assert "S" in tickers
    assert "DARK.L" in tickers
    assert "VRNS" in tickers


def test_cyber_platform_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "cyber_platform" in LAYER_IDS
    assert LAYER_IDS["cyber_platform"] == 15   # was 13, shifted by robotics split
    tickers = tickers_in_layer("cyber_platform")
    assert len(tickers) == 9
    for expected in ["PANW", "FTNT", "CHKP", "CYBR", "TENB", "QLYS", "OKTA", "AKAM", "RPD"]:
        assert expected in tickers, f"{expected} missing from cyber_platform"


def test_cik_map_covers_domestic_tickers():
    """CIK_MAP must have entries for original 83 domestic US tickers."""
    from ingestion.edgar_fundamentals_ingestion import CIK_MAP
    from ingestion.ticker_registry import TICKER_EXCHANGE
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
    foreign = {"TSM", "ASML", "ARM", "NOK", "IREN", "STM", "ERIC"}
    domestic = [t for t in us_listed if t not in foreign]
    missing = [t for t in domestic if t not in CIK_MAP]
    assert missing == [], f"Missing CIKs for: {missing}"


def test_tickerinfo_fields_complete():
    from ingestion.ticker_registry import TICKERS_INFO
    for t in TICKERS_INFO:
        assert t.symbol,   f"Empty symbol in entry: {t}"
        assert t.layer,    f"Empty layer for {t.symbol}"
        assert t.exchange, f"Empty exchange for {t.symbol}"
        assert t.currency, f"Empty currency for {t.symbol}"
        assert t.country,  f"Empty country for {t.symbol}"
    symbols = [t.symbol for t in TICKERS_INFO]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_robotics_industrial_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics_industrial" in LAYER_IDS
    assert LAYER_IDS["robotics_industrial"] == 11
    industrial = tickers_in_layer("robotics_industrial")
    assert len(industrial) == 11
    expected = {"ROK", "ZBRA", "CGNX", "SYM", "ABBN.SW", "KGX.DE",
                "HEXA-B.ST", "6954.T", "6506.T", "6861.T", "EMR"}
    assert set(industrial) == expected


def test_robotics_medical_humanoid_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics_medical_humanoid" in LAYER_IDS
    assert LAYER_IDS["robotics_medical_humanoid"] == 12
    mh = tickers_in_layer("robotics_medical_humanoid")
    assert len(mh) == 4
    assert set(mh) == {"ISRG", "TSLA", "1683.HK", "005380.KS"}


def test_robotics_mcu_chips_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics_mcu_chips" in LAYER_IDS
    assert LAYER_IDS["robotics_mcu_chips"] == 13
    mcu = tickers_in_layer("robotics_mcu_chips")
    assert len(mcu) == 4
    assert set(mcu) == {"TXN", "MCHP", "6723.T", "ADI"}


def test_non_usd_tickers():
    from ingestion.ticker_registry import non_usd_tickers, TICKER_CURRENCY
    # Was 37; +1683.HK (HKD) +005380.KS (KRW) +6723.T (JPY) = 40
    result = non_usd_tickers()
    assert len(result) == 40
    for t in result:
        assert TICKER_CURRENCY[t] != "USD", f"{t} is USD but in non_usd_tickers()"
    assert "NVDA" not in result
    assert "ABBN.SW" in result
    assert "DARK.L" in result
    assert "1683.HK" in result
    assert "005380.KS" in result
    assert "6723.T" in result


def test_pending_ipo_watchlist_structure():
    """PENDING_IPO_WATCHLIST entries are well-formed and reference real layers."""
    from ingestion.ticker_registry import PENDING_IPO_WATCHLIST, LAYER_IDS
    assert len(PENDING_IPO_WATCHLIST) >= 2
    required_keys = {"name", "expected_symbol", "layer", "expected_date"}
    for entry in PENDING_IPO_WATCHLIST:
        assert required_keys <= entry.keys(), f"Missing keys in {entry}"
        assert entry["layer"] in LAYER_IDS, (
            f"Layer {entry['layer']!r} not in LAYER_IDS"
        )
    names = {e["name"] for e in PENDING_IPO_WATCHLIST}
    assert "Unitree Robotics" in names
    assert "Boston Dynamics" in names
