# tests/test_ticker_registry.py
def test_ticker_count():
    from ingestion.ticker_registry import TICKERS, TICKER_LAYERS
    assert len(TICKERS) == 83
    assert len(TICKER_LAYERS) == 83

def test_all_layers_present():
    from ingestion.ticker_registry import TICKER_LAYERS, LAYER_IDS
    layers_used = set(TICKER_LAYERS.values())
    assert layers_used == set(LAYER_IDS.keys())

def test_tickers_in_layer():
    from ingestion.ticker_registry import tickers_in_layer
    cloud = tickers_in_layer("cloud")
    assert "MSFT" in cloud and "AMZN" in cloud
    assert len(cloud) == 6

def test_hyperscalers_are_cloud():
    from ingestion.ticker_registry import HYPERSCALERS, TICKER_LAYERS
    for t in HYPERSCALERS:
        assert TICKER_LAYERS[t] == "cloud"

def test_layers_returns_10():
    from ingestion.ticker_registry import layers
    assert len(layers()) == 10
