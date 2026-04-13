import requests
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

TICKERS = [
    "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AMD", "AVGO", "MRVL", "TSM",
    "ASML", "AMAT", "LRCX", "KLAC",
    "VRT", "SMCI", "DELL", "HPE",
    "EQIX", "DLR", "AMT",
    "CEG", "VST", "NRG", "TLN",
    "NEE", "ETR", "SO", "D", "EXC",
]

CIK_MAP = {
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
}

CAPEX_CONCEPT = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
CAPEX_CONCEPTS = [
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsToAcquireProductiveAssets",
]
REVENUE_CONCEPT = "us-gaap:Revenues"
RD_CONCEPT = "us-gaap:ResearchAndDevelopmentExpense"


def fetch_edgar_xbrl(cik: str, concept: str) -> list[dict]:
    """
    Fetch XBRL financial data from SEC EDGAR.
    Free, no API key required.
    SEC fair-use requires a descriptive User-Agent header.
    Only returns 10-K, 10-Q, and 20-F filings.
    """
    taxonomy, tag = concept.split(":")
    url = f"https://data.sec.gov/api/xbrl/companyconcept/{cik}/{taxonomy}/{tag}.json"
    headers = {"User-Agent": "ai-infra-predictor research@example.com"}

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    units = data.get("units", {}).get("USD", [])

    return [
        {
            "cik": cik,
            "concept": concept,
            "value": item["val"],
            "start": item.get("start"),
            "end": item["end"],
            "form": item["form"],
            "filed": item["filed"],
            "accn": item["accn"],
        }
        for item in units
        if item.get("form") in ("10-K", "10-Q", "20-F")
    ]


def fetch_all_hyperscaler_capex(output_dir: Path):
    """
    Fetch quarterly capex for all hyperscalers from EDGAR XBRL.
    Tries each concept in CAPEX_CONCEPTS in order, using the first that returns data.
    Free, no API key required.
    """
    records = []
    for ticker, cik in CIK_MAP.items():
        fetched = []
        for concept in CAPEX_CONCEPTS:
            try:
                fetched = fetch_edgar_xbrl(cik, concept)
                if fetched:
                    break  # Use first concept that returns data
            except requests.exceptions.RequestException as e:
                print(f"[Financial] {ticker} concept {concept}: {e} — trying next")
        time.sleep(1)  # Rate limit: one sleep per ticker — SEC fair use
        if fetched:
            for row in fetched:
                row["ticker"] = ticker
                records.append(row)
            print(f"[Financial] {ticker}: {len(fetched)} capex records from EDGAR")
        else:
            print(f"[Financial] {ticker}: no capex data found across all concepts")

    if records:
        path = output_dir / "financials" / "capex_history.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(path, compression="snappy")


def fetch_fred_energy_indicators() -> dict:
    """
    Fetch macro energy indicators from FRED.
    Free, no key required (FRED_API_KEY env var increases rate limit).
    FRED uses '.' for missing observations — converted to None.
    """
    fred_base = "https://api.stlouisfed.org/fred/series/observations"
    api_key = os.getenv("FRED_API_KEY", "")

    series = {
        "henry_hub_gas": "DHHNGSP",           # Henry Hub Natural Gas Spot (daily)
        "electricity_retail_price": "APU000072610",  # US avg retail electricity price (monthly)
        "electricity_production": "IPG2211A2N",      # Electric power production index
        "core_cpi": "CPIUSLFESL",                    # Core CPI
    }

    results = {}
    for name, series_id in series.items():
        params = {
            "series_id": series_id,
            "file_type": "json",
            "limit": 365,
            "sort_order": "desc",
        }
        if api_key:
            params["api_key"] = api_key
        try:
            resp = requests.get(fred_base, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("observations", [])
            results[name] = [
                {
                    "date": obs["date"],
                    "value": float(obs["value"]) if obs["value"] != "." else None,
                }
                for obs in data
            ]
            print(f"[FRED] {name}: {len(data)} observations")
        except Exception as e:
            print(f"[FRED] ERROR {name}: {e}")
        time.sleep(1)  # Rate limit compliance

    return results


def fetch_eia_grid_data() -> dict:
    """
    Fetch US electricity grid data from EIA.
    Free, EIA_API_KEY env var increases rate limit.
    """
    eia_key = os.getenv("EIA_API_KEY", "")
    base = "https://api.eia.gov/v2"

    endpoints = {
        "generation_by_fuel": {
            "url": f"{base}/electricity/electric-power-operational-data/data/",
            "params": {
                "frequency": "monthly",
                "data[0]": "generation",
                "facets[fueltypeid][]": ["NG", "NUC", "SUN", "WND"],
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": 0,
                "length": 24,
                "api_key": eia_key,
            },
        }
    }

    results = {}
    for name, config in endpoints.items():
        try:
            resp = requests.get(config["url"], params=config["params"], timeout=30)
            resp.raise_for_status()
            results[name] = resp.json().get("response", {}).get("data", [])
            print(f"[EIA] {name}: {len(results[name])} records")
        except Exception as e:
            print(f"[EIA] ERROR {name}: {e}")

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    output_dir = Path("data/raw")

    print("[Financial] Fetching hyperscaler capex from EDGAR XBRL...")
    fetch_all_hyperscaler_capex(output_dir)

    time.sleep(1)

    print("[Financial] Fetching FRED energy indicators...")
    fetch_fred_energy_indicators()

    time.sleep(1)

    print("[Financial] Fetching EIA grid data...")
    fetch_eia_grid_data()

    print("[Financial] Done.")
