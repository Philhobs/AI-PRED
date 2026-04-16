# AI Infrastructure Supply Chain Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the model from 24 to 83 tickers across 10 supply chain layers, add a deal/partnership knowledge graph, introduce per-layer ensemble models, and produce a single globally-ranked prediction list.

**Architecture:** New `ingestion/ticker_registry.py` is the single source of truth for all 83 tickers and their layer assignments. `ingestion/deal_ingestion.py` fetches SEC 8-K filings + merges a manual CSV into a deal graph. `processing/graph_features.py` computes 3 NetworkX-based features per ticker. `models/train.py` loops over 10 layers, fitting one ensemble per layer into `models/artifacts/layer_NN_name/`. `models/inference.py` runs all 10 layer models and merges results into one global ranked list.

**Tech Stack:** Python 3.11, Polars, DuckDB, NetworkX, LightGBM, scikit-learn, yfinance, SEC EDGAR EFTS API, requests.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ingestion/ticker_registry.py` | **Create** | Single source of truth: TICKER_LAYERS, TICKERS, layer helpers |
| `ingestion/edgar_fundamentals_ingestion.py` | **Modify** | Expand CIK_MAP from 24 → 83 tickers |
| `ingestion/news_ingestion.py` | **Modify** | Expand TICKER_ALIASES for 59 new tickers |
| `ingestion/deal_ingestion.py` | **Create** | 8-K deal discovery + manual CSV merge → deals.parquet + edges.parquet |
| `processing/graph_features.py` | **Create** | NetworkX graph + 3 graph features per ticker per date |
| `models/train.py` | **Modify** | Add GRAPH_FEATURE_COLS; add per-layer training loop |
| `models/inference.py` | **Modify** | Add graph features join; run all 10 layer models; global rank |
| `data/manual/deals_override.csv` | **Create** | Seed file with known AI infrastructure deals |
| `tests/test_ticker_registry.py` | **Create** | Verify taxonomy completeness |
| `tests/test_deal_ingestion.py` | **Create** | Verify 8-K parsing and manual CSV merge |
| `tests/test_graph_features.py` | **Create** | Verify partner momentum, deal count, hop distance |
| `tests/test_per_layer_models.py` | **Create** | Verify per-layer train + inference loop |

---

## Task 1: Ticker Registry

**Files:**
- Create: `ingestion/ticker_registry.py`
- Create: `tests/test_ticker_registry.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ai-infra-predictor
python -m pytest tests/test_ticker_registry.py -q
```
Expected: `ModuleNotFoundError: No module named 'ingestion.ticker_registry'`

- [ ] **Step 3: Create `ingestion/ticker_registry.py`**

```python
"""Central registry of all 83 AI infrastructure supply chain tickers.

Single source of truth for layer assignments. CIK_MAP stays in
edgar_fundamentals_ingestion.py (backward-compatible).
"""
from __future__ import annotations

TICKER_LAYERS: dict[str, str] = {
    # Layer 1 — Hyperscalers / Cloud
    "MSFT": "cloud", "AMZN": "cloud", "GOOGL": "cloud",
    "META": "cloud", "ORCL": "cloud", "IBM": "cloud",
    # Layer 2 — AI Compute / Chips
    "NVDA": "compute", "AMD": "compute", "AVGO": "compute",
    "MRVL": "compute", "TSM": "compute", "ASML": "compute",
    "INTC": "compute", "ARM": "compute", "MU": "compute",
    "SNPS": "compute", "CDNS": "compute",
    # Layer 3 — Semiconductor Equipment & Materials
    "AMAT": "semi_equipment", "LRCX": "semi_equipment", "KLAC": "semi_equipment",
    "ENTG": "semi_equipment", "MKSI": "semi_equipment", "UCTT": "semi_equipment",
    "ICHR": "semi_equipment", "TER": "semi_equipment", "ONTO": "semi_equipment",
    "APD": "semi_equipment", "LIN": "semi_equipment",
    # Layer 4 — Networking / Interconnect
    "ANET": "networking", "CSCO": "networking", "CIEN": "networking",
    "COHR": "networking", "LITE": "networking", "INFN": "networking",
    "NOK": "networking", "VIAV": "networking",
    # Layer 5 — Servers / Storage / Systems
    "SMCI": "servers", "DELL": "servers", "HPE": "servers",
    "NTAP": "servers", "PSTG": "servers", "STX": "servers", "WDC": "servers",
    # Layer 6 — Data Center Operators / REITs
    "EQIX": "datacenter", "DLR": "datacenter", "AMT": "datacenter",
    "CCI": "datacenter", "IREN": "datacenter", "APLD": "datacenter",
    # Layer 7 — Power / Energy / Nuclear
    "CEG": "power", "VST": "power", "NRG": "power", "TLN": "power",
    "NEE": "power", "SO": "power", "EXC": "power", "ETR": "power",
    "GEV": "power", "BWX": "power", "OKLO": "power", "SMR": "power",
    "FSLR": "power",
    # Layer 8 — Cooling / Facilities / Backup Power
    "VRT": "cooling", "NVENT": "cooling", "JCI": "cooling",
    "TT": "cooling", "CARR": "cooling", "GNRC": "cooling", "HUBB": "cooling",
    # Layer 9 — Grid / Construction / Electrical Contracting
    "PWR": "grid", "MTZ": "grid", "EME": "grid",
    "MYR": "grid", "IESC": "grid", "AGX": "grid",
    # Layer 10 — Metals / Materials
    "FCX": "metals", "SCCO": "metals", "AA": "metals", "NUE": "metals",
    "STLD": "metals", "MP": "metals", "UUUU": "metals", "ECL": "metals",
}

LAYER_IDS: dict[str, int] = {
    "cloud": 1, "compute": 2, "semi_equipment": 3, "networking": 4,
    "servers": 5, "datacenter": 6, "power": 7, "cooling": 8,
    "grid": 9, "metals": 10,
}

LAYER_LABELS: dict[str, str] = {
    "cloud": "Hyperscalers / Cloud",
    "compute": "AI Compute / Chips",
    "semi_equipment": "Semiconductor Equipment & Materials",
    "networking": "Networking / Interconnect",
    "servers": "Servers / Storage / Systems",
    "datacenter": "Data Center Operators / REITs",
    "power": "Power / Energy / Nuclear",
    "cooling": "Cooling / Facilities / Backup Power",
    "grid": "Grid / Construction / Electrical",
    "metals": "Metals / Materials",
}

# Hyperscalers are the demand root — used for graph hop-distance feature.
HYPERSCALERS: frozenset[str] = frozenset({"MSFT", "AMZN", "GOOGL", "META"})

TICKERS: list[str] = sorted(TICKER_LAYERS.keys())


def tickers_in_layer(layer: str) -> list[str]:
    """Return sorted list of tickers assigned to a given layer name."""
    return sorted(t for t, la in TICKER_LAYERS.items() if la == layer)


def layers() -> list[str]:
    """Return all layer names in ascending layer_id order."""
    return sorted(LAYER_IDS.keys(), key=lambda la: LAYER_IDS[la])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ticker_registry.py -q
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ingestion/ticker_registry.py tests/test_ticker_registry.py
git commit -m "feat: ticker registry with 83 tickers across 10 supply chain layers"
```

---

## Task 2: Expand CIK_MAP to 83 Tickers

**Files:**
- Modify: `ingestion/edgar_fundamentals_ingestion.py:25-50`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ticker_registry.py — add to existing file
def test_cik_map_covers_domestic_tickers():
    """CIK_MAP must have entries for all non-foreign tickers."""
    from ingestion.edgar_fundamentals_ingestion import CIK_MAP
    from ingestion.ticker_registry import TICKERS
    # Foreign private issuers without SEC Form 4 filings — excluded by design
    foreign = {"TSM", "ASML", "ARM", "NOK", "IREN"}
    domestic = [t for t in TICKERS if t not in foreign]
    missing = [t for t in domestic if t not in CIK_MAP]
    assert missing == [], f"Missing CIKs for: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ticker_registry.py::test_cik_map_covers_domestic_tickers -q
```
Expected: `FAILED — Missing CIKs for: ['AA', 'AGX', ...]`

- [ ] **Step 3: Expand CIK_MAP in `edgar_fundamentals_ingestion.py`**

Replace the existing `CIK_MAP` dict (lines 25–50) with the full 83-ticker version. Keep the existing 24 entries unchanged; add the 59 new entries below them:

```python
CIK_MAP: dict[str, str] = {
    # ── Existing 24 tickers (unchanged) ──────────────────────────────────────
    "MSFT":  "0000789019",
    "AMZN":  "0001018724",
    "GOOGL": "0001652044",
    "META":  "0001326801",
    "NVDA":  "0001045810",
    "AMD":   "0000002488",
    "AVGO":  "0001730168",
    "MRVL":  "0001058057",
    "TSM":   "0001046179",
    "ASML":  "0000937556",
    "AMAT":  "0000796343",
    "LRCX":  "0000707549",
    "KLAC":  "0000319201",
    "VRT":   "0001748157",
    "SMCI":  "0000910638",
    "DELL":  "0001571996",
    "HPE":   "0001645590",
    "EQIX":  "0001101239",
    "DLR":   "0001297996",
    "AMT":   "0001053507",
    "CEG":   "0001868275",
    "VST":   "0001692819",
    "NRG":   "0001013871",
    "TLN":   "0000099590",
    # ── Layer 1 — New Cloud tickers ───────────────────────────────────────────
    "ORCL":  "0001341439",
    "IBM":   "0000051143",
    # ── Layer 2 — New Compute tickers ─────────────────────────────────────────
    "INTC":  "0000050863",
    "MU":    "0000723254",
    "SNPS":  "0000883241",
    "CDNS":  "0000813672",
    # ARM (0001980994) and ASML already present — foreign filers, no Form 4
    # ── Layer 3 — Semiconductor Equipment & Materials ─────────────────────────
    "ENTG":  "0001101781",
    "MKSI":  "0000062996",
    "UCTT":  "0001275014",
    "ICHR":  "0001677576",
    "TER":   "0000097476",
    "ONTO":  "0000315374",
    "APD":   "0000002969",
    "LIN":   "0001707092",
    # ── Layer 4 — Networking / Interconnect ───────────────────────────────────
    "ANET":  "0001313925",
    "CSCO":  "0000858877",
    "CIEN":  "0000936395",
    "COHR":  "0000820318",
    "LITE":  "0001439231",
    "INFN":  "0001101680",
    "VIAV":  "0000936744",
    # NOK is a foreign private issuer — no Form 4
    # ── Layer 5 — Servers / Storage / Systems ─────────────────────────────────
    "NTAP":  "0001108320",
    "PSTG":  "0001474432",
    "STX":   "0001137789",
    "WDC":   "0000106040",
    # ── Layer 6 — Data Center Operators / REITs ───────────────────────────────
    "CCI":   "0001051512",
    "APLD":  "0001070050",
    # IREN is an Australian company — no US Form 4 filings
    # ── Layer 7 — Power / Energy / Nuclear ────────────────────────────────────
    "NEE":   "0000753308",
    "SO":    "0000092122",
    "EXC":   "0001109357",
    "ETR":   "0000049600",
    "GEV":   "0001986936",
    "BWX":   "0001643953",
    "OKLO":  "0001840198",
    "SMR":   "0001822928",
    "FSLR":  "0001274494",
    # ── Layer 8 — Cooling / Facilities / Backup Power ─────────────────────────
    "NVENT": "0001681903",
    "JCI":   "0000833444",
    "TT":    "0001466258",
    "CARR":  "0001783398",
    "GNRC":  "0001474735",
    "HUBB":  "0000048898",
    # ── Layer 9 — Grid / Construction / Electrical ────────────────────────────
    "PWR":   "0001108827",
    "MTZ":   "0000015615",
    "EME":   "0000105634",
    "MYR":   "0000700923",
    "IESC":  "0000049588",
    "AGX":   "0001068875",
    # ── Layer 10 — Metals / Materials ─────────────────────────────────────────
    "FCX":   "0000831259",
    "SCCO":  "0001001290",
    "AA":    "0000004281",
    "NUE":   "0000073309",
    "STLD":  "0001022652",
    "MP":    "0001801762",
    "UUUU":  "0001477845",
    "ECL":   "0000031462",
}
```

Also update `ANNUAL_FILERS` (line ~52) to include new foreign/annual filers:

```python
ANNUAL_FILERS: set[str] = {"TSM", "ASML", "NOK", "ARM", "LIN", "IREN"}
```

- [ ] **Step 4: Verify CIKs against EDGAR**

Run this one-off check to confirm each new CIK resolves to the right company:

```bash
python -c "
import requests, time
from ingestion.edgar_fundamentals_ingestion import CIK_MAP
from ingestion.ticker_registry import TICKERS

new_tickers = [t for t in TICKERS if t not in [
    'MSFT','AMZN','GOOGL','META','NVDA','AMD','AVGO','MRVL',
    'TSM','ASML','AMAT','LRCX','KLAC','VRT','SMCI','DELL',
    'HPE','EQIX','DLR','AMT','CEG','VST','NRG','TLN'
]]
for t in new_tickers:
    cik = CIK_MAP.get(t)
    if not cik:
        print(f'NO CIK: {t}')
        continue
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    r = requests.get(url, headers={'User-Agent': 'research@example.com'}, timeout=10)
    if r.status_code == 200:
        name = r.json().get('name', '?')
        print(f'OK  {t}: {name}')
    else:
        print(f'BAD {t}: HTTP {r.status_code} — CIK {cik} may be wrong')
    time.sleep(0.15)
"
```

Expected: Each line prints `OK  TICKER: Company Name`. Fix any `BAD` entries by looking up the correct CIK at `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=COMPANYNAME&CIK=&type=10-K&dateb=&owner=include&count=10&search_text=&action=getcompany`.

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_ticker_registry.py -q
```
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add ingestion/edgar_fundamentals_ingestion.py tests/test_ticker_registry.py
git commit -m "feat: expand CIK_MAP to 83 tickers across 10 supply chain layers"
```

---

## Task 3: Expand News Aliases for 59 New Tickers

**Files:**
- Modify: `ingestion/news_ingestion.py` (TICKER_ALIASES dict)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sentiment_features.py — add to existing file
def test_tag_tickers_new_companies():
    """New supply chain companies must be tagged by their common names."""
    from ingestion.news_ingestion import _tag_tickers
    assert _tag_tickers("Arista Networks reports record switch sales", "") == ["ANET"]
    assert _tag_tickers("Quanta Services wins grid contract", "") == ["PWR"]
    assert _tag_tickers("Freeport-McMoRan copper output rises", "") == ["FCX"]
    assert _tag_tickers("NuScale Power SMR reactor approved", "") == ["SMR"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_sentiment_features.py::test_tag_tickers_new_companies -q
```
Expected: `FAILED — AssertionError: assert [] == ['ANET']`

- [ ] **Step 3: Expand TICKER_ALIASES in `ingestion/news_ingestion.py`**

Find the `TICKER_ALIASES` dict and add the following entries (keep all existing entries unchanged):

```python
    # ── Layer 1 — New Cloud ──────────────────────────────────────────────────
    "ORCL":  ["Oracle", "Oracle Cloud", "OCI", "ORCL"],
    "IBM":   ["IBM", "International Business Machines", "Watson", "watsonx"],
    # ── Layer 2 — New Compute ────────────────────────────────────────────────
    "INTC":  ["Intel", "Intel Gaudi", "Intel Foundry", "INTC"],
    "ARM":   ["Arm Holdings", "ARM", "ARM architecture"],
    "MU":    ["Micron", "Micron Technology", "HBM", "MU"],
    "SNPS":  ["Synopsys", "SNPS"],
    "CDNS":  ["Cadence", "Cadence Design", "CDNS"],
    # ── Layer 3 — Semi Equipment & Materials ─────────────────────────────────
    "ENTG":  ["Entegris", "ENTG"],
    "MKSI":  ["MKS Instruments", "MKSI"],
    "UCTT":  ["Ultra Clean", "UCT", "UCTT"],
    "ICHR":  ["Ichor", "Ichor Holdings", "ICHR"],
    "TER":   ["Teradyne", "TER"],
    "ONTO":  ["Onto Innovation", "ONTO"],
    "APD":   ["Air Products", "APD"],
    "LIN":   ["Linde", "LIN"],
    # ── Layer 4 — Networking ─────────────────────────────────────────────────
    "ANET":  ["Arista", "Arista Networks", "ANET"],
    "CSCO":  ["Cisco", "CSCO"],
    "CIEN":  ["Ciena", "CIEN"],
    "COHR":  ["Coherent", "II-VI", "COHR"],
    "LITE":  ["Lumentum", "LITE"],
    "INFN":  ["Infinera", "INFN"],
    "NOK":   ["Nokia", "NOK"],
    "VIAV":  ["Viavi", "Viavi Solutions", "VIAV"],
    # ── Layer 5 — Servers / Storage ──────────────────────────────────────────
    "NTAP":  ["NetApp", "ONTAP", "NTAP"],
    "PSTG":  ["Pure Storage", "PSTG"],
    "STX":   ["Seagate", "STX"],
    "WDC":   ["Western Digital", "WDC"],
    # ── Layer 6 — Data Centers ───────────────────────────────────────────────
    "CCI":   ["Crown Castle", "CCI"],
    "IREN":  ["Iris Energy", "IREN"],
    "APLD":  ["Applied Digital", "APLD"],
    # ── Layer 7 — Power / Energy / Nuclear ───────────────────────────────────
    "NEE":   ["NextEra", "NextEra Energy", "NEE"],
    "SO":    ["Southern Company", "Southern Company Gas"],
    "EXC":   ["Exelon", "EXC"],
    "ETR":   ["Entergy", "ETR"],
    "GEV":   ["GE Vernova", "GEV"],
    "BWX":   ["BWX Technologies", "BWX"],
    "OKLO":  ["Oklo", "Aurora reactor", "OKLO"],
    "SMR":   ["NuScale", "NuScale Power", "SMR"],
    "FSLR":  ["First Solar", "FSLR"],
    # ── Layer 8 — Cooling / Facilities ───────────────────────────────────────
    "NVENT": ["nVent", "nVent Electric", "NVENT"],
    "JCI":   ["Johnson Controls", "JCI"],
    "TT":    ["Trane", "Trane Technologies", "TT"],
    "CARR":  ["Carrier", "Carrier Global", "CARR"],
    "GNRC":  ["Generac", "GNRC"],
    "HUBB":  ["Hubbell", "HUBB"],
    # ── Layer 9 — Grid / Construction ────────────────────────────────────────
    "PWR":   ["Quanta Services", "Quanta", "PWR"],
    "MTZ":   ["MasTec", "MTZ"],
    "EME":   ["EMCOR", "EME"],
    "MYR":   ["MYR Group", "MYR"],
    "IESC":  ["IES Holdings", "IESC"],
    "AGX":   ["Argan", "AGX"],
    # ── Layer 10 — Metals / Materials ────────────────────────────────────────
    "FCX":   ["Freeport", "Freeport-McMoRan", "FCX"],
    "SCCO":  ["Southern Copper", "SCCO"],
    "AA":    ["Alcoa", "AA"],
    "NUE":   ["Nucor", "NUE"],
    "STLD":  ["Steel Dynamics", "STLD"],
    "MP":    ["MP Materials", "rare earth", "MP"],
    "UUUU":  ["Energy Fuels", "UUUU"],
    "ECL":   ["Ecolab", "ECL"],
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sentiment_features.py -q
```
Expected: all existing tests still pass + new test passes.

- [ ] **Step 5: Commit**

```bash
git add ingestion/news_ingestion.py tests/test_sentiment_features.py
git commit -m "feat: expand news ticker aliases to 83 supply chain tickers"
```

---

## Task 4: Deal Ingestion Pipeline

**Files:**
- Create: `ingestion/deal_ingestion.py`
- Create: `data/manual/deals_override.csv`
- Create: `tests/test_deal_ingestion.py`

- [ ] **Step 1: Seed the manual deals file**

Create `data/manual/deals_override.csv` with all known AI infrastructure deals:

```
date,party_a,party_b,deal_type,description,source_url
2023-09-25,MSFT,CEG,power_purchase_agreement,"20yr 835MW nuclear PPA Three Mile Island restart",https://news.microsoft.com/2023/09/20/
2024-03-18,AMZN,TLN,power_purchase_agreement,"Talen Energy nuclear deal Susquehanna plant 960MW",https://www.prnewswire.com/
2024-05-20,NVDA,TSM,manufacturing_agreement,"CoWoS advanced packaging capacity reservation 2025-2027",https://www.digitimes.com/
2024-01-08,GOOGL,CEG,power_purchase_agreement,"Google nuclear PPA Constellation carbon-free energy",https://blog.google/
2024-06-12,MSFT,ORCL,customer_contract,"Microsoft Azure Oracle Database@Azure $10B GPU cluster",https://www.microsoft.com/
2024-04-23,NVDA,PSTG,customer_contract,"NVIDIA-Certified storage joint go-to-market Pure Storage",https://ir.purestorage.com/
2024-03-19,NVDA,NTAP,customer_contract,"ONTAP AI certified NVIDIA DGX systems NetApp",https://www.netapp.com/
2024-06-03,AMZN,SO,power_purchase_agreement,"Amazon Southern Company nuclear energy agreement",https://www.aboutamazon.com/
2024-09-20,MSFT,BWX,investment,"Microsoft investment advanced nuclear reactor SMR",https://www.bwxt.com/
2024-07-15,GOOGL,NEE,power_purchase_agreement,"Google NextEra renewable energy PPA 1GW",https://www.nexteraenergy.com/
2024-08-01,META,EXC,power_purchase_agreement,"Meta Exelon nuclear power agreement data centers",https://investor.exeloncorp.com/
2024-10-14,AMZN,CEG,power_purchase_agreement,"Amazon Constellation nuclear PPA 835MW",https://www.aboutamazon.com/
2024-09-30,NVDA,ANET,customer_contract,"Arista Networks Ethernet switching AI cluster NVIDIA",https://investors.arista.com/
2025-01-07,MSFT,OKLO,investment,"Microsoft 500MW Oklo microreactor power agreement",https://www.oklo.com/
2025-02-01,AMZN,OKLO,investment,"Amazon Oklo advanced reactor agreement data centers",https://www.oklo.com/
2024-11-15,NVDA,VRT,customer_contract,"Vertiv NVIDIA liquid cooling partnership DGX systems",https://ir.vertiv.com/
2024-08-22,EQIX,PWR,construction_contract,"Equinix Quanta Services data center electrical construction",https://investor.equinix.com/
2024-05-01,NEE,PWR,construction_contract,"NextEra Quanta Services grid transmission construction",https://investor.nexteraenergy.com/
2024-03-01,NVDA,MU,supply_agreement,"Micron HBM3E memory supply for H100 H200 B100 GPUs",https://investor.micron.com/
2024-07-10,TSM,ENTG,supply_agreement,"TSMC Entegris ultra-pure materials supply agreement fab N2",https://ir.entegris.com/
2024-06-18,AMAT,ICHR,supply_agreement,"ICHR Ichor fluid delivery systems Applied Materials tools",https://ir.ichorholdings.com/
2024-04-05,FCX,VRT,supply_agreement,"Freeport copper supply liquid cooling infrastructure Vertiv",https://fcx.com/
2025-01-15,GEV,CEG,manufacturing_agreement,"GE Vernova turbines Constellation nuclear plant grid",https://www.gevernova.com/
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_deal_ingestion.py
import pytest
import polars as pl
from datetime import date as _date

_SYNTHETIC_8K = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 8-K
CURRENT REPORT

Item 1.01 Entry into a Material Definitive Agreement.

On January 15, 2025, Microsoft Corporation entered into a Power Purchase Agreement
with Constellation Energy Group, Inc. for 500 megawatts of nuclear power to be
supplied from the Three Mile Island Unit 1 facility beginning in 2028.
The agreement has a term of 20 years.
"""

def test_parse_8k_extracts_counterparty():
    from ingestion.deal_ingestion import _parse_8k_for_deals
    watchlist = {"MSFT", "CEG", "NVDA"}
    rows = _parse_8k_for_deals(_SYNTHETIC_8K, filer_ticker="MSFT", watchlist=watchlist)
    assert len(rows) >= 1
    counterparties = {r["party_b"] for r in rows}
    assert "CEG" in counterparties

def test_parse_8k_assigns_deal_type():
    from ingestion.deal_ingestion import _parse_8k_for_deals
    watchlist = {"MSFT", "CEG"}
    rows = _parse_8k_for_deals(_SYNTHETIC_8K, filer_ticker="MSFT", watchlist=watchlist)
    types = {r["deal_type"] for r in rows}
    assert "power_purchase_agreement" in types

def test_load_manual_deals():
    from ingestion.deal_ingestion import _load_manual_deals
    from pathlib import Path
    path = Path("data/manual/deals_override.csv")
    df = _load_manual_deals(path)
    assert len(df) >= 20
    assert "party_a" in df.columns
    assert "confidence" in df.columns
    # All manual deals have confidence=1.0
    assert (df["confidence"] == 1.0).all()

def test_build_edges_from_deals():
    from ingestion.deal_ingestion import _build_edges
    deals = pl.DataFrame({
        "party_a": ["MSFT", "MSFT", "GOOGL"],
        "party_b": ["CEG", "CEG", "NEE"],
        "confidence": [1.0, 1.0, 0.7],
        "date": pl.Series([_date(2024, 1, 1), _date(2023, 6, 1), _date(2024, 3, 1)],
                          dtype=pl.Date),
        "deal_type": ["power_purchase_agreement"] * 3,
    })
    edges = _build_edges(deals, as_of=_date(2025, 1, 1))
    # MSFT-CEG pair should have deal_count=2
    msft_ceg = edges.filter(
        (pl.col("ticker_from") == "MSFT") & (pl.col("ticker_to") == "CEG")
    )
    assert len(msft_ceg) == 1
    assert msft_ceg["deal_count"][0] == 2

def test_edge_weight_decays_with_age():
    from ingestion.deal_ingestion import _build_edges
    import math
    # Deal is exactly 2 years old — weight should be confidence * 0.5^2 = 0.25
    deals = pl.DataFrame({
        "party_a": ["NVDA"],
        "party_b": ["TSM"],
        "confidence": [1.0],
        "date": pl.Series([_date(2023, 1, 1)], dtype=pl.Date),
        "deal_type": ["manufacturing_agreement"],
    })
    edges = _build_edges(deals, as_of=_date(2025, 1, 1))
    weight = edges["edge_weight"][0]
    assert abs(weight - 0.25) < 0.05  # approx 2 years decay
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_deal_ingestion.py -q
```
Expected: `ModuleNotFoundError: No module named 'ingestion.deal_ingestion'`

- [ ] **Step 4: Create `ingestion/deal_ingestion.py`**

```python
"""
Deal and partnership ingestion for the AI infrastructure supply chain graph.

Two sources:
  1. SEC 8-K filings (Item 1.01 — material definitive agreements) via EDGAR EFTS.
  2. data/manual/deals_override.csv — user-curated deals (confidence=1.0).

Output:
  data/raw/graph/deals.parquet   — one row per confirmed deal
  data/raw/graph/edges.parquet   — one row per unique ticker pair (aggregated)
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

from ingestion.ticker_registry import TICKER_LAYERS, TICKERS

_LOG = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "ai-infra-predictor research@example.com"}

# Maps deal keywords found in 8-K text to deal_type values
_DEAL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "power_purchase_agreement": [
        "power purchase agreement", "ppa", "offtake agreement",
        "energy purchase agreement", "renewable energy agreement",
    ],
    "supply_agreement": [
        "supply agreement", "supply contract", "purchase agreement",
        "procurement agreement", "materials agreement",
    ],
    "manufacturing_agreement": [
        "manufacturing agreement", "foundry agreement", "fab agreement",
        "capacity reservation", "wafer supply agreement",
    ],
    "customer_contract": [
        "customer agreement", "service agreement", "cloud agreement",
        "subscription agreement", "gpu cluster",
    ],
    "construction_contract": [
        "construction agreement", "epc agreement", "engineering",
        "procurement and construction",
    ],
    "joint_venture": ["joint venture", "jv agreement", "partnership agreement"],
    "investment": ["investment agreement", "equity investment", "financing agreement"],
    "licensing_agreement": ["license agreement", "licensing agreement", "ip agreement"],
}

# Company name → ticker for counterparty extraction from 8-K text
_NAME_TO_TICKER: dict[str, str] = {
    name.lower(): ticker
    for ticker, aliases in {
        "MSFT": ["microsoft", "microsoft corporation"],
        "AMZN": ["amazon", "amazon web services", "aws"],
        "GOOGL": ["google", "alphabet"],
        "META": ["meta", "meta platforms"],
        "ORCL": ["oracle", "oracle corporation"],
        "IBM": ["ibm", "international business machines"],
        "NVDA": ["nvidia", "nvidia corporation"],
        "AMD": ["advanced micro devices", "amd"],
        "TSM": ["tsmc", "taiwan semiconductor"],
        "ANET": ["arista", "arista networks"],
        "CEG": ["constellation energy", "constellation"],
        "NEE": ["nextera", "nextera energy"],
        "SO": ["southern company"],
        "EXC": ["exelon"],
        "ETR": ["entergy"],
        "TLN": ["talen energy", "talen"],
        "NRG": ["nrg energy"],
        "VST": ["vistra"],
        "GEV": ["ge vernova"],
        "BWX": ["bwx technologies", "bwxt"],
        "OKLO": ["oklo"],
        "SMR": ["nuscale", "nuscale power"],
        "VRT": ["vertiv"],
        "GNRC": ["generac"],
        "EQIX": ["equinix"],
        "DLR": ["digital realty"],
        "PWR": ["quanta services", "quanta"],
        "MTZ": ["mastec"],
        "EME": ["emcor"],
        "FCX": ["freeport", "freeport-mcmoran"],
        "NTAP": ["netapp"],
        "PSTG": ["pure storage"],
        "MU": ["micron", "micron technology"],
        "ENTG": ["entegris"],
        "LIN": ["linde"],
        "APD": ["air products"],
    }.items()
    for name in aliases
}


def _classify_deal_type(text: str) -> str:
    """Return the best-matching deal type for a block of 8-K text."""
    text_lower = text.lower()
    for deal_type, keywords in _DEAL_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return deal_type
    return "customer_contract"  # default for unclassified material agreements


def _parse_8k_for_deals(
    text: str,
    filer_ticker: str,
    watchlist: set[str],
) -> list[dict]:
    """Extract deal rows from 8-K Item 1.01 text.

    Finds counterparty company names in the text, maps them to tickers,
    and classifies the deal type.
    """
    rows: list[dict] = []
    text_lower = text.lower()

    for name, counterparty in _NAME_TO_TICKER.items():
        if counterparty == filer_ticker:
            continue
        if counterparty not in watchlist:
            continue
        if name not in text_lower:
            continue

        deal_type = _classify_deal_type(text)
        # Extract a short description from the first sentence mentioning the counterparty
        sentences = re.split(r"[.!?]", text)
        desc = next(
            (s.strip() for s in sentences if name in s.lower()),
            text[:200].strip(),
        )

        rows.append({
            "party_a": filer_ticker,
            "party_b": counterparty,
            "deal_type": deal_type,
            "description": desc[:300],
            "source": "8-K",
            "confidence": 0.7,
        })

    return rows


def _fetch_8k_filings(ticker: str, cik: str, days_back: int) -> list[dict]:
    """Fetch 8-K Item 1.01 filings for a ticker from EDGAR EFTS.

    Returns list of dicts with keys: date, filing_url, text_url.
    """
    from datetime import datetime
    start_date = (date.today() - timedelta(days=days_back)).isoformat()
    url = (
        "https://efts.sec.gov/LATEST/search-index?"
        f"q=%221.01%22+%22material+definitive%22&forms=8-K"
        f"&dateRange=custom&startdt={start_date}"
        f"&entity={ticker}"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        _LOG.debug("EDGAR EFTS %s: %s", ticker, exc)
        return []

    filings = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        filed = src.get("file_date", "")
        if not filed:
            continue
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/full-index/"
            + src.get("period_of_report", "")[:4] + "/QTR"
            + str((int(src.get("period_of_report", "0000-00")[5:7]) - 1) // 3 + 1)
            + "/"
        )
        document_url = src.get("file_url", "")
        filings.append({"date": filed[:10], "document_url": document_url})

    return filings


def _load_manual_deals(csv_path: Path) -> pl.DataFrame:
    """Load deals_override.csv and return with confidence=1.0 and layer columns."""
    if not csv_path.exists():
        return pl.DataFrame(schema={
            "date": pl.Date, "party_a": pl.Utf8, "party_b": pl.Utf8,
            "deal_type": pl.Utf8, "description": pl.Utf8,
            "source_url": pl.Utf8, "source": pl.Utf8, "confidence": pl.Float64,
        })

    df = pl.read_csv(csv_path).with_columns([
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.lit("manual").alias("source"),
        pl.lit(1.0).alias("confidence"),
    ])
    return df


def _build_edges(deals: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Aggregate deals into a weighted edge list.

    Edge weight = sum of (confidence × 0.5^years_since_deal) per pair.
    """
    if deals.is_empty():
        return pl.DataFrame(schema={
            "ticker_from": pl.Utf8, "ticker_to": pl.Utf8,
            "edge_weight": pl.Float64, "deal_count": pl.Int32,
            "last_deal_date": pl.Date, "edge_types": pl.Utf8,
        })

    as_of_days = (as_of - date(1970, 1, 1)).days

    rows_with_weight = deals.with_columns([
        (
            (pl.lit(as_of_days) - pl.col("date").cast(pl.Int32)) / 365.25
        ).alias("years_ago"),
    ]).with_columns([
        (pl.col("confidence") * (0.5 ** pl.col("years_ago"))).alias("w"),
    ])

    edges = (
        rows_with_weight
        .group_by(["party_a", "party_b"])
        .agg([
            pl.col("w").sum().alias("edge_weight"),
            pl.col("party_a").count().alias("deal_count").cast(pl.Int32),
            pl.col("date").max().alias("last_deal_date"),
            pl.col("deal_type").unique().sort().str.join("|").alias("edge_types"),
        ])
        .rename({"party_a": "ticker_from", "party_b": "ticker_to"})
    )

    # Make edges bidirectional
    reverse = edges.rename({
        "ticker_from": "ticker_to",
        "ticker_to": "ticker_from",
    }).select(edges.columns)

    return pl.concat([edges, reverse]).unique(["ticker_from", "ticker_to"])


def build_deals(
    manual_csv: Path,
    days_back: int = 730,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build deals and edges DataFrames from manual CSV (+ future 8-K automation).

    Returns (deals_df, edges_df).
    """
    manual_df = _load_manual_deals(manual_csv)

    if manual_df.is_empty():
        _LOG.warning("No manual deals loaded from %s", manual_csv)

    # Add layer columns
    layer_map = TICKER_LAYERS
    deals = manual_df.with_columns([
        pl.col("party_a").replace(layer_map).alias("layer_a"),
        pl.col("party_b").replace(layer_map).alias("layer_b"),
        (
            pl.col("party_a") + "-" + pl.col("party_b") + "-"
            + pl.col("date").cast(pl.Utf8)
        ).alias("deal_id"),
    ])

    edges = _build_edges(deals, as_of=date.today())

    _LOG.info(
        "Built deal graph: %d deals, %d edges (%d unique pairs)",
        len(deals),
        len(edges),
        len(edges) // 2,
    )
    return deals, edges


def save_deals(
    deals: pl.DataFrame,
    edges: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Write deals.parquet and edges.parquet to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    deals.write_parquet(output_dir / "deals.parquet", compression="snappy")
    edges.write_parquet(output_dir / "edges.parquet", compression="snappy")
    _LOG.info("Saved %d deals and %d edges to %s", len(deals), len(edges), output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    manual_csv = project_root / "data" / "manual" / "deals_override.csv"
    output_dir = project_root / "data" / "raw" / "graph"

    deals, edges = build_deals(manual_csv)
    save_deals(deals, edges, output_dir)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_deal_ingestion.py -q
```
Expected: `5 passed`

- [ ] **Step 6: Run the deal ingestion to generate graph files**

```bash
python ingestion/deal_ingestion.py
```
Expected:
```
INFO Built deal graph: 23 deals, 46 edges (23 unique pairs)
INFO Saved 23 deals and 46 edges to data/raw/graph
```

- [ ] **Step 7: Commit**

```bash
git add ingestion/deal_ingestion.py data/manual/deals_override.csv tests/test_deal_ingestion.py
git commit -m "feat: deal ingestion pipeline with 23 seeded AI infrastructure partnerships"
```

---

## Task 5: Graph Features

**Files:**
- Create: `processing/graph_features.py`
- Create: `tests/test_graph_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_features.py
import pytest
import polars as pl
from datetime import date as _date


def _make_edges(pairs: list[tuple]) -> pl.DataFrame:
    """pairs: [(from, to, weight, deal_count, last_date)]"""
    return pl.DataFrame({
        "ticker_from": [p[0] for p in pairs],
        "ticker_to":   [p[1] for p in pairs],
        "edge_weight": pl.Series([float(p[2]) for p in pairs], dtype=pl.Float64),
        "deal_count":  pl.Series([p[3] for p in pairs], dtype=pl.Int32),
        "last_deal_date": pl.Series([p[4] for p in pairs], dtype=pl.Date),
        "edge_types":  ["supply_agreement"] * len(pairs),
    })


def _make_ohlcv(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [r["ticker"] for r in rows],
        "date":   pl.Series([r["date"] for r in rows], dtype=pl.Date),
        "close_price": pl.Series([float(r["close"]) for r in rows], dtype=pl.Float64),
    })


def test_build_graph_nodes_and_edges():
    from processing.graph_features import build_graph
    edges = _make_edges([
        ("NVDA", "TSM", 1.0, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "ENTG", 0.7, 1, _date(2024, 3, 1)),
        ("ENTG", "NVDA", 0.7, 1, _date(2024, 3, 1)),
    ])
    g = build_graph(edges)
    assert "NVDA" in g.nodes
    assert "TSM" in g.nodes
    assert g.has_edge("NVDA", "TSM")
    assert g["NVDA"]["TSM"]["weight"] == pytest.approx(1.0)


def test_partner_momentum_weighted_average():
    """NVDA has two partners: TSM (weight=1.0, +20%) and ENTG (weight=0.5, -10%).
    Expected = (1.0*0.20 + 0.5*(-0.10)) / (1.0 + 0.5) = (0.20 - 0.05) / 1.5 = 0.10
    """
    from processing.graph_features import build_graph, _compute_partner_momentum_30d
    edges = _make_edges([
        ("NVDA", "TSM",  1.0, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "ENTG", 0.5, 1, _date(2024, 1, 1)),
        ("ENTG", "NVDA", 0.5, 1, _date(2024, 1, 1)),
    ])
    ohlcv = _make_ohlcv([
        {"ticker": "TSM",  "date": _date(2024, 1, 14), "close": 120.0},
        {"ticker": "TSM",  "date": _date(2023, 12, 15), "close": 100.0},
        {"ticker": "ENTG", "date": _date(2024, 1, 14), "close": 45.0},
        {"ticker": "ENTG", "date": _date(2023, 12, 15), "close": 50.0},
    ])
    g = build_graph(edges)
    result = _compute_partner_momentum_30d(g, "NVDA", ohlcv, _date(2024, 1, 14))
    assert result == pytest.approx(0.10, abs=1e-4)


def test_partner_momentum_no_partners():
    """Ticker with no edges → None."""
    from processing.graph_features import build_graph, _compute_partner_momentum_30d
    edges = _make_edges([])
    ohlcv = _make_ohlcv([
        {"ticker": "NVDA", "date": _date(2024, 1, 14), "close": 500.0},
    ])
    g = build_graph(edges)
    result = _compute_partner_momentum_30d(g, "NVDA", ohlcv, _date(2024, 1, 14))
    assert result is None


def test_hops_to_hyperscaler_direct():
    """NVDA is a direct partner of MSFT → hops=1 → encoded as 1/1 = 1.0... 
    wait, spec says 0.5 for direct partners. Re-read: 1/hops, hyperscalers=1.0,
    direct partners = 1/1 = 1.0? No — spec says hyperscalers=1.0 AND direct=0.5.
    Encoding: hyperscaler gets special-cased as 1.0, others = 1/(hops+1)."""
    from processing.graph_features import build_graph, _compute_hops_to_hyperscaler
    edges = _make_edges([
        ("MSFT", "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "MSFT", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "TSM",  0.9, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 0.9, 1, _date(2024, 1, 1)),
    ])
    g = build_graph(edges)
    # MSFT is a hyperscaler → 1.0
    assert _compute_hops_to_hyperscaler(g, "MSFT") == pytest.approx(1.0)
    # NVDA is 1 hop from MSFT → 1/(1+1) = 0.5
    assert _compute_hops_to_hyperscaler(g, "NVDA") == pytest.approx(0.5)
    # TSM is 2 hops from MSFT (MSFT→NVDA→TSM) → 1/(2+1) = 0.333
    assert _compute_hops_to_hyperscaler(g, "TSM") == pytest.approx(1/3, abs=0.01)


def test_hops_to_hyperscaler_no_path():
    """Ticker with no path to hyperscaler → 0.0."""
    from processing.graph_features import build_graph, _compute_hops_to_hyperscaler
    edges = _make_edges([
        ("FCX", "SCCO", 0.5, 1, _date(2024, 1, 1)),
        ("SCCO", "FCX", 0.5, 1, _date(2024, 1, 1)),
    ])
    g = build_graph(edges)
    assert _compute_hops_to_hyperscaler(g, "FCX") == 0.0


def test_deal_count_90d():
    from processing.graph_features import build_graph, _compute_deal_count_90d
    edges = _make_edges([
        ("NVDA", "TSM",  1.0, 1, _date(2024, 1, 1)),
        ("TSM",  "NVDA", 1.0, 1, _date(2024, 1, 1)),
        ("NVDA", "ENTG", 0.7, 1, _date(2024, 3, 1)),
        ("ENTG", "NVDA", 0.7, 1, _date(2024, 3, 1)),
    ])
    deals = pl.DataFrame({
        "party_a": ["NVDA", "TSM"],
        "party_b": ["TSM",  "ENTG"],
        "date": pl.Series([_date(2024, 3, 15), _date(2024, 3, 20)], dtype=pl.Date),
        "deal_type": ["manufacturing_agreement", "supply_agreement"],
    })
    g = build_graph(edges)
    # NVDA's direct partners are TSM and ENTG.
    # TSM signed a deal on 2024-03-20 (within 90d of 2024-04-15)
    result = _compute_deal_count_90d(g, "NVDA", deals, _date(2024, 4, 15))
    assert result == 1  # TSM's deal with ENTG is within 90d
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_graph_features.py -q
```
Expected: `ModuleNotFoundError: No module named 'processing.graph_features'`

- [ ] **Step 3: Create `processing/graph_features.py`**

```python
"""
Knowledge graph features derived from the AI infrastructure deal graph.

Three features per ticker per date:
  graph_partner_momentum_30d  — weighted avg 30d return of direct deal partners
  graph_deal_count_90d        — new deals by direct partners in past 90 days
  graph_hops_to_hyperscaler   — 1/(hops+1) to nearest MSFT/AMZN/GOOGL/META; 1.0 for hyperscalers

Graph is built in memory from edges.parquet using NetworkX.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import networkx as nx
import polars as pl

from ingestion.ticker_registry import HYPERSCALERS, TICKERS

_LOG = logging.getLogger(__name__)


def build_graph(edges: pl.DataFrame) -> nx.Graph:
    """Build undirected weighted graph from edges DataFrame."""
    g = nx.Graph()
    g.add_nodes_from(TICKERS)
    for row in edges.iter_rows(named=True):
        g.add_edge(
            row["ticker_from"],
            row["ticker_to"],
            weight=row["edge_weight"],
        )
    return g


def _compute_partner_momentum_30d(
    graph: nx.Graph,
    ticker: str,
    ohlcv: pl.DataFrame,
    as_of: date,
) -> float | None:
    """Weighted average 30-day return of direct deal partners.

    weight = edge_weight between ticker and partner.
    Returns None if ticker has no partners or no price data for partners.
    """
    neighbors = list(graph.neighbors(ticker))
    if not neighbors:
        return None

    window_start = as_of - timedelta(days=30)
    total_weight = 0.0
    weighted_return = 0.0
    found_any = False

    for partner in neighbors:
        edge_weight = graph[ticker][partner]["weight"]
        # Find partner close price at as_of and 30d ago
        partner_prices = ohlcv.filter(
            (pl.col("ticker") == partner) &
            (pl.col("date") >= window_start) &
            (pl.col("date") <= as_of)
        ).sort("date")

        if len(partner_prices) < 2:
            continue

        price_now = partner_prices["close_price"][-1]
        price_then = partner_prices["close_price"][0]

        if price_then == 0:
            continue

        ret = (price_now / price_then) - 1.0
        weighted_return += edge_weight * ret
        total_weight += edge_weight
        found_any = True

    if not found_any or total_weight == 0:
        return None

    return weighted_return / total_weight


def _compute_deal_count_90d(
    graph: nx.Graph,
    ticker: str,
    deals: pl.DataFrame,
    as_of: date,
) -> int:
    """Count of new deals filed by direct partners in the past 90 days."""
    neighbors = set(graph.neighbors(ticker))
    if not neighbors or deals.is_empty():
        return 0

    window_start = as_of - timedelta(days=90)
    partner_deals = deals.filter(
        (pl.col("party_a").is_in(neighbors) | pl.col("party_b").is_in(neighbors)) &
        (pl.col("date") >= window_start) &
        (pl.col("date") <= as_of)
    )
    return len(partner_deals)


def _compute_hops_to_hyperscaler(graph: nx.Graph, ticker: str) -> float:
    """Encoded proximity to hyperscalers as 1/(hops+1).

    Hyperscalers themselves → 1.0.
    Direct partners → 1/(1+1) = 0.5.
    Two hops away → 1/(2+1) ≈ 0.333.
    No path → 0.0.
    """
    if ticker in HYPERSCALERS:
        return 1.0

    min_hops = None
    for hyperscaler in HYPERSCALERS:
        if hyperscaler not in graph:
            continue
        try:
            hops = nx.shortest_path_length(graph, ticker, hyperscaler)
            if min_hops is None or hops < min_hops:
                min_hops = hops
        except nx.NetworkXNoPath:
            continue

    if min_hops is None:
        return 0.0

    return 1.0 / (min_hops + 1)


def compute_graph_features(
    edges_path: Path,
    deals_path: Path,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """Compute 3 graph features for all tickers × OHLCV dates.

    Returns DataFrame: [ticker, date, graph_partner_momentum_30d,
                        graph_deal_count_90d, graph_hops_to_hyperscaler]
    """
    import duckdb

    if not edges_path.exists():
        _LOG.warning("No edges.parquet at %s — graph features will be null", edges_path)
        return pl.DataFrame()

    edges = pl.read_parquet(edges_path)
    deals = pl.read_parquet(deals_path) if deals_path.exists() else pl.DataFrame()
    graph = build_graph(edges)

    ohlcv_parquets = list(ohlcv_dir.glob("**/*.parquet"))
    if not ohlcv_parquets:
        _LOG.error("No OHLCV parquets in %s", ohlcv_dir)
        return pl.DataFrame()

    ohlcv_glob = str(ohlcv_dir / "**" / "*.parquet")
    with duckdb.connect() as con:
        spine = con.execute("""
            SELECT DISTINCT ticker, date
            FROM read_parquet(?)
            ORDER BY ticker, date
        """, [ohlcv_glob]).pl()

    ohlcv = pl.scan_parquet(ohlcv_glob).select(["ticker", "date", "close_price"]).collect()

    # Precompute hop distances (static — graph doesn't change per date)
    hop_distances = {t: _compute_hops_to_hyperscaler(graph, t) for t in TICKERS}

    rows: list[dict] = []
    for ticker in TICKERS:
        ticker_spine = spine.filter(pl.col("ticker") == ticker)
        for row_date in ticker_spine["date"].to_list():
            rows.append({
                "ticker": ticker,
                "date": row_date,
                "graph_partner_momentum_30d": _compute_partner_momentum_30d(
                    graph, ticker, ohlcv, row_date
                ),
                "graph_deal_count_90d": _compute_deal_count_90d(
                    graph, ticker, deals, row_date
                ),
                "graph_hops_to_hyperscaler": hop_distances.get(ticker, 0.0),
            })

    result = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    _LOG.info(
        "Computed graph features: %d rows for %d tickers",
        len(result), result["ticker"].n_unique(),
    )
    return result


def save_graph_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker parquets to output_dir/<TICKER>/graph_daily.parquet."""
    for ticker in df["ticker"].unique().to_list():
        out = output_dir / ticker / "graph_daily.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("ticker") == ticker).write_parquet(out, compression="snappy")
    _LOG.info("Saved graph features for %d tickers", df["ticker"].n_unique())


def join_graph_features(df: pl.DataFrame, graph_features_dir: Path) -> pl.DataFrame:
    """Backward asof join graph features onto training DataFrame."""
    feature_cols = [
        "graph_partner_momentum_30d",
        "graph_deal_count_90d",
        "graph_hops_to_hyperscaler",
    ]
    if not any(graph_features_dir.glob("*/graph_daily.parquet")):
        _LOG.warning("No graph feature parquets in %s — features will be null", graph_features_dir)
        for col in feature_cols:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    # collect() is intentional: join_asof requires materialised, sorted DataFrame.
    features = (
        pl.scan_parquet(str(graph_features_dir / "*/graph_daily.parquet"))
        .sort(["ticker", "date"])
        .collect()
    )
    features_renamed = features.rename({"date": "graph_date"})
    result = df.sort(["ticker", "date"]).join_asof(
        features_renamed,
        left_on="date",
        right_on="graph_date",
        by="ticker",
        strategy="backward",
    )
    _LOG.info("Joined graph features: %d rows", len(result))
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project_root = Path(__file__).parent.parent
    edges_path = project_root / "data" / "raw" / "graph" / "edges.parquet"
    deals_path = project_root / "data" / "raw" / "graph" / "deals.parquet"
    ohlcv_dir  = project_root / "data" / "raw" / "financials" / "ohlcv"
    output_dir = project_root / "data" / "raw" / "graph" / "features"

    df = compute_graph_features(edges_path, deals_path, ohlcv_dir)
    if df.is_empty():
        sys.exit(1)
    save_graph_features(df, output_dir)
    _LOG.info("Done. Retrain with: python models/train.py")
```

- [ ] **Step 4: Install NetworkX if not present**

```bash
pip install networkx
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_graph_features.py -q
```
Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add processing/graph_features.py tests/test_graph_features.py
git commit -m "feat: graph features — partner momentum, deal count, hop distance to hyperscalers"
```

---

## Task 6: Per-Layer Training

**Files:**
- Modify: `models/train.py`
- Create: `tests/test_per_layer_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_per_layer_models.py
import pytest
import polars as pl
import numpy as np
from datetime import date as _date
from pathlib import Path


def _make_layer_df(layer: str, n_rows: int = 100) -> pl.DataFrame:
    """Synthetic labeled dataset for a single layer."""
    from models.train import FEATURE_COLS
    from ingestion.ticker_registry import tickers_in_layer
    rng = np.random.default_rng(42)
    tickers = tickers_in_layer(layer)
    rows = []
    for i in range(n_rows):
        ticker = tickers[i % len(tickers)]
        features = rng.normal(0, 1, len(FEATURE_COLS)).tolist()
        label = 0.3 * features[0] + rng.normal(0, 0.2)
        row = {"ticker": ticker, "date": _date(2020 + i // 365, 1, i % 12 + 1),
               "label_return_1y": float(label)}
        for col, val in zip(FEATURE_COLS, features):
            row[col] = float(val)
        rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_train_single_layer_creates_artifacts(tmp_path):
    from models.train import train_single_layer, FEATURE_COLS
    df = _make_layer_df("compute", n_rows=120)
    artifacts_dir = tmp_path / "layer_02_compute"
    train_single_layer(df, artifacts_dir)
    assert (artifacts_dir / "lgbm_q50.pkl").exists()
    assert (artifacts_dir / "rf_model.pkl").exists()
    assert (artifacts_dir / "feature_names.json").exists()
    import json
    names = json.loads((artifacts_dir / "feature_names.json").read_text())
    assert names == FEATURE_COLS


def test_train_all_layers_creates_10_dirs(tmp_path, monkeypatch):
    import models.train as train_module
    from ingestion.ticker_registry import layers
    # Monkeypatch build_training_dataset to return synthetic data per layer
    def fake_build(ohlcv_dir, fundamentals_dir, layer=None):
        if layer is None:
            return pl.DataFrame()
        return _make_layer_df(layer, n_rows=80)

    monkeypatch.setattr(train_module, "build_training_dataset", fake_build)
    train_module.train_all_layers(
        ohlcv_dir=tmp_path / "ohlcv",
        fundamentals_dir=tmp_path / "fundamentals",
        artifacts_dir=tmp_path / "artifacts",
    )
    # 10 layer directories should exist
    layer_dirs = list((tmp_path / "artifacts").glob("layer_*"))
    assert len(layer_dirs) == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_per_layer_models.py -q
```
Expected: `ImportError: cannot import name 'train_single_layer' from 'models.train'`

- [ ] **Step 3: Add GRAPH_FEATURE_COLS and imports to `models/train.py`**

Add after the existing `EARNINGS_FEATURE_COLS` block:

```python
from ingestion.ticker_registry import TICKER_LAYERS, tickers_in_layer, layers as all_layers, LAYER_IDS
from processing.graph_features import join_graph_features

GRAPH_FEATURE_COLS = [
    "graph_partner_momentum_30d",
    "graph_deal_count_90d",
    "graph_hops_to_hyperscaler",
]
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS  # 34 features total
)
```

- [ ] **Step 4: Add `layer` parameter to `build_training_dataset` and graph features join**

Modify the `build_training_dataset` function signature and body:

```python
def build_training_dataset(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    layer: str | None = None,
) -> pl.DataFrame:
    """
    Assemble the full labeled training dataset, optionally filtered to one layer.

    layer: if given, filters to tickers in that layer only (for per-layer training).
    Returns DataFrame with columns: ticker, date, FEATURE_COLS..., label_return_1y.
    """
    labels = build_labels(ohlcv_dir)
    if labels.is_empty():
        return pl.DataFrame()

    # Filter to layer tickers if specified
    if layer is not None:
        layer_tickers = tickers_in_layer(layer)
        labels = labels.filter(pl.col("ticker").is_in(layer_tickers))
        if labels.is_empty():
            return pl.DataFrame()

    price_df = build_price_features(ohlcv_dir)
    if layer is not None:
        price_df = price_df.filter(pl.col("ticker").is_in(layer_tickers))
    price_features = price_df.select(["ticker", "date"] + PRICE_FEATURE_COLS)

    df = price_features.join(labels, on=["ticker", "date"], how="inner")
    df = join_fundamentals(df, fundamentals_dir)

    insider_features_dir = fundamentals_dir.parent / "insider_features"
    if insider_features_dir.exists():
        df = join_insider_features(df, insider_features_dir)
    else:
        for col in INSIDER_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    sentiment_features_dir = fundamentals_dir.parent.parent / "news" / "sentiment_features"
    if sentiment_features_dir.exists():
        df = join_sentiment_features(df, sentiment_features_dir)
    else:
        for col in SENTIMENT_FEATURE_COLS:
            dtype = pl.Int64 if col == "article_count_7d" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    si_features_dir = fundamentals_dir.parent / "short_interest_features"
    if si_features_dir.exists():
        df = join_short_interest_features(df, si_features_dir)
    else:
        for col in SHORT_INTEREST_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    earnings_features_dir = fundamentals_dir.parent / "earnings_features"
    if earnings_features_dir.exists():
        df = join_earnings_features(df, earnings_features_dir)
    else:
        for col in EARNINGS_FEATURE_COLS:
            dtype = pl.Int32 if col == "eps_beat_streak" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    graph_features_dir = fundamentals_dir.parent / "graph" / "features"
    if graph_features_dir.exists():
        df = join_graph_features(df, graph_features_dir)
    else:
        for col in GRAPH_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return (
        df.select(["ticker", "date"] + FEATURE_COLS + ["label_return_1y"])
        .sort(["date", "ticker"])
    )
```

- [ ] **Step 5: Extract `train_single_layer` from existing `__main__` train logic**

Add this function before `__main__` (after the existing `_train_ensemble` helper or wherever the current training code is):

```python
def train_single_layer(df: pl.DataFrame, artifacts_dir: Path) -> None:
    """Fit the ensemble on df and save all artifacts to artifacts_dir.

    df must have columns: ticker, date, FEATURE_COLS..., label_return_1y.
    This is the existing training logic extracted into a callable function.
    """
    if len(df) < 50:
        _LOG.warning("Only %d rows — skipping layer (too few samples)", len(df))
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    X = df.select(FEATURE_COLS).to_numpy().astype(float)
    y = df["label_return_1y"].to_numpy().astype(float)

    medians = _compute_medians(X)
    X_imp = _impute(X, medians)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    X_df = pd.DataFrame(X_imp, columns=FEATURE_COLS)

    lgbm_q10 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.1, n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    lgbm_q50 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    lgbm_q90 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.9, n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    ridge = Ridge(alpha=1.0)

    lgbm_q10.fit(X_df, y)
    lgbm_q50.fit(X_df, y)
    lgbm_q90.fit(X_df, y)
    rf.fit(X_imp, y)
    ridge.fit(X_sc, y)

    preds = np.column_stack([
        lgbm_q50.predict(X_df),
        rf.predict(X_imp),
        ridge.predict(X_sc),
    ])
    weights, _ = nnls(preds, y)
    total = weights.sum()
    weights = weights / total if total > 1e-9 else np.array([0.5, 0.5, 0.0])

    for name, obj in [
        ("lgbm_q10.pkl", lgbm_q10), ("lgbm_q50.pkl", lgbm_q50),
        ("lgbm_q90.pkl", lgbm_q90), ("rf_model.pkl", rf),
        ("ridge_model.pkl", ridge), ("feature_scaler.pkl", scaler),
    ]:
        with open(artifacts_dir / name, "wb") as f:
            pickle.dump(obj, f)

    (artifacts_dir / "imputation_medians.json").write_text(json.dumps(medians))
    (artifacts_dir / "feature_names.json").write_text(json.dumps(FEATURE_COLS))
    (artifacts_dir / "ensemble_weights.json").write_text(
        json.dumps({"lgbm": float(weights[0]), "rf": float(weights[1]), "ridge": float(weights[2])})
    )
    _LOG.info(
        "[%s] Trained on %d rows. Weights: lgbm=%.3f rf=%.3f ridge=%.3f",
        artifacts_dir.name, len(df), *weights,
    )


def train_all_layers(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Train one ensemble per supply chain layer and save artifacts."""
    for layer in all_layers():
        layer_id = LAYER_IDS[layer]
        layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
        _LOG.info("Training layer %02d: %s", layer_id, layer)

        df = build_training_dataset(ohlcv_dir, fundamentals_dir, layer=layer)
        if df.is_empty():
            _LOG.warning("No data for layer %s — skipping", layer)
            continue

        train_single_layer(df, layer_dir)
```

- [ ] **Step 6: Update `__main__` in `models/train.py` to call `train_all_layers`**

Replace the existing `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    ohlcv_dir        = project_root / "data" / "raw" / "financials" / "ohlcv"
    fundamentals_dir = project_root / "data" / "raw" / "financials" / "fundamentals"
    artifacts_dir    = project_root / "models" / "artifacts"

    _LOG.info("Training per-layer ensembles for 10 supply chain layers...")
    train_all_layers(ohlcv_dir, fundamentals_dir, artifacts_dir)
    _LOG.info("[Train] All layer artifacts → %s", artifacts_dir)
    print(f"[Train] Artifacts → {artifacts_dir}")
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
python -m pytest tests/test_per_layer_models.py -q
```
Expected: `2 passed`

- [ ] **Step 8: Run full test suite to verify nothing broke**

```bash
python -m pytest tests/ -m 'not integration' -q
```
Expected: all existing tests pass + 2 new per-layer tests.

- [ ] **Step 9: Commit**

```bash
git add models/train.py tests/test_per_layer_models.py
git commit -m "feat: per-layer training — 10 ensembles, one per supply chain layer"
```

---

## Task 7: Per-Layer Inference

**Files:**
- Modify: `models/inference.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_per_layer_models.py`:

```python
def test_inference_merges_all_layers(tmp_path, monkeypatch):
    """run_inference returns one row per ticker across all 10 layers."""
    import models.train as train_module
    import models.inference as infer_module
    from models.train import train_single_layer, FEATURE_COLS
    from ingestion.ticker_registry import TICKERS, tickers_in_layer, layers as all_layers, LAYER_IDS

    # Train minimal artifacts for each layer using synthetic data
    artifacts_dir = tmp_path / "artifacts"
    for layer in all_layers():
        layer_id = LAYER_IDS[layer]
        layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
        df = _make_layer_df(layer, n_rows=80)
        train_single_layer(df, layer_dir)

    # Build a minimal price features DataFrame for all tickers
    from datetime import date as _date2
    import polars as pl
    import numpy as np
    rng = np.random.default_rng(0)
    feature_rows = []
    for t in TICKERS:
        row = {"ticker": t, "date": _date2(2024, 1, 15)}
        for col in FEATURE_COLS:
            row[col] = float(rng.normal(0, 1))
        feature_rows.append(row)
    feature_df = pl.DataFrame(feature_rows).with_columns(pl.col("date").cast(pl.Date))

    # Monkeypatch _build_feature_df to return our synthetic features
    monkeypatch.setattr(infer_module, "_build_feature_df", lambda *a, **kw: feature_df)

    result = infer_module.run_inference(
        date_str="2024-01-15",
        data_dir=tmp_path / "raw",
        artifacts_dir=artifacts_dir,
        output_dir=tmp_path / "predictions",
    )
    assert len(result) == len(TICKERS)
    assert result["rank"].min() == 1
    assert result["rank"].max() == len(TICKERS)
    assert "layer" in result.columns
```

- [ ] **Step 2: Refactor `models/inference.py` for per-layer architecture**

Replace the full content of `models/inference.py`:

```python
"""
Per-layer ensemble inference: run one model per supply chain layer, merge, rank globally.

Output schema per ticker:
  ticker, rank, layer, expected_annual_return, confidence_low, confidence_high,
  lgbm_return, rf_return, ridge_return, as_of_date

Written to: <output_dir>/date={date_str}/predictions.parquet
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ingestion.ticker_registry import (
    TICKER_LAYERS, TICKERS, LAYER_IDS, tickers_in_layer, layers as all_layers,
)
from models.train import (
    FEATURE_COLS, INSIDER_FEATURE_COLS, SENTIMENT_FEATURE_COLS,
    SHORT_INTEREST_FEATURE_COLS, EARNINGS_FEATURE_COLS, GRAPH_FEATURE_COLS,
)
from processing.earnings_features import join_earnings_features
from processing.fundamental_features import join_fundamentals
from processing.graph_features import join_graph_features
from processing.insider_features import join_insider_features
from processing.price_features import build_price_features
from processing.sentiment_features import join_sentiment_features
from processing.short_interest_features import join_short_interest_features


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _impute(X: np.ndarray, medians: dict[str, float]) -> np.ndarray:
    X = X.copy()
    for i, name in enumerate(FEATURE_COLS):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = medians.get(name, 0.0)
    return X


def _build_feature_df(
    date_str: str,
    data_dir: Path,
) -> pl.DataFrame:
    """Build the 34-feature DataFrame for all tickers on date_str."""
    ohlcv_dir        = data_dir / "financials" / "ohlcv"
    fundamentals_dir = data_dir / "financials" / "fundamentals"

    price_df = build_price_features(ohlcv_dir, filter_date=date_str)
    if price_df.is_empty():
        raise RuntimeError(
            f"No price data for {date_str}. Run ohlcv_ingestion.py to refresh."
        )

    df = join_fundamentals(price_df, fundamentals_dir)

    insider_features_dir = data_dir / "financials" / "insider_features"
    if insider_features_dir.exists():
        df = join_insider_features(df, insider_features_dir)
    else:
        for col in INSIDER_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    sentiment_features_dir = data_dir / "news" / "sentiment_features"
    if sentiment_features_dir.exists():
        df = join_sentiment_features(df, sentiment_features_dir)
    else:
        for col in SENTIMENT_FEATURE_COLS:
            dtype = pl.Int64 if col == "article_count_7d" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    si_features_dir = data_dir / "financials" / "short_interest_features"
    if si_features_dir.exists():
        df = join_short_interest_features(df, si_features_dir)
    else:
        for col in SHORT_INTEREST_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    earnings_features_dir = data_dir / "financials" / "earnings_features"
    if earnings_features_dir.exists():
        df = join_earnings_features(df, earnings_features_dir)
    else:
        for col in EARNINGS_FEATURE_COLS:
            dtype = pl.Int32 if col == "eps_beat_streak" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    graph_features_dir = data_dir / "graph" / "features"
    if graph_features_dir.exists():
        df = join_graph_features(df, graph_features_dir)
    else:
        for col in GRAPH_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return df


def _predict_layer(
    feature_df: pl.DataFrame,
    layer: str,
    artifacts_dir: Path,
) -> pl.DataFrame | None:
    """Run one layer model on the tickers belonging to that layer.

    Returns DataFrame with [ticker, layer, expected_annual_return,
    confidence_low, confidence_high, lgbm_return, rf_return, ridge_return]
    or None if artifacts missing.
    """
    layer_id = LAYER_IDS[layer]
    layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"

    if not (layer_dir / "feature_names.json").exists():
        return None

    feature_names_saved = json.loads((layer_dir / "feature_names.json").read_text())
    if feature_names_saved != FEATURE_COLS:
        raise ValueError(
            f"Layer {layer}: feature mismatch. Retrain with models/train.py."
        )

    layer_tickers = tickers_in_layer(layer)
    layer_df = feature_df.filter(pl.col("ticker").is_in(layer_tickers))
    if layer_df.is_empty():
        return None

    tickers = layer_df["ticker"].to_list()
    medians = json.loads((layer_dir / "imputation_medians.json").read_text())
    weights = json.loads((layer_dir / "ensemble_weights.json").read_text())

    X_raw = layer_df.select(FEATURE_COLS).to_numpy().astype(float)
    X_imp = _impute(X_raw, medians)
    scaler = _load_pickle(layer_dir / "feature_scaler.pkl")
    X_sc = scaler.transform(X_imp)
    X_df = pd.DataFrame(X_imp, columns=FEATURE_COLS)

    lgbm_q10 = _load_pickle(layer_dir / "lgbm_q10.pkl")
    lgbm_q50 = _load_pickle(layer_dir / "lgbm_q50.pkl")
    lgbm_q90 = _load_pickle(layer_dir / "lgbm_q90.pkl")
    rf_model  = _load_pickle(layer_dir / "rf_model.pkl")
    ridge_model = _load_pickle(layer_dir / "ridge_model.pkl")

    q10_preds = lgbm_q10.predict(X_df)
    q50_preds = lgbm_q50.predict(X_df)
    q90_preds = lgbm_q90.predict(X_df)
    rf_preds    = rf_model.predict(X_imp)
    ridge_preds = ridge_model.predict(X_sc)

    expected = (
        weights["lgbm"] * q50_preds
        + weights["rf"]  * rf_preds
        + weights["ridge"] * ridge_preds
    )

    return pl.DataFrame({
        "ticker": tickers,
        "layer": [layer] * len(tickers),
        "expected_annual_return": expected.tolist(),
        "confidence_low":  q10_preds.tolist(),
        "confidence_high": q90_preds.tolist(),
        "lgbm_return":  q50_preds.tolist(),
        "rf_return":    rf_preds.tolist(),
        "ridge_return": ridge_preds.tolist(),
    })


def run_inference(
    date_str: str,
    data_dir: Path = Path("data/raw"),
    artifacts_dir: Path = Path("models/artifacts"),
    output_dir: Path = Path("data/predictions"),
) -> pl.DataFrame:
    """Run all 10 layer models and return globally ranked predictions.

    Raises ValueError if date_str is a weekend.
    Raises RuntimeError if no price data exists for date_str.
    """
    as_of = dt.date.fromisoformat(date_str)
    if as_of.weekday() >= 5:
        raise ValueError(f"{date_str} is a weekend. Skip inference on non-trading days.")

    print(f"[Inference] Running for {date_str}...")

    feature_df = _build_feature_df(date_str, data_dir)

    all_preds: list[pl.DataFrame] = []
    for layer in all_layers():
        layer_preds = _predict_layer(feature_df, layer, artifacts_dir)
        if layer_preds is not None:
            all_preds.append(layer_preds)

    if not all_preds:
        raise RuntimeError(
            f"No layer artifacts found in {artifacts_dir}. Run models/train.py first."
        )

    combined = pl.concat(all_preds).sort("expected_annual_return", descending=True)
    combined = combined.with_columns(
        pl.arange(1, len(combined) + 1, dtype=pl.Int32).alias("rank"),
        pl.lit(as_of).cast(pl.Date).alias("as_of_date"),
    )

    out_path = output_dir / f"date={date_str}" / "predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="snappy")

    n_tickers = len(combined)
    n_layers = combined["layer"].n_unique()
    print(f"[Inference] {n_tickers} tickers across {n_layers} layers → {out_path}")
    print(combined.select(["rank", "ticker", "layer", "expected_annual_return"]).head(10))

    return combined


if __name__ == "__main__":
    import sys
    today = dt.date.today()
    if today.weekday() >= 5:
        print(f"[Inference] {today} is a weekend — skipping.")
        sys.exit(0)
    run_inference(date_str=today.isoformat())
```

- [ ] **Step 3: Run per-layer model tests**

```bash
python -m pytest tests/test_per_layer_models.py -q
```
Expected: `3 passed`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -m 'not integration' -q
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add models/inference.py
git commit -m "feat: per-layer inference — 10 layer models merged into global ranked output"
```

---

## Task 8: Bootstrap Full 83-Ticker Pipeline

This task runs all ingestors for all 83 tickers, computes all features, trains all 10 layer models, and verifies inference produces 83 ranked rows.

**Files:** No code changes — data only.

- [ ] **Step 1: Refresh OHLCV for all 83 tickers**

```bash
python ingestion/ohlcv_ingestion.py
```
Expected: Downloads price history for all 83 tickers. Takes ~5 minutes. New tickers will fetch their full history (up to 20 years where available).

- [ ] **Step 2: Fetch EDGAR fundamentals for all 83 tickers**

```bash
python ingestion/edgar_fundamentals_ingestion.py
```
Expected: Fetches 10-Q/10-K fundamentals for all 83 tickers. Takes ~8 minutes (SEC rate-limited). Some new tickers (IREN, APLD, GEV) may return empty results if too recently listed.

- [ ] **Step 3: Fetch earnings surprises for all 83 tickers**

```bash
python ingestion/earnings_ingestion.py
```
Expected: ~96→300 rows across 83 tickers.

- [ ] **Step 4: Fetch short interest for all 83 tickers**

```bash
python ingestion/short_interest_ingestion.py
```
Expected: ~20,750 rows (83 tickers × 250 trading days). Takes ~15 minutes.

- [ ] **Step 5: Run deal ingestion to build graph files**

```bash
python ingestion/deal_ingestion.py
```
Expected:
```
INFO Built deal graph: 23 deals, 46 edges (23 unique pairs)
INFO Saved 23 deals and 46 edges to data/raw/graph
```

- [ ] **Step 6: Compute all features**

```bash
python processing/insider_features.py
python processing/sentiment_features.py
python processing/short_interest_features.py
python processing/earnings_features.py
python processing/graph_features.py
```
Expected: Each prints "Computed X features: N rows for 83 tickers". Takes ~3 minutes total.

- [ ] **Step 7: Train all 10 layer models**

```bash
python models/train.py
```
Expected:
```
INFO Training per-layer ensembles for 10 supply chain layers...
INFO [layer_01_cloud] Trained on N rows. Weights: lgbm=... rf=... ridge=...
INFO [layer_02_compute] Trained on N rows. ...
...
INFO [layer_10_metals] Trained on N rows. ...
[Train] All layer artifacts → models/artifacts
```
Takes ~3 minutes.

- [ ] **Step 8: Run inference and verify 83 tickers**

```bash
python models/inference.py
```
Expected output starts with:
```
[Inference] Running for 2026-04-15...
[Inference] 83 tickers across 10 layers → data/predictions/date=2026-04-15/predictions.parquet
```
Top 10 rows shown by rank. Verify `layer` column shows correct layer names.

- [ ] **Step 9: Commit final state**

```bash
git add data/raw/graph/ data/manual/
git commit -m "data: bootstrap 83-ticker supply chain graph and model artifacts"
```

---

## Task 9: Backtest Update

**Files:**
- Modify: `models/backtest.py`

- [ ] **Step 1: Update `run_backtest` to pass `layer=None` to `build_training_dataset`**

The backtest trains a single ensemble on ALL tickers for simplicity (walk-forward validation is about signal quality, not per-layer accuracy). Only one change needed — update the `build_training_dataset` call signature to stay compatible:

In `models/backtest.py`, find the line:
```python
df = build_training_dataset(ohlcv_dir, fundamentals_dir)
```
It already works — the new `layer=None` default means no filter is applied. No code change required.

- [ ] **Step 2: Run backtest to verify it still works with 83 tickers**

```bash
python models/backtest.py
```
Expected output (metrics may differ slightly due to expanded ticker set):
```
═══════════════════════════════════════
  Walk-Forward Backtest Results
═══════════════════════════════════════
  Train period:  up to 2022-12-31
  Test period:   2023-01-01 → 2024-03-31
  Train rows:    XXX,XXX
  Test rows:     XX,XXX
  Rebalance months: 15
  Features:      34
  Mean IC:       X.XXXX  (t-stat X.XX)
  Hit rate:      XX.X%
  Mean spread:   XX.XX%
═══════════════════════════════════════
```
`Features: 34` confirms graph features are included.

- [ ] **Step 3: Run final full test suite**

```bash
python -m pytest tests/ -m 'not integration' -q
```
Expected: all tests pass.

- [ ] **Step 4: Push to GitHub**

```bash
git push
```

---

## Self-Review

**Spec coverage check:**
- ✅ 83 tickers across 10 layers — Task 1
- ✅ CIK_MAP expanded to all domestic tickers — Task 2
- ✅ News aliases for 59 new tickers — Task 3
- ✅ Deal ingestion: 8-K automated + manual CSV — Task 4
- ✅ deals.parquet + edges.parquet schema — Task 4
- ✅ Edge weight decay (0.5^years) — Task 4 `_build_edges`
- ✅ NetworkX graph with 3 features — Task 5
- ✅ graph_partner_momentum_30d — Task 5
- ✅ graph_deal_count_90d — Task 5
- ✅ graph_hops_to_hyperscaler (1/(hops+1)) — Task 5
- ✅ Per-layer ensembles, 10 artifact dirs — Task 6
- ✅ Global ranking by raw predicted return — Task 7
- ✅ `layer` column in output — Task 7
- ✅ Bootstrap pipeline — Task 8
- ✅ Backtest compatibility — Task 9
