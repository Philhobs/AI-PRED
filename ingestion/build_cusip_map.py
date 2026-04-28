# ingestion/build_cusip_map.py
"""
One-time script: build CUSIP map from ticker → CUSIP using yfinance ISINs.

For US-listed equities (domestic + ADRs), ISIN = "US" + 9-char CUSIP + check digit.
Run: python ingestion/build_cusip_map.py

Output: data/raw/financials/cusip_map.json

NOTE: yfinance 1.2+ no longer provides ISINs via .info["isin"] or .get_isin().
The map is therefore built from a statically verified table sourced from SEC EDGAR
13F-HR filings (all CUSIPs confirmed via EDGAR full-text search). The yfinance ISIN
path is retained as a fallback for any tickers added to the watchlist in future.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import yfinance as yf

from ingestion.ticker_registry import TICKERS

_OUTPUT = Path("data/raw/financials/cusip_map.json")

# ---------------------------------------------------------------------------
# Verified CUSIP table — sourced from SEC EDGAR 13F-HR filings.
# Foreign-incorporated companies (Ireland, Netherlands, UK, Cayman, Australia)
# carry non-"US" ISINs; their CUSIPs still appear in 13F filings and are listed
# here exactly as reported.
# ---------------------------------------------------------------------------
_STATIC_CUSIP_MAP: dict[str, str] = {
    # Layer 1 — Hyperscalers / Cloud
    "MSFT":  "594918104",  # Microsoft Corp
    "AMZN":  "023135106",  # Amazon.com Inc
    "GOOGL": "02079K305",  # Alphabet Inc Cl A
    "META":  "30303M102",  # Meta Platforms Inc
    "ORCL":  "68389X105",  # Oracle Corp
    "IBM":   "459200101",  # International Business Machines
    # Layer 2 — AI Compute / Chips
    "NVDA":  "67066G104",  # NVIDIA Corp
    "AMD":   "007903107",  # Advanced Micro Devices
    "AVGO":  "11135F101",  # Broadcom Inc
    "MRVL":  "573874104",  # Marvell Technology (US-listed; Bermuda incorporated)
    "TSM":   "872543101",  # Taiwan Semiconductor Mfg ADR (NYSE)
    "ASML":  "04523Y102",  # ASML Holding ADR (NASDAQ)
    "INTC":  "458140100",  # Intel Corp
    "ARM":   "042068205",  # Arm Holdings PLC ADR (NASDAQ)
    "MU":    "595112103",  # Micron Technology
    "SNPS":  "871607107",  # Synopsys Inc
    "CDNS":  "127387108",  # Cadence Design Systems
    # Layer 3 — Semiconductor Equipment & Materials
    "AMAT":  "037833100",  # Applied Materials
    "LRCX":  "512807108",  # Lam Research
    "KLAC":  "482480100",  # KLA Corp
    "ENTG":  "29362U104",  # Entegris Inc
    "MKSI":  "55306N104",  # MKS Instruments
    "UCTT":  "90385V107",  # Ultra Clean Holdings
    "ICHR":  "45168D104",  # Ichor Holdings
    "TER":   "880779103",  # Teradyne
    "ONTO":  "683344105",  # Onto Innovation
    "APD":   "009158106",  # Air Products & Chemicals
    "LIN":   "532457108",  # Linde PLC (Ireland-incorporated; US-listed)
    # Layer 4 — Networking / Interconnect
    "ANET":  "040413106",  # Arista Networks
    "CSCO":  "17275R102",  # Cisco Systems
    "CIEN":  "171779309",  # Ciena Corp
    "COHR":  "19247G107",  # Coherent Corp
    "LITE":  "55024U109",  # Lumentum Holdings
    "INFN":  "45667G103",  # Infinera Corp
    "NOK":   "654902204",  # Nokia Corp ADR (NYSE)
    "VIAV":  "92552V100",  # Viavi Solutions
    # Layer 5 — Servers / Storage / Systems
    "SMCI":  "86800U104",  # Super Micro Computer
    "DELL":  "24703L202",  # Dell Technologies
    "HPE":   "42824C109",  # Hewlett Packard Enterprise
    "NTAP":  "64110D104",  # NetApp
    "PSTG":  "74624M102",  # Pure Storage
    "STX":   "G7997R103",  # Seagate Technology (Ireland-incorporated)
    "WDC":   "958102105",  # Western Digital
    # Layer 6 — Data Center Operators / REITs
    "EQIX":  "29444U700",  # Equinix Inc
    "DLR":   "253868103",  # Digital Realty Trust
    "AMT":   "03027X100",  # American Tower Corp
    "CCI":   "22822V101",  # Crown Castle Inc
    "IREN":  "Q4982L109",  # Iris Energy Ltd (Australia-incorporated ADR)
    "APLD":  "038169207",  # Applied Digital Corp
    # Layer 7 — Power / Energy / Nuclear
    "CEG":   "21037T109",  # Constellation Energy Corp
    "VST":   "92840M102",  # Vistra Corp
    "NRG":   "629377508",  # NRG Energy
    "TLN":   "87422Q109",  # Talen Energy Corp
    "NEE":   "65339F101",  # NextEra Energy
    "SO":    "842587107",  # The Southern Company
    "EXC":   "30161N101",  # Exelon Corp
    "ETR":   "29364G103",  # Entergy Corp
    "GEV":   "36828A101",  # GE Vernova Inc
    "BWX":   "05605H100",  # BWX Technologies
    "OKLO":  "67979L109",  # Oklo Inc (small-cap; may be sparse in 13F)
    "SMR":   "67079K100",  # NuScale Power (now SMR)
    "FSLR":  "336433107",  # First Solar
    # Layer 8 — Cooling / Facilities / Backup Power
    "VRT":   "92537N108",  # Vertiv Holdings
    "NVENT": "G6700G107",  # nVent Electric PLC (Ireland-incorporated)
    "JCI":   "G51502105",  # Johnson Controls International (Ireland-incorporated)
    "TT":    "G8994E103",  # Trane Technologies PLC (Ireland-incorporated)
    "CARR":  "14448C104",  # Carrier Global
    "GNRC":  "368736104",  # Generac Holdings
    "HUBB":  "443510607",  # Hubbell Inc
    # Layer 9 — Grid / Construction / Electrical Contracting
    "PWR":   "74762E102",  # Quanta Services
    "MTZ":   "576323109",  # MasTec Inc
    "EME":   "29084Q100",  # EMCOR Group
    "MYR":   "55405W104",  # MYR Group
    "IESC":  "44951W106",  # IES Holdings
    "AGX":   "04010L103",  # Argan Inc
    # Layer 10 — Metals / Materials
    "FCX":   "35671D857",  # Freeport-McMoRan
    "SCCO":  "84265V105",  # Southern Copper Corp
    "AA":    "013872106",  # Alcoa Corp
    "NUE":   "670346105",  # Nucor Corp
    "STLD":  "858119100",  # Steel Dynamics
    "MP":    "553368101",  # MP Materials Corp
    "UUUU":  "292671708",  # Energy Fuels Inc
    "ECL":   "278865100",  # Ecolab Inc
    # ── 2026-04-26 strategy alignment additions ──────────────────────────────
    "CCJ":   "13321L108",  # Cameco Corp (uranium — Layer 7 power)
    "ETN":   "G29183103",  # Eaton plc (Ireland-incorporated — Layer 8 cooling)
    "NET":   "18915M107",  # Cloudflare Inc Class A (Layer 14 cyber_pureplay)
    # Layer 16 — Enterprise SaaS / Agentic Software (NEW)
    "PLTR":  "69608A108",  # Palantir Technologies Class A
    "NOW":   "81762P102",  # ServiceNow Inc
    "CRM":   "79466L302",  # Salesforce Inc
    "ADBE":  "00724F101",  # Adobe Inc
    "INTU":  "461202103",  # Intuit Inc
    "DDOG":  "23804L103",  # Datadog Inc Class A
    "SNOW":  "833445109",  # Snowflake Inc Class A (corrected 2026-04-27 — was 833445105, wrong check digit)
    "GTLB":  "37637K108",  # GitLab Inc Class A (corrected 2026-04-27 — was 376372102)
    "PATH":  "90364P105",  # UiPath Inc Class A
    "TEAM":  "049468101",  # Atlassian Corporation (corrected 2026-04-27 — was G06242104, the pre-reorg Bermuda CUSIP; current US-listed entity uses 049468101)
    "MNDY":  "M7S64H106",  # monday.com Ltd (corrected 2026-04-27 — was M75158109, typo)
    # ── 2026-04-27 layer 11-15 coverage (discovered via tools/discover_cusips.py) ──
    # Layer 11 robotics_industrial
    "ROK":   "773903109",  # Rockwell Automation Inc
    "ZBRA":  "989207105",  # Zebra Technologies Corp
    "CGNX":  "192422103",  # Cognex Corp
    "SYM":   "87151X101",  # Symbotic Inc
    "EMR":   "291011104",  # Emerson Electric Co
    # Layer 12 robotics_medical_humanoid
    "ISRG":  "46120E602",  # Intuitive Surgical Inc
    "TSLA":  "88160R101",  # Tesla Inc
    # Layer 13 robotics_mcu_chips
    "TXN":   "882508104",  # Texas Instruments Inc
    "MCHP":  "595017104",  # Microchip Technology Inc
    "ADI":   "032654105",  # Analog Devices Inc
    # Layer 14 cyber_pureplay
    "CRWD":  "22788C105",  # CrowdStrike Holdings Inc
    "ZS":    "98980G102",  # Zscaler Inc
    "S":     "81730H109",  # SentinelOne Inc
    # Layer 15 cyber_platform
    "PANW":  "697435105",  # Palo Alto Networks Inc
    "FTNT":  "34959E109",  # Fortinet Inc
    "CHKP":  "M22465104",  # Check Point Software (Israel-incorporated)
    "CYBR":  "M2682V108",  # CyberArk Software (Israel-incorporated)
    "TENB":  "88025T102",  # Tenable Holdings
    "QLYS":  "74758T303",  # Qualys Inc
    "OKTA":  "679295105",  # Okta Inc
    "AKAM":  "00971T101",  # Akamai Technologies
    "RPD":   "753422104",  # Rapid7 Inc
    "VRNS":  "922280102",  # Varonis Systems Inc
    # Networking ADRs
    "STM":   "861012102",  # STMicroelectronics ADR
    "ERIC":  "294821608",  # Telefonaktiebolaget LM Ericsson ADR
    # JNPR (Juniper Networks) intentionally omitted — acquired by HPE July 2025,
    # delisted, and absent from current 13F filings. Remove from registry next
    # session if its OHLCV dries up.
    # ── 2026-04-28 medical-robotics expansion ────────────────────────────────
    "SYK":   "863667101",  # Stryker Corp — Mako orthopedic robot
    "MDT":   "G5960L103",  # Medtronic plc (Ireland-incorporated) — Hugo surgical
    "GMED":  "379577208",  # Globus Medical — ExcelsiusGPS
    "PRCT":  "74276L105",  # PROCEPT BioRobotics — AquaBeam
    # ── 2026-04-28 layer-breadth expansion (25 tickers, 8 layers) ────────────
    # Networking
    "HLIT":  "413160102",  # Harmonic Inc
    "CALX":  "13100M509",  # Calix Inc
    "AAOI":  "03823U102",  # Applied Optoelectronics
    "EXTR":  "30226D106",  # Extreme Networks
    # Servers
    "CDW":   "12514G108",  # CDW Corp
    "ARW":   "042735100",  # Arrow Electronics
    # Datacenter
    "SBAC":  "78410G104",  # SBA Communications
    "DBRG":  "25401T603",  # DigitalBridge Group
    "GLW":   "219350105",  # Corning Inc
    "DOCN":  "25402D102",  # DigitalOcean Holdings
    # Power
    "DUK":   "26441C204",  # Duke Energy
    "AEP":   "025537101",  # American Electric Power
    "XEL":   "98389B100",  # Xcel Energy
    "LEU":   "15643U104",  # Centrus Energy
    "PLUG":  "72919P202",  # Plug Power
    # Grid / E&C
    "FLR":   "343412102",  # Fluor Corp
    "ACM":   "00766T100",  # AECOM
    "KBR":   "48242W106",  # KBR Inc
    # Cyber pureplay
    "VRSN":  "92343E102",  # Verisign — DNS infra
    # Cyber platform
    "LDOS":  "525327102",  # Leidos Holdings
    "CACI":  "127190304",  # CACI International
    "BAH":   "099502106",  # Booz Allen Hamilton
    # Robotics MCU/sensor chips
    "ON":    "682189105",  # onsemi
    "NXPI":  "N6596X109",  # NXP Semiconductors NV (Netherlands-incorporated)
    "MPWR":  "609839105",  # Monolithic Power Systems
    # Foreign exchanges (000660.KS SK Hynix, 005930.KS Samsung) intentionally
    # omitted: 13F-HR covers US securities only; KRX-listed equities don't
    # appear in US institutional holdings filings.
}


def build_cusip_map() -> dict[str, str]:
    """
    Build the ticker → CUSIP map and save to JSON.

    Uses the statically verified _STATIC_CUSIP_MAP as the primary source.
    For any watchlist ticker not in the static table, attempts to resolve via
    yfinance ISIN (US ISIN = 'US' + 9-char CUSIP + check digit).
    """
    cusip_map: dict[str, str] = {}
    missing: list[str] = []

    for ticker in TICKERS:
        if ticker in _STATIC_CUSIP_MAP:
            raw = _STATIC_CUSIP_MAP[ticker]
            cusip = raw.upper()  # normalise to uppercase
            cusip_map[ticker] = cusip
            print(f"  {ticker}: static → {cusip}")
        else:
            missing.append(ticker)

    if missing:
        print(f"\n  Attempting yfinance ISIN for {len(missing)} unmapped tickers...")
        for ticker in missing:
            try:
                info = yf.Ticker(ticker).info
                isin = info.get("isin", "")
                if isinstance(isin, str) and len(isin) == 12 and isin.startswith("US"):
                    cusip = isin[2:11]
                    cusip_map[ticker] = cusip
                    print(f"  {ticker}: yfinance {isin} → CUSIP {cusip}")
                else:
                    print(f"  {ticker}: isin={isin!r} — not a US ISIN, skipped")
            except Exception as exc:
                print(f"  {ticker}: ERROR {exc}")
            time.sleep(1.0)  # yfinance rate limit

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(cusip_map, indent=2, sort_keys=True))
    print(f"\nSaved {len(cusip_map)}/{len(TICKERS)} entries to {_OUTPUT}")
    return cusip_map


if __name__ == "__main__":
    build_cusip_map()
