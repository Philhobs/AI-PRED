import feedparser
import re
import requests
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

AI_INFRA_THEMES = [
    "TECH_SEMICONDUCTORS",
    "ENERGY_NUCLEAR",
    "ENERGY_ELECTRICITY",
    "SANCTION",
    "EXPORT_CONTROL",
    "TAX_DISPUTE",
    "ENV_CARBONCAP",
    "CRISISLEX_T04_ELECTRICITY",
]

RSS_FEEDS = {
    "datacenter_dynamics": "https://www.datacenterdynamics.com/feed/",
    "semiconductor_engineering": "https://semiengineering.com/feed/",
    "power_magazine": "https://www.powermag.com/feed/",
    "greentechmedia": "https://www.greentechmedia.com/feed",
    "ars_technica_tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "reuters_tech": "https://feeds.reuters.com/reuters/technologyNews",
}

FERC_RSS = "https://elibrary.ferc.gov/eLibrary/search?activity=recent&rss=true"
BIS_RSS = "https://www.bis.doc.gov/index.php?format=feed&type=rss"

TICKER_ALIASES: dict[str, list[str]] = {
    "MSFT": ["Microsoft", "MSFT"],
    "AMZN": ["Amazon", "AWS", "Amazon Web Services", "AMZN"],
    "GOOGL": ["Google", "Alphabet", "DeepMind", "GOOGL"],
    "META":  ["Meta", "Facebook", "META"],
    "NVDA":  ["NVIDIA", "NVDA"],
    "AMD":   ["AMD", "Advanced Micro Devices"],
    "AVGO":  ["Broadcom", "AVGO"],
    "MRVL":  ["Marvell", "MRVL"],
    "TSM":   ["TSMC", "Taiwan Semiconductor", "TSM"],
    "ASML":  ["ASML"],
    "AMAT":  ["Applied Materials", "AMAT"],
    "LRCX":  ["Lam Research", "LRCX"],
    "KLAC":  ["KLA", "KLAC"],
    "VRT":   ["Vertiv", "VRT"],
    "SMCI":  ["Super Micro", "Supermicro", "SMCI"],
    "DELL":  ["Dell", "DELL"],
    "HPE":   ["Hewlett Packard Enterprise", "HPE"],
    "EQIX":  ["Equinix", "EQIX"],
    "DLR":   ["Digital Realty", "DLR"],
    "AMT":   ["American Tower", "AMT"],
    "CEG":   ["Constellation Energy", "CEG"],
    "VST":   ["Vistra", "VST"],
    "NRG":   ["NRG Energy", "NRG"],
    "TLN":   ["Talen Energy", "TLN"],
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
    # ── 2026-04-27 layer 11-16 + ADR aliases ────────────────────────────────
    # Layer 11 robotics_industrial
    "ROK":   ["Rockwell Automation", "Rockwell", "ROK"],
    "ZBRA":  ["Zebra Technologies", "Zebra Tech", "ZBRA"],
    "CGNX":  ["Cognex", "CGNX"],
    "SYM":   ["Symbotic", "SYM"],
    "EMR":   ["Emerson Electric", "Emerson", "EMR"],
    # Layer 12 robotics_medical_humanoid
    "ISRG":  ["Intuitive Surgical", "da Vinci surgical", "ISRG"],
    "TSLA":  ["Tesla", "Optimus robot", "TSLA"],
    # Layer 13 robotics_mcu_chips
    "TXN":   ["Texas Instruments", "TXN"],
    "MCHP":  ["Microchip Technology", "Microchip", "MCHP"],
    "ADI":   ["Analog Devices", "ADI"],
    # Layer 14 cyber_pureplay
    "CRWD":  ["CrowdStrike", "Falcon platform", "CRWD"],
    "ZS":    ["Zscaler", "ZS"],
    "S":     ["SentinelOne", "Sentinel One", "Singularity platform"],
    # Layer 15 cyber_platform
    "PANW":  ["Palo Alto Networks", "Palo Alto Nets", "Cortex XSIAM", "PANW"],
    "FTNT":  ["Fortinet", "FortiGate", "FTNT"],
    "CHKP":  ["Check Point Software", "Check Point", "CHKP"],
    "CYBR":  ["CyberArk", "CYBR"],
    "TENB":  ["Tenable", "TENB"],
    "QLYS":  ["Qualys", "QLYS"],
    "OKTA":  ["Okta", "OKTA"],
    "AKAM":  ["Akamai", "AKAM"],
    "RPD":   ["Rapid7", "Rapid 7", "RPD"],
    "VRNS":  ["Varonis", "VRNS"],
    # Layer 16 enterprise_saas
    "PLTR":  ["Palantir", "Foundry", "AIP", "PLTR"],
    "NOW":   ["ServiceNow", "Now Platform"],
    "CRM":   ["Salesforce", "Agentforce", "Slack acquisition", "CRM"],
    "ADBE":  ["Adobe", "Firefly", "Photoshop AI", "ADBE"],
    "INTU":  ["Intuit", "TurboTax AI", "QuickBooks AI", "INTU"],
    "DDOG":  ["Datadog", "DDOG"],
    "SNOW":  ["Snowflake", "Cortex AI", "SNOW"],
    "GTLB":  ["GitLab", "GTLB"],
    "TEAM":  ["Atlassian", "Jira AI", "Rovo", "TEAM"],
    "PATH":  ["UiPath", "PATH"],
    "MNDY":  ["Monday.com", "monday.com", "MNDY"],
    # Other US-listed
    "NET":   ["Cloudflare", "NET"],
    "ETN":   ["Eaton", "Eaton Corp", "ETN"],
    "CCJ":   ["Cameco", "CCJ"],
    # ADRs / foreign issuers commonly mentioned in English news
    "STM":   ["STMicroelectronics", "ST Micro", "STM"],
    "ERIC":  ["Ericsson", "Telefonaktiebolaget LM Ericsson", "ERIC"],
    # ── 2026-04-28 medical robotics expansion ───────────────────────────────
    "SYK":   ["Stryker", "Mako robot", "Mako system", "SYK"],
    "MDT":   ["Medtronic", "Hugo robotic", "Hugo RAS", "MDT"],
    "GMED":  ["Globus Medical", "ExcelsiusGPS", "Excelsius surgical", "GMED"],
    "PRCT":  ["PROCEPT BioRobotics", "PROCEPT", "AquaBeam", "Aquablation", "PRCT"],
    # ── 2026-04-28 layer-breadth expansion ──────────────────────────────────
    # Networking
    "HLIT":  ["Harmonic Inc", "HLIT"],
    "CALX":  ["Calix Inc", "Calix broadband", "CALX"],
    "AAOI":  ["Applied Optoelectronics", "AAOI"],
    "EXTR":  ["Extreme Networks", "EXTR"],
    # Servers
    "CDW":   ["CDW Corp", "CDW"],
    "ARW":   ["Arrow Electronics", "ARW"],
    # Datacenter
    "SBAC":  ["SBA Communications", "SBAC"],
    "DBRG":  ["DigitalBridge", "Digital Bridge", "DBRG"],
    "GLW":   ["Corning", "Corning Glass", "GLW"],
    "DOCN":  ["DigitalOcean", "Digital Ocean cloud", "DOCN"],
    # Power
    "DUK":   ["Duke Energy", "DUK"],
    "AEP":   ["American Electric Power", "AEP"],
    "XEL":   ["Xcel Energy", "Xcel"],
    "LEU":   ["Centrus Energy", "Centrus", "uranium enrichment", "LEU"],
    "PLUG":  ["Plug Power", "PLUG"],
    # Grid
    "FLR":   ["Fluor Corp", "Fluor"],
    "ACM":   ["AECOM", "ACM"],
    "KBR":   ["KBR Inc", "KBR"],
    # Cyber pureplay
    "VRSN":  ["Verisign", "VRSN"],
    # Cyber platform
    "LDOS":  ["Leidos", "LDOS"],
    "CACI":  ["CACI International", "CACI"],
    "BAH":   ["Booz Allen Hamilton", "Booz Allen", "BAH"],
    # Robotics MCU/sensor chips
    "ON":    ["onsemi", "ON Semiconductor"],
    "NXPI":  ["NXP Semiconductors", "NXP Semi", "NXPI"],
    "MPWR":  ["Monolithic Power Systems", "Monolithic Power", "MPWR"],
}

# Pre-compiled patterns keyed by ticker — avoids recompiling on every article
_TICKER_PATTERNS: dict[str, re.Pattern] = {
    ticker: re.compile(
        "|".join(r"\b" + re.escape(alias) + r"\b" for alias in aliases),
        re.IGNORECASE,
    )
    for ticker, aliases in TICKER_ALIASES.items()
}


def _tag_tickers(title: str, content: str) -> list[str]:
    """Return sorted list of tickers whose aliases appear in title or content.

    Case-insensitive word-boundary match. One article can match multiple tickers.
    Returns [] if no match.
    """
    text = f"{title or ''} {content or ''}"
    return sorted(
        ticker
        for ticker, pattern in _TICKER_PATTERNS.items()
        if pattern.search(text)
    )


SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("s", tz="UTC")),
    pa.field("source", pa.string()),
    pa.field("url", pa.string()),
    pa.field("title", pa.string()),
    pa.field("content_snippet", pa.string()),
    pa.field("theme_tags", pa.list_(pa.string())),
    pa.field("goldstein_score", pa.float32()),
    pa.field("tone_score", pa.float32()),
    pa.field("num_articles", pa.int32()),
    pa.field("actors", pa.list_(pa.string())),
    pa.field("countries", pa.list_(pa.string())),
    pa.field("mentioned_tickers", pa.list_(pa.string())),
])


def fetch_gdelt_events(query: str, days_back: int = 1) -> list[dict]:
    """
    Fetch GDELT Document API results for a query.
    Free API, no key required.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 250,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
        "format": "json",
    }

    resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "timestamp": datetime.now(timezone.utc),
            "source": "gdelt",
            "url": art.get("url", ""),
            "title": art.get("title", ""),
            "content_snippet": (art.get("title", "") + " " + art.get("excerpt", ""))[:500].strip(),
            "theme_tags": [],
            "goldstein_score": 0.0,
            "tone_score": float(str(art.get("tone", "0") or "0").split(",")[0]) if art.get("tone") else 0.0,
            "num_articles": 1,
            "actors": [],
            "countries": [],
            "mentioned_tickers": _tag_tickers(art.get("title", ""), art.get("excerpt", "")),
        }
        for art in data.get("articles", [])
    ]


def fetch_gdelt_events_per_ticker(
    ticker: str,
    alias: str,
    days_back: int = 1,
    max_retries: int = 3,
) -> list[dict]:
    """Query GDELT for a single ticker's primary alias and force-tag the ticker.

    The thematic GDELT query in `fetch_gdelt_events` returns articles about
    AI-infra topics generally; _tag_tickers can only label them if the ticker
    name happens to appear in the short snippet/title. Most thematic articles
    don't. This per-ticker variant inverts the flow: ask GDELT for articles
    mentioning the ticker by name, then force-tag with the ticker regardless
    of whether _tag_tickers re-detects it in the snippet (the snippet is short
    and often omits the very name that matched the query).

    Retries on 429 (rate-limited) with exponential backoff: 8s, 16s, 32s.

    Result: every returned article carries `ticker` in mentioned_tickers, so
    the downstream sentiment_features aggregator gets per-ticker rows it can
    actually average over.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            articles = fetch_gdelt_events(query=f'"{alias}"', days_back=days_back)
            for art in articles:
                existing = set(art.get("mentioned_tickers") or [])
                existing.add(ticker)
                art["mentioned_tickers"] = sorted(existing)
            return articles
        except Exception as e:  # noqa: BLE001 — covers 429s and JSON-parse failures
            last_exc = e
            msg = str(e)
            # Treat 429s and empty-body JSON-parse errors as rate-limit signals
            if "429" in msg or "Expecting value" in msg or "timed out" in msg.lower():
                wait = 8 * (2 ** attempt)
                time.sleep(wait)
                continue
            raise
    raise last_exc if last_exc else RuntimeError(f"fetch_gdelt_events_per_ticker {ticker} failed")


def fetch_gdelt_gkg_tone(theme: str, days_back: int = 7) -> dict:
    """
    Fetch GDELT GKG tone timeline for a theme.
    Returns daily average tone scores — core financial signal.
    Free, no key required.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    params = {
        "query": f"theme:{theme}",
        "mode": "timelinetone",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "TIMELINESMOOTH": 3,
        "format": "json",
    }

    resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def scrape_rss_feeds(output_dir: Path):
    """
    Parse all configured RSS feeds via feedparser (no API key).
    Writes article metadata to Parquet partitioned by date.
    """
    records = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:50]:
                records.append({
                    "timestamp": datetime.now(timezone.utc),
                    "source": source,
                    "url": entry.get("link", ""),
                    "title": entry.get("title", ""),
                    "content_snippet": (entry.get("summary") or "")[:500],
                    "theme_tags": [],
                    "goldstein_score": 0.0,
                    "tone_score": 0.0,
                    "num_articles": 1,
                    "actors": [],
                    "countries": [],
                    "mentioned_tickers": _tag_tickers(
                        entry.get("title", ""),
                        entry.get("summary") or "",
                    ),
                })
            print(f"[News] {source}: {len(feed.entries)} articles")
        except Exception as e:
            print(f"[News] ERROR {source}: {e}")
        time.sleep(1)  # rate limit compliance

    if records:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = output_dir / "news" / "rss" / f"date={date_str}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        pq.write_table(table, path, compression="snappy")


_SEC_HEADERS = {"User-Agent": "ai-infra-predictor research@example.com"}


def search_edgar_fulltext(query: str, form_type: str = "8-K", days_back: int = 7) -> list[dict]:
    """
    Search SEC EDGAR full-text for financial signals.
    Free API, no key required — but SEC fair-use requires a descriptive User-Agent
    or the request returns 403 Forbidden.
    """
    start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": query,
        "dateRange": "custom",
        "startdt": start_date,
        "forms": form_type,
        "_source": "file_date,entity_name,file_num,period_of_report,form_type",
    }

    resp = requests.get(url, params=params, headers=_SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hits", {}).get("hits", [])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    import argparse
    parser = argparse.ArgumentParser(description="Ingest news (RSS + GDELT + EDGAR full-text).")
    parser.add_argument(
        "--days", type=int, default=7,
        help="Lookback window in days for GDELT events (default 7). "
             "Use a larger value (e.g. 90) for cold-start backfill of sentiment history.",
    )
    parser.add_argument(
        "--per-ticker", action="store_true", default=False,
        help="OPT-IN: after the thematic GDELT fetch, run an additional "
             "per-ticker query for each ticker's primary alias. Articles "
             "are force-tagged with the matching ticker so the downstream "
             "sentiment_features aggregator gets per-ticker rows to average "
             "over. NOTE: GDELT DOC API rate-limits aggressive callers with "
             "429s; even at 30s between queries the free API throttles "
             "~70%% of requests. Realistic full-150-ticker pass takes ~1-2 "
             "hours with most queries retrying. Use sparingly (e.g. weekly) "
             "or in cold-start backfill, not every daily run.",
    )
    parser.add_argument(
        "--per-ticker-sleep", type=float, default=30.0,
        help="Sleep seconds between per-ticker GDELT queries (default 30). "
             "GDELT throttling on the free DOC API is severe; the fetcher "
             "retries with exponential backoff (8/16/32s) but the simpler "
             "fix is just running fewer queries. Lower the value if you've "
             "observed your IP gets less aggressive throttling.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw")

    print("[News] Scraping RSS feeds...")
    scrape_rss_feeds(output_dir)

    print(f"[News] Fetching GDELT events (last {args.days} days)...")
    try:
        # GDELT treats space-separated terms as a phrase; we want OR semantics
        # across the AI-infra + robotics + cybersecurity + enterprise-SaaS themes
        # so that any matching article is returned.
        # GDELT requires every quoted phrase to be ≥5 chars (otherwise the API
        # returns 200 with body "The specified phrase is too short." and 0
        # articles). Keep all terms above that threshold.
        articles = fetch_gdelt_events(
            '("data center" OR "semiconductor" OR "nuclear power" OR '
            '"export control" OR "AI chip" OR "hyperscaler" OR '
            '"humanoid robot" OR "industrial automation" OR "AI agent" OR '
            '"agentic AI" OR "ransomware" OR "zero trust")',
            days_back=args.days,
        )
        print(f"[News] GDELT: {len(articles)} articles fetched")

        if articles:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = output_dir / "news" / "gdelt" / f"date={date_str}" / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(articles, schema=SCHEMA)
            pq.write_table(table, path, compression="snappy")
            print(f"[News] Wrote {len(articles)} GDELT articles → {path}")
    except Exception as e:
        print(f"[News] GDELT ERROR: {e}")
    time.sleep(1)  # rate limit compliance between GDELT and EDGAR calls

    # ── Per-ticker GDELT fetch (lightweight sentiment coverage fix) ─────────
    # The thematic query above almost never returns articles where the short
    # snippet mentions one of our 194 tickers by name, so the downstream
    # sentiment_features aggregator gets ~zero per-ticker rows. Looping over
    # each ticker's primary alias instead gives every fetched article an
    # unambiguous ticker tag.
    if args.per_ticker:
        print(f"[News] Per-ticker GDELT fetch ({len(TICKER_ALIASES)} tickers, "
              f"~{len(TICKER_ALIASES)*args.per_ticker_sleep:.0f}s)…")
        per_ticker_rows: list[dict] = []
        n_ok = 0
        n_err = 0
        for ticker, aliases in TICKER_ALIASES.items():
            if not aliases:
                continue
            primary_alias = aliases[0]
            try:
                rows = fetch_gdelt_events_per_ticker(
                    ticker, primary_alias, days_back=args.days,
                )
                per_ticker_rows.extend(rows)
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — fail-soft per project convention
                n_err += 1
                if n_err < 5:
                    print(f"[News] per-ticker GDELT {ticker}: {e}")
            time.sleep(args.per_ticker_sleep)

        print(f"[News] Per-ticker GDELT: {n_ok}/{len(TICKER_ALIASES)} OK, "
              f"{n_err} errors, {len(per_ticker_rows)} articles fetched")

        if per_ticker_rows:
            # Append to the same date partition as the thematic fetch so the
            # downstream nlp_pipeline (which globs data.parquet only) picks
            # both up in one pass. De-dupe by URL to avoid re-scoring the
            # same article when thematic + per-ticker queries overlap.
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = output_dir / "news" / "gdelt" / f"date={date_str}" / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            existing_rows: list[dict] = []
            if path.exists():
                try:
                    existing_table = pq.read_table(path)
                    existing_rows = existing_table.to_pylist()
                except Exception as e:  # noqa: BLE001
                    print(f"[News] couldn't read existing {path} ({e}); overwriting")

            # Merge thematic + per-ticker, dedupe on url, keep per-ticker tags
            by_url: dict[str, dict] = {}
            for row in existing_rows + per_ticker_rows:
                url = row.get("url") or ""
                if url and url in by_url:
                    # Merge mentioned_tickers — same article seen via both paths
                    by_url[url]["mentioned_tickers"] = sorted(
                        set(by_url[url].get("mentioned_tickers") or [])
                        | set(row.get("mentioned_tickers") or [])
                    )
                else:
                    by_url[url] = row

            merged = list(by_url.values())
            table = pa.Table.from_pylist(merged, schema=SCHEMA)
            pq.write_table(table, path, compression="snappy")
            print(f"[News] Wrote {len(merged)} GDELT articles "
                  f"({len(per_ticker_rows)} per-ticker new) → {path}")

    print("[News] Searching EDGAR for AI infrastructure filings...")
    try:
        edgar_hits = search_edgar_fulltext('"power purchase agreement" AND "data center"', days_back=7)
        print(f"[News] EDGAR: {len(edgar_hits)} filings found")
    except Exception as e:
        print(f"[News] EDGAR ERROR: {e}")
