import feedparser
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
        }
        for art in data.get("articles", [])
    ]


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


def search_edgar_fulltext(query: str, form_type: str = "8-K", days_back: int = 7) -> list[dict]:
    """
    Search SEC EDGAR full-text for financial signals.
    Free API, no key required.
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

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hits", {}).get("hits", [])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    output_dir = Path("data/raw")

    print("[News] Scraping RSS feeds...")
    scrape_rss_feeds(output_dir)

    print("[News] Fetching GDELT events...")
    try:
        articles = fetch_gdelt_events(
            "semiconductor export control chip data center nuclear power",
            days_back=1,
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

    print("[News] Searching EDGAR for AI infrastructure filings...")
    try:
        edgar_hits = search_edgar_fulltext('"power purchase agreement" AND "data center"', days_back=7)
        print(f"[News] EDGAR: {len(edgar_hits)} filings found")
    except Exception as e:
        print(f"[News] EDGAR ERROR: {e}")
