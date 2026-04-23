"""
Ingest federal contract award data from SAM.gov for AI/datacenter NAICS codes.

Raw storage: data/raw/gov_contracts/date=YYYY-MM-DD/awards.parquet
Schema: (date, awardee_name, uei, contract_value_usd, naics_code, agency)

NAICS codes: 541511, 541512, 541519 (IT services), 518210 (hosting), 334413 (semiconductors)

Requires: SAM_GOV_API_KEY environment variable (free key at sam.gov)
Rate limit: 10 requests/minute → time.sleep(6.0) between pages.

Usage:
    python ingestion/sam_gov_ingestion.py               # fetch today (rolling 90-day window)
    python ingestion/sam_gov_ingestion.py --date 2024-01-15
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import requests
import polars as pl

_LOG = logging.getLogger(__name__)

_SAM_GOV_BASE_URL = "https://api.sam.gov/opportunities/v2/search"
_NAICS_CODES = "541511,541512,541519,518210,334413"

_CONTRACT_SCHEMA = {
    "date": pl.Date,
    "awardee_name": pl.Utf8,
    "uei": pl.Utf8,
    "contract_value_usd": pl.Float64,
    "naics_code": pl.Utf8,
    "agency": pl.Utf8,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_CONTRACT_SCHEMA)


@runtime_checkable
class GovContractSource(Protocol):
    def fetch(self, date_str: str) -> pl.DataFrame:
        """Return AI/DC NAICS contract awards for rolling 90-day window ending date_str.

        Returns DataFrame matching _CONTRACT_SCHEMA.
        Returns empty DataFrame if no awards found.
        """
        ...


class SamGovSource:
    """Fetch contract awards from SAM.gov API (requires SAM_GOV_API_KEY)."""

    def fetch(self, date_str: str) -> pl.DataFrame:
        api_key = os.environ.get("SAM_GOV_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SAM_GOV_API_KEY not set. Get a free key at https://sam.gov/content/duns-sam"
            )

        as_of = datetime.date.fromisoformat(date_str)
        start = as_of - datetime.timedelta(days=90)

        rows: list[dict] = []
        offset = 0
        limit = 100

        while True:
            params = {
                "limit": limit,
                "offset": offset,
                "awardDateRange": f"{start},{as_of}",
                "naicsCode": _NAICS_CODES,
            }
            resp = requests.get(_SAM_GOV_BASE_URL, headers={"X-Api-Key": api_key}, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            awards = data.get("opportunitiesData") or data.get("results") or []
            for award in awards:
                awardee = award.get("awardee") or {}
                award_info = award.get("award") or {}
                org = (award.get("organizationHierarchy") or [{}])[0]
                rows.append({
                    "date": as_of,
                    "awardee_name": awardee.get("name") or "",
                    "uei": awardee.get("ueiSAM") or "",
                    "contract_value_usd": float(award_info.get("amount") or 0),
                    "naics_code": award.get("naicsCode") or "",
                    "agency": award.get("department") or org.get("name") or "",
                })

            total = int(data.get("totalRecords") or 0)
            offset += len(awards)

            if not awards or offset >= total:
                break

            time.sleep(6.0)  # 10 req/min rate limit

        if not rows:
            return _empty()

        return pl.DataFrame(rows, schema=_CONTRACT_SCHEMA)


def ingest_sam_gov(
    date_str: str,
    output_dir: Path,
    source: GovContractSource | None = None,
) -> None:
    """Fetch contract awards and write Hive-partitioned parquet.

    Dates with no awards produce no file (silently skipped).
    """
    if source is None:
        source = SamGovSource()

    df = source.fetch(date_str)
    if not df.is_empty():
        out_dir = output_dir / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / "awards.parquet", compression="snappy")
        _LOG.info("Wrote %d contract awards for %s", len(df), date_str)
    else:
        _LOG.debug("No contract awards for %s — skipping", date_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest SAM.gov contract awards")
    parser.add_argument(
        "--date",
        default=str(datetime.date.today()),
        help="As-of date (YYYY-MM-DD). Fetches rolling 90-day window. Defaults to today.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw/gov_contracts")
    _LOG.info("Fetching SAM.gov awards as of %s", args.date)
    ingest_sam_gov(args.date, output_dir)
