import requests
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path
import time

CARGO_AIRPORTS = ["RCTP", "PANC", "KLAX", "KSFO", "KORD", "KDFW", "RJAA", "RKSI"]
EXEC_AIRPORTS = ["KBFI", "KPAO", "KSJC", "KSFO"]
DC_AIRPORTS = ["KIWA", "KPSC", "KGYY", "KHIO"]

CARGO_AIRCRAFT_TYPES = ["B748", "B77F", "A332F", "A333F", "MD11F", "B763F", "B744F"]
EXEC_AIRCRAFT_TYPES = ["GL7T", "GL6T", "GL5T", "C68A", "C56X", "FA7X", "C750"]

SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("s", tz="UTC")),
    pa.field("icao24", pa.string()),
    pa.field("callsign", pa.string()),
    pa.field("origin_country", pa.string()),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("altitude_m", pa.float32()),
    pa.field("velocity_ms", pa.float32()),
    pa.field("on_ground", pa.bool_()),
    pa.field("aircraft_category", pa.string()),
    pa.field("signal_type", pa.string()),
])

OPENSKY_BASE = "https://opensky-network.org/api"


def fetch_arrivals_at_airport(
    airport_icao: str,
    start_ts: int,
    end_ts: int,
    username: str = "",
    password: str = "",
) -> list[dict]:
    """
    Fetch arrivals at an airport within a time window.
    Free, no key needed. Rate limit: 400 calls/day anonymous.
    Returns empty list on 404 (no flights in window).
    """
    url = f"{OPENSKY_BASE}/flights/arrival"
    params = {"airport": airport_icao, "begin": start_ts, "end": end_ts}
    auth = (username, password) if username else None

    try:
        resp = requests.get(url, params=params, auth=auth, timeout=30)
        resp.raise_for_status()
        flights = resp.json() or []
        for f in flights:
            if f.get("callsign"):
                f["callsign"] = f["callsign"].strip()
        return flights
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return []
        raise


def fetch_state_vectors_bbox(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
) -> list:
    """Fetch all aircraft in a bounding box. Free, no auth required."""
    url = f"{OPENSKY_BASE}/states/all"
    params = {"lamin": lat_min, "lamax": lat_max, "lomin": lon_min, "lomax": lon_max}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("states", [])


def run_daily_cargo_scan(output_dir: Path, username: str = "", password: str = ""):
    """
    Fetch yesterday's arrivals at key cargo airports.
    Writes Parquet partitioned by date.
    Rate: 1 request/second (free tier compliance).
    """
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - 86400

    records = []
    for airport in CARGO_AIRPORTS:
        try:
            flights = fetch_arrivals_at_airport(airport, start_ts, end_ts, username, password)
            for f in flights:
                records.append({
                    "timestamp": datetime.fromtimestamp(
                        f.get("lastSeen", end_ts), tz=timezone.utc
                    ),
                    "icao24": f.get("icao24", ""),
                    "callsign": (f.get("callsign") or "").strip(),
                    "origin_country": f.get("estDepartureAirport", ""),
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "altitude_m": 0.0,
                    "velocity_ms": 0.0,
                    "on_ground": True,
                    "aircraft_category": "cargo",
                    "signal_type": f"cargo_arrival_{airport}",
                })
            print(f"[Flights] {airport}: {len(flights)} arrivals")
        except Exception as e:
            print(f"[Flights] ERROR {airport}: {e}")
        time.sleep(1)  # Rate limit compliance

    if records:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = output_dir / "flights" / f"date={date_str}" / "cargo.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        pq.write_table(table, path, compression="snappy")
        print(f"[Flights] Wrote {len(records)} cargo flight records")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    run_daily_cargo_scan(
        Path("data/raw"),
        username=os.getenv("OPENSKY_USERNAME", ""),
        password=os.getenv("OPENSKY_PASSWORD", ""),
    )
