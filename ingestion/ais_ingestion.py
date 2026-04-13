import asyncio
import json
import websockets
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path

CORRIDORS = {
    "taiwan_strait": [[21.5, 119.0], [25.5, 122.5]],
    "anchorage": [[60.0, -150.5], [61.5, -147.0]],
    "rotterdam": [[51.8, 4.0], [52.0, 4.5]],
    "long_beach": [[33.5, -118.5], [34.0, -118.0]],
}

CARGO_TYPES = list(range(70, 90))

SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("s", tz="UTC")),
    pa.field("mmsi", pa.string()),
    pa.field("imo", pa.string()),
    pa.field("vessel_name", pa.string()),
    pa.field("vessel_type", pa.int32()),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("speed_knots", pa.float32()),
    pa.field("course", pa.float32()),
    pa.field("destination", pa.string()),
    pa.field("draught", pa.float32()),
    pa.field("corridor", pa.string()),
])


async def stream_ais(api_key: str, output_dir: Path):
    """
    Stream AIS data from aisstream.io WebSocket.
    Requires free API key from aisstream.io (https://aisstream.io).
    Buffers 1000 messages before writing to Parquet.
    """
    uri = "wss://stream.aisstream.io/v0/stream"
    buffer = []

    subscription = {
        "Apikey": api_key,
        "BoundingBoxes": list(CORRIDORS.values()),
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(subscription))
        print(f"[AIS] Connected to aisstream.io — monitoring {len(CORRIDORS)} corridors")

        async for raw in ws:
            msg = json.loads(raw)
            if _is_cargo(msg):
                row = _parse_message(msg)
                if row:
                    buffer.append(row)

            if len(buffer) >= 1000:
                _write_parquet(buffer, output_dir)
                buffer = []


def _is_cargo(msg: dict) -> bool:
    """True if AIS type code is 70–89 (cargo/tanker) or message has a PositionReport."""
    ship_type = msg.get("Message", {}).get("ShipStaticData", {}).get("Type", 0)
    pos_report = msg.get("Message", {}).get("PositionReport", {})
    return ship_type in CARGO_TYPES or bool(pos_report)


def _parse_message(msg: dict) -> dict | None:
    """Extract fields from AIS message. Returns None if no lat/lon available."""
    try:
        metadata = msg.get("MetaData", {})
        position = msg.get("Message", {}).get("PositionReport", {})
        static = msg.get("Message", {}).get("ShipStaticData", {})

        lat = metadata.get("latitude") or position.get("Latitude")
        lon = metadata.get("longitude") or position.get("Longitude")

        if not lat or not lon:
            return None

        return {
            "timestamp": datetime.now(timezone.utc),
            "mmsi": str(metadata.get("MMSI", "")),
            "imo": str(static.get("ImoNumber", "")),
            "vessel_name": metadata.get("ShipName", ""),
            "vessel_type": int(static.get("Type", 0)),
            "latitude": float(lat),
            "longitude": float(lon),
            "speed_knots": float(position.get("Sog", 0)),
            "course": float(position.get("Cog", 0)),
            "destination": static.get("Destination", ""),
            "draught": float(static.get("Draught", 0)),
            "corridor": _identify_corridor(float(lat), float(lon)) or "other",
        }
    except Exception:
        return None


def _identify_corridor(lat: float, lon: float) -> str | None:
    """Return corridor name if vessel is inside a defined bbox, else None."""
    for name, bbox in CORRIDORS.items():
        min_lat, min_lon = bbox[0]
        max_lat, max_lon = bbox[1]
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return name
    return None


def _write_parquet(buffer: list, output_dir: Path):
    """Append buffer to daily Parquet file (creates or appends existing)."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = output_dir / "ais" / f"date={date_str}" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(buffer, schema=SCHEMA)

    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, path, compression="snappy")
    print(f"[AIS] Wrote {len(buffer)} rows → {path}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("AISSTREAM_API_KEY", "")
    if not api_key:
        print("[AIS] No AISSTREAM_API_KEY set.")
        print("[AIS] Register free at https://aisstream.io and add to .env:")
        print("[AIS]   AISSTREAM_API_KEY=your_key_here")
    else:
        asyncio.run(stream_ais(api_key=api_key, output_dir=Path("data/raw")))
