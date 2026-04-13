import pytest
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory matching the real Hive-partitioned layout."""
    for subdir in ["ais", "flights", "news/rss", "news/scored", "satellite", "financials"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
    return tmp_path
