# Robotics Layer Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Layer 11 (`robotics`) into three sub-layers, add 8 new tickers (141 → 149), support HKD/KRW currencies (FX pairs 7 → 9), and add a metadata-only `PENDING_IPO_WATCHLIST`.

**Architecture:** Pure registry/data change. The model, training, inference, supply-chain features, and FX-features modules read layers/currencies generically and require no changes. Two new FX pairs are additive. Cyber layer ids shift 12/13 → 14/15.

**Tech Stack:** Python 3.11+, Polars, PyArrow, yfinance, pytest

**Spec:** [docs/superpowers/specs/2026-04-25-robotics-layer-expansion.md](../specs/2026-04-25-robotics-layer-expansion.md)

---

### Task 1: FX expansion — add HKDUSD/KRWUSD pairs

**Files:**
- Modify: `ingestion/fx_ingestion.py`
- Modify: `tests/test_fx_ingestion.py`

- [ ] **Step 1: Add failing tests for HKD/KRW**

Append to `tests/test_fx_ingestion.py`:

```python
def test_supported_currencies_includes_hkd_krw():
    """SUPPORTED_CURRENCIES must cover the 9 currencies used by the registry."""
    from ingestion.fx_ingestion import SUPPORTED_CURRENCIES
    assert SUPPORTED_CURRENCIES == frozenset({
        "EUR", "CHF", "JPY", "DKK", "SEK", "NOK", "GBP", "HKD", "KRW",
    })


def test_currency_to_pair_includes_hkd_krw():
    """CURRENCY_TO_PAIR must map HKD→HKDUSD and KRW→KRWUSD."""
    from ingestion.fx_ingestion import CURRENCY_TO_PAIR
    assert CURRENCY_TO_PAIR["HKD"] == "HKDUSD"
    assert CURRENCY_TO_PAIR["KRW"] == "KRWUSD"


def test_fetch_fx_rates_default_pair_count():
    """Default fetch covers 9 pairs."""
    from ingestion.fx_ingestion import _FX_SYMBOLS
    assert len(_FX_SYMBOLS) == 9
    assert _FX_SYMBOLS["HKDUSD"] == "HKDUSD=X"
    assert _FX_SYMBOLS["KRWUSD"] == "KRWUSD=X"


def test_fetch_fx_rates_hkd_krw_happy_path():
    """fetch_fx_rates returns valid DataFrames for HKDUSD and KRWUSD."""
    from unittest.mock import patch
    from ingestion.fx_ingestion import fetch_fx_rates

    mock_data = _mock_yf_response([7.78, 7.79], ["2025-01-01", "2025-01-02"])
    with patch("ingestion.fx_ingestion.yf.download", return_value=mock_data):
        result = fetch_fx_rates(["HKDUSD", "KRWUSD"], years=1)

    assert "HKDUSD" in result and "KRWUSD" in result
    for pair in ("HKDUSD", "KRWUSD"):
        df = result[pair]
        assert df["rate"].dtype == pl.Float64
        assert df["date"].dtype == pl.Date
        assert len(df) == 2
```

- [ ] **Step 2: Run new tests — confirm they fail**

Run: `pytest tests/test_fx_ingestion.py -v -k "hkd_krw or default_pair_count"`
Expected: 4 FAIL with `KeyError: 'HKD'` or `assert frozenset({...}) == frozenset({...})`.

- [ ] **Step 3: Add HKDUSD and KRWUSD to `_FX_SYMBOLS`**

In `ingestion/fx_ingestion.py`, replace the `_FX_SYMBOLS` block:

```python
_FX_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "CHFUSD": "CHFUSD=X",
    "JPYUSD": "JPYUSD=X",
    "DKKUSD": "DKKUSD=X",
    "SEKUSD": "SEKUSD=X",
    "NOKUSD": "NOKUSD=X",
    "GBPUSD": "GBPUSD=X",
    "HKDUSD": "HKDUSD=X",
    "KRWUSD": "KRWUSD=X",
}
```

- [ ] **Step 4: Update `SUPPORTED_CURRENCIES` and `CURRENCY_TO_PAIR`**

Replace:

```python
SUPPORTED_CURRENCIES: frozenset[str] = frozenset(
    {"EUR", "CHF", "JPY", "DKK", "SEK", "NOK", "GBP", "HKD", "KRW"}
)

CURRENCY_TO_PAIR: dict[str, str] = {
    "EUR": "EURUSD",
    "CHF": "CHFUSD",
    "JPY": "JPYUSD",
    "DKK": "DKKUSD",
    "SEK": "SEKUSD",
    "NOK": "NOKUSD",
    "GBP": "GBPUSD",
    "HKD": "HKDUSD",
    "KRW": "KRWUSD",
}
```

- [ ] **Step 5: Run all FX tests — confirm pass**

Run: `pytest tests/test_fx_ingestion.py -v`
Expected: all pass (5 prior + 4 new = 9 tests).

- [ ] **Step 6: Commit**

```bash
git add ingestion/fx_ingestion.py tests/test_fx_ingestion.py
git commit -m "feat: add HKDUSD/KRWUSD FX pairs (7→9 currencies)"
```

---

### Task 2: Update existing registry tests for new robotics structure (red phase)

**Files:**
- Modify: `tests/test_ticker_registry.py`

This task only updates tests — they will FAIL until Task 3 lands. We commit Task 2 + Task 3 as a single atomic change.

- [ ] **Step 1: Update `test_ticker_count`**

Replace lines 4–7 of `tests/test_ticker_registry.py`:

```python
def test_ticker_count():
    from ingestion.ticker_registry import TICKERS, TICKER_LAYERS
    # 141 + 8 new robotics-pillar tickers (EMR, TSLA, 1683.HK, 005380.KS,
    # TXN, MCHP, 6723.T, ADI) = 149.
    assert len(TICKERS) == 149
    assert len(TICKER_LAYERS) == 149
```

- [ ] **Step 2: Update `test_layers_returns_13` → `test_layers_returns_15`**

Rename and update:

```python
def test_layers_returns_15():
    from ingestion.ticker_registry import layers
    assert len(layers()) == 15
```

- [ ] **Step 3: Update `test_layers_order` for new tail**

Replace `test_layers_order`:

```python
def test_layers_order():
    from ingestion.ticker_registry import layers
    result = layers()
    assert result[0] == "cloud"
    # robotics pillar (ids 11/12/13) comes before cyber pillar (14/15)
    assert result[-5] == "robotics_industrial"
    assert result[-4] == "robotics_medical_humanoid"
    assert result[-3] == "robotics_mcu_chips"
    assert result[-2] == "cyber_pureplay"
    assert result[-1] == "cyber_platform"
```

- [ ] **Step 4: Update cyber layer-id assertions**

Edit `test_cyber_pureplay_layer_populated`:

```python
def test_cyber_pureplay_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "cyber_pureplay" in LAYER_IDS
    assert LAYER_IDS["cyber_pureplay"] == 14   # was 12, shifted by robotics split
    tickers = tickers_in_layer("cyber_pureplay")
    assert len(tickers) == 5
    assert "CRWD" in tickers
    assert "ZS" in tickers
    assert "S" in tickers
    assert "DARK.L" in tickers
    assert "VRNS" in tickers
```

Edit `test_cyber_platform_layer_populated`:

```python
def test_cyber_platform_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "cyber_platform" in LAYER_IDS
    assert LAYER_IDS["cyber_platform"] == 15   # was 13, shifted by robotics split
    tickers = tickers_in_layer("cyber_platform")
    assert len(tickers) == 9
    for expected in ["PANW", "FTNT", "CHKP", "CYBR", "TENB", "QLYS", "OKTA", "AKAM", "RPD"]:
        assert expected in tickers, f"{expected} missing from cyber_platform"
```

- [ ] **Step 5: Replace `test_robotics_layer_populated` with three sub-layer tests**

Delete the existing `test_robotics_layer_populated` (lines 103–111) and add three replacements:

```python
def test_robotics_industrial_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics_industrial" in LAYER_IDS
    assert LAYER_IDS["robotics_industrial"] == 11
    industrial = tickers_in_layer("robotics_industrial")
    assert len(industrial) == 11
    expected = {"ROK", "ZBRA", "CGNX", "SYM", "ABBN.SW", "KGX.DE",
                "HEXA-B.ST", "6954.T", "6506.T", "6861.T", "EMR"}
    assert set(industrial) == expected


def test_robotics_medical_humanoid_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics_medical_humanoid" in LAYER_IDS
    assert LAYER_IDS["robotics_medical_humanoid"] == 12
    mh = tickers_in_layer("robotics_medical_humanoid")
    assert len(mh) == 4
    assert set(mh) == {"ISRG", "TSLA", "1683.HK", "005380.KS"}


def test_robotics_mcu_chips_layer_populated():
    from ingestion.ticker_registry import tickers_in_layer, LAYER_IDS
    assert "robotics_mcu_chips" in LAYER_IDS
    assert LAYER_IDS["robotics_mcu_chips"] == 13
    mcu = tickers_in_layer("robotics_mcu_chips")
    assert len(mcu) == 4
    assert set(mcu) == {"TXN", "MCHP", "6723.T", "ADI"}


def test_legacy_robotics_key_removed():
    """The flat 'robotics' layer key must not exist after the split."""
    from ingestion.ticker_registry import LAYER_IDS, LAYER_LABELS
    assert "robotics" not in LAYER_IDS
    assert "robotics" not in LAYER_LABELS
```

- [ ] **Step 6: Update `test_non_usd_tickers`**

Edit:

```python
def test_non_usd_tickers():
    from ingestion.ticker_registry import non_usd_tickers, TICKER_CURRENCY
    # Was 37; +1683.HK (HKD) +005380.KS (KRW) +6723.T (JPY) = 40
    result = non_usd_tickers()
    assert len(result) == 40
    for t in result:
        assert TICKER_CURRENCY[t] != "USD", f"{t} is USD but in non_usd_tickers()"
    assert "NVDA" not in result
    assert "ABBN.SW" in result
    assert "DARK.L" in result
    assert "1683.HK" in result
    assert "005380.KS" in result
    assert "6723.T" in result
```

- [ ] **Step 7: Add `test_pending_ipo_watchlist_structure`**

Append to `tests/test_ticker_registry.py`:

```python
def test_pending_ipo_watchlist_structure():
    """PENDING_IPO_WATCHLIST entries are well-formed and reference real layers."""
    from ingestion.ticker_registry import PENDING_IPO_WATCHLIST, LAYER_IDS
    assert len(PENDING_IPO_WATCHLIST) >= 2
    required_keys = {"name", "expected_symbol", "layer", "expected_date"}
    for entry in PENDING_IPO_WATCHLIST:
        assert required_keys <= entry.keys(), f"Missing keys in {entry}"
        assert entry["layer"] in LAYER_IDS, (
            f"Layer {entry['layer']!r} not in LAYER_IDS"
        )
    names = {e["name"] for e in PENDING_IPO_WATCHLIST}
    assert "Unitree Robotics" in names
    assert "Boston Dynamics" in names
```

- [ ] **Step 8: Run tests — confirm reds**

Run: `pytest tests/test_ticker_registry.py -v`
Expected: many failures (counts wrong, robotics_industrial / _medical_humanoid / _mcu_chips not defined, `PENDING_IPO_WATCHLIST` import error). **Do not commit yet** — Task 3 implements.

---

### Task 3: Implement registry changes (green phase) + new sub-layer test file

**Files:**
- Modify: `ingestion/ticker_registry.py`
- Create: `tests/test_robotics_sub_layers.py`

- [ ] **Step 1: Replace robotics layer block in `TICKERS_INFO`**

In `ingestion/ticker_registry.py`, replace the existing `# ── Layer 11: Robotics / Automation / Industrial AI (11) ──` block (lines 149–160) with three sub-layer blocks:

```python
    # ── Layer 11: Robotics — Industrial Automation (11) ──────────────────────────
    TickerInfo("ROK",       "robotics_industrial",       "US", "USD", "US"),
    TickerInfo("ZBRA",      "robotics_industrial",       "US", "USD", "US"),
    TickerInfo("CGNX",      "robotics_industrial",       "US", "USD", "US"),
    TickerInfo("SYM",       "robotics_industrial",       "US", "USD", "US"),
    TickerInfo("EMR",       "robotics_industrial",       "US", "USD", "US"),
    TickerInfo("ABBN.SW",   "robotics_industrial",       "SW", "CHF", "CH"),
    TickerInfo("KGX.DE",    "robotics_industrial",       "DE", "EUR", "DE"),
    TickerInfo("HEXA-B.ST", "robotics_industrial",       "ST", "SEK", "SE"),
    TickerInfo("6954.T",    "robotics_industrial",       "T",  "JPY", "JP"),
    TickerInfo("6506.T",    "robotics_industrial",       "T",  "JPY", "JP"),
    TickerInfo("6861.T",    "robotics_industrial",       "T",  "JPY", "JP"),
    # ── Layer 12: Robotics — Medical & Humanoid (4) ──────────────────────────────
    TickerInfo("ISRG",      "robotics_medical_humanoid", "US", "USD", "US"),
    TickerInfo("TSLA",      "robotics_medical_humanoid", "US", "USD", "US"),
    TickerInfo("1683.HK",   "robotics_medical_humanoid", "HK", "HKD", "HK"),
    TickerInfo("005380.KS", "robotics_medical_humanoid", "KS", "KRW", "KR"),
    # ── Layer 13: Robotics — MCU & Sensor Chips (4) ──────────────────────────────
    TickerInfo("TXN",       "robotics_mcu_chips",        "US", "USD", "US"),
    TickerInfo("MCHP",      "robotics_mcu_chips",        "US", "USD", "US"),
    TickerInfo("ADI",       "robotics_mcu_chips",        "US", "USD", "US"),
    TickerInfo("6723.T",    "robotics_mcu_chips",        "T",  "JPY", "JP"),
```

- [ ] **Step 2: Update `LAYER_IDS` — split robotics, shift cyber**

Replace the `LAYER_IDS` block:

```python
LAYER_IDS: dict[str, int] = {
    "cloud": 1, "compute": 2, "semi_equipment": 3, "networking": 4,
    "servers": 5, "datacenter": 6, "power": 7, "cooling": 8,
    "grid": 9, "metals": 10,
    "robotics_industrial": 11, "robotics_medical_humanoid": 12, "robotics_mcu_chips": 13,
    "cyber_pureplay": 14, "cyber_platform": 15,
}
```

- [ ] **Step 3: Update `LAYER_LABELS`**

Replace the `LAYER_LABELS` block:

```python
LAYER_LABELS: dict[str, str] = {
    "cloud":                       "Hyperscalers / Cloud",
    "compute":                     "AI Compute / Chips",
    "semi_equipment":              "Semiconductor Equipment & Materials",
    "networking":                  "Networking / Interconnect",
    "servers":                     "Servers / Storage / Systems",
    "datacenter":                  "Data Center Operators / REITs",
    "power":                       "Power / Energy / Nuclear",
    "cooling":                     "Cooling / Facilities / Backup Power",
    "grid":                        "Grid / Construction / Electrical",
    "metals":                      "Metals / Materials",
    "robotics_industrial":         "Robotics — Industrial Automation",
    "robotics_medical_humanoid":   "Robotics — Medical & Humanoid",
    "robotics_mcu_chips":          "Robotics — MCU & Sensor Chips",
    "cyber_pureplay":              "AI Cybersecurity — Pure Plays",
    "cyber_platform":              "AI Cybersecurity — Platform Vendors",
}
```

- [ ] **Step 4: Update module docstring**

Replace the first line of `ticker_registry.py`:

```python
"""Central registry of all 149 AI infrastructure + robotics + cybersecurity tickers."""
```

- [ ] **Step 5: Add `PENDING_IPO_WATCHLIST` constant**

Append to `ingestion/ticker_registry.py` (after `HYPERSCALERS`):

```python
# Pending-IPO watchlist — humanoid plays expected to list mid-2026 onward.
# Metadata-only. Not in TICKERS_INFO. Not fetched. Add to TICKERS_INFO when each lists.
PENDING_IPO_WATCHLIST: list[dict[str, str]] = [
    {
        "name": "Unitree Robotics",
        "expected_symbol": "TBD.SS",
        "layer": "robotics_medical_humanoid",
        "expected_date": "2026-Q3",
    },
    {
        "name": "Boston Dynamics",
        "expected_symbol": "TBD",
        "layer": "robotics_medical_humanoid",
        "expected_date": "TBD",
    },
]
```

- [ ] **Step 6: Create new test file `tests/test_robotics_sub_layers.py`**

```python
"""Membership and integrity tests for the three robotics sub-layers."""
from __future__ import annotations


def test_industrial_membership_exact():
    from ingestion.ticker_registry import tickers_in_layer
    assert set(tickers_in_layer("robotics_industrial")) == {
        "ROK", "ZBRA", "CGNX", "SYM", "EMR",
        "ABBN.SW", "KGX.DE", "HEXA-B.ST",
        "6954.T", "6506.T", "6861.T",
    }


def test_medical_humanoid_membership_exact():
    from ingestion.ticker_registry import tickers_in_layer
    assert set(tickers_in_layer("robotics_medical_humanoid")) == {
        "ISRG", "TSLA", "1683.HK", "005380.KS",
    }


def test_mcu_chips_membership_exact():
    from ingestion.ticker_registry import tickers_in_layer
    assert set(tickers_in_layer("robotics_mcu_chips")) == {
        "TXN", "MCHP", "ADI", "6723.T",
    }


def test_no_orphan_robotics_tickers():
    """Every TICKERS_INFO entry whose layer starts with 'robotics_' is in exactly
    one of the three sub-layers."""
    from ingestion.ticker_registry import TICKERS_INFO

    sub_layers = {
        "robotics_industrial",
        "robotics_medical_humanoid",
        "robotics_mcu_chips",
    }
    robotics_entries = [t for t in TICKERS_INFO if t.layer.startswith("robotics")]
    for t in robotics_entries:
        assert t.layer in sub_layers, (
            f"Ticker {t.symbol} has unrecognised robotics layer {t.layer!r}"
        )
    # No legacy flat-robotics layer should remain
    assert all(t.layer != "robotics" for t in TICKERS_INFO)


def test_legacy_robotics_layer_absent():
    """Flat 'robotics' key must not appear in either lookup."""
    from ingestion.ticker_registry import LAYER_IDS, LAYER_LABELS
    assert "robotics" not in LAYER_IDS
    assert "robotics" not in LAYER_LABELS


def test_new_currencies_present_for_humanoid_pillar():
    """1683.HK and 005380.KS introduce HKD and KRW into the registry."""
    from ingestion.ticker_registry import TICKER_CURRENCY
    assert TICKER_CURRENCY["1683.HK"] == "HKD"
    assert TICKER_CURRENCY["005380.KS"] == "KRW"
```

- [ ] **Step 7: Run registry tests — confirm pass**

Run: `pytest tests/test_ticker_registry.py tests/test_robotics_sub_layers.py -v`
Expected: all pass.

- [ ] **Step 8: Run full test suite — confirm green**

Run: `pytest tests/ -m 'not integration' -q`
Expected: 422 + 4 (FX) + 5 (registry — 4 new sub-layer + 1 watchlist, net +1 because 1 robotics test was replaced by 3 new) + 6 (new sub-layers file) = ~433 passing. (Exact number may vary by ±2 — what matters is **0 failures, 0 errors**.)

- [ ] **Step 9: Commit**

```bash
git add ingestion/ticker_registry.py tests/test_ticker_registry.py tests/test_robotics_sub_layers.py
git commit -m "feat: split robotics layer into 3 sub-layers, +8 tickers, +PENDING_IPO_WATCHLIST"
```

---

### Task 4: Pending-IPO startup log line in `run_refresh.sh`

**Files:**
- Modify: `tools/run_refresh.sh`

- [ ] **Step 1: Add log line after "Starting full pipeline refresh"**

In `tools/run_refresh.sh`, replace the line:

```bash
echo "Starting full pipeline refresh at $(date)"
```

with:

```bash
echo "Starting full pipeline refresh at $(date)"
python -c "from ingestion.ticker_registry import PENDING_IPO_WATCHLIST; print(f'  ({len(PENDING_IPO_WATCHLIST)} pending-IPO tickers awaiting listing)')"
```

- [ ] **Step 2: Verify the script parses**

Run: `bash -n tools/run_refresh.sh`
Expected: no output (syntactically valid). **Do not** run the full script — it would trigger the entire refresh.

- [ ] **Step 3: Verify the Python one-liner runs**

Run: `python -c "from ingestion.ticker_registry import PENDING_IPO_WATCHLIST; print(f'  ({len(PENDING_IPO_WATCHLIST)} pending-IPO tickers awaiting listing)')"`
Expected output: `  (2 pending-IPO tickers awaiting listing)`

- [ ] **Step 4: Commit**

```bash
git add tools/run_refresh.sh
git commit -m "chore: log pending-IPO watchlist count at refresh startup"
```

---

### Task 5: Final acceptance gate

**Files:** none (validation only)

- [ ] **Step 1: Run full non-integration suite**

Run: `pytest tests/ -m 'not integration' -q`
Expected: **all green, 0 failures, 0 errors**. Test count ~433 (was 422; +4 FX + ~7 registry).

- [ ] **Step 2: Confirm `fx_features.py` is registry-driven (no edit needed)**

Run: `grep -n 'EUR\|JPY\|GBP\|CHF\|DKK\|SEK\|NOK\|HKD\|KRW' processing/fx_features.py`
Expected: only references that come from `TICKER_CURRENCY` / `CURRENCY_TO_PAIR` (no hard-coded currency literals). If any literal currency strings are found, switch them to derive from `SUPPORTED_CURRENCIES` and re-run tests.

- [ ] **Step 3: Quick sanity — `non_usd_tickers()` returns 40**

Run: `python -c "from ingestion.ticker_registry import non_usd_tickers; print(len(non_usd_tickers()))"`
Expected output: `40`

- [ ] **Step 4: Quick sanity — TICKERS count is 149**

Run: `python -c "from ingestion.ticker_registry import TICKERS; print(len(TICKERS))"`
Expected output: `149`

- [ ] **Step 5: Quick sanity — layers count is 15**

Run: `python -c "from ingestion.ticker_registry import layers; print(len(layers()))"`
Expected output: `15`

- [ ] **Step 6: Push branch (only if all gates pass)**

```bash
git push -u origin feature/robotics-layer-expansion
```

(Optional — wait for user confirmation if pushing to a shared remote.)

---

## Self-review notes

**Spec coverage check:**
- §2 layer restructure → Tasks 2 + 3 ✓
- §3 new exchange codes / FX → Task 1 ✓
- §4 pending-IPO watchlist → Task 3 (constant) + Task 4 (log) ✓
- §5 test plan → Tasks 1, 2, 3 ✓
- §6 rollout file order → Tasks 1 → 3 → 4 ✓
- §6 `processing/fx_features.py` audit → Task 5 step 2 ✓
- §6 `tools/run_refresh.sh` log line → Task 4 ✓

**Type / name consistency:**
- `robotics_industrial`, `robotics_medical_humanoid`, `robotics_mcu_chips` used identically across spec, plan, and code blocks.
- `PENDING_IPO_WATCHLIST` keys (`name`, `expected_symbol`, `layer`, `expected_date`) match between spec, registry, and tests.
- New currency codes `HKD` / `KRW` and pair names `HKDUSD` / `KRWUSD` consistent.

**No placeholders.** Every code block is complete; no "TBD" inside implementation steps (the only `"TBD"` strings are intentional data values inside `PENDING_IPO_WATCHLIST`).
