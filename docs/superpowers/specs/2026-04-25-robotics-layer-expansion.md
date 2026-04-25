# Robotics Layer Expansion — Design Spec

**Date:** 2026-04-25
**Status:** Approved
**Sequencing:** Spec 1 of 2. Spec 2 (Robotics signals ingestion module) follows after this lands.

## Goal

Expand Layer 11 (`robotics`) into three sub-layers, add 8 new tickers (141 → 149), support 2 new currencies (HKD, KRW), and record a watchlist of pending-IPO humanoid plays. No new ingestion or feature modules in this spec.

## Motivation

- Robotics has matured into three distinct economic regimes: industrial automation (cyclical capex), medical/humanoid (TAM-expansion narrative + Tesla-driven), and robotics MCU suppliers (commoditization risk). Splitting them gives the model meaningful within-pillar contrasts and mirrors the existing cyber pillar pattern (`cyber_pureplay` / `cyber_platform`).
- The current 11-ticker robotics layer is missing key public exposures: EMR (process automation), TXN/MCHP/Renesas/ADI (robotics MCU/sensor suppliers), and humanoid plays (TSLA, UBTECH, Hyundai/Boston Dynamics).
- Two pending IPOs (Unitree, Boston Dynamics) should be tracked as placeholders so they can be flipped on quickly when they list.

## Layer restructure

Replace the single `robotics` layer (id=11) with three sub-layers; shift cyber layer ids forward by 2.

| Layer name | id | Tickers (count) |
|---|---:|---|
| `robotics_industrial` | 11 | ROK, ZBRA, CGNX, SYM, ABBN.SW, KGX.DE, HEXA-B.ST, 6954.T, 6506.T, 6861.T, **EMR** (11) |
| `robotics_medical_humanoid` | 12 | ISRG, **TSLA**, **1683.HK**, **005380.KS** (4) |
| `robotics_mcu_chips` | 13 | **TXN**, **MCHP**, **6723.T**, **ADI** (4) |
| `cyber_pureplay` | 14 (was 12) | unchanged (5) |
| `cyber_platform` | 15 (was 13) | unchanged (9) |

**Totals:** 141 → 149 tickers; 13 → 15 layers.

`LAYER_LABELS` adds three entries; the standalone `"robotics"` key is removed.

```python
LAYER_LABELS = {
    ...,
    "robotics_industrial":       "Robotics — Industrial Automation",
    "robotics_medical_humanoid": "Robotics — Medical & Humanoid",
    "robotics_mcu_chips":        "Robotics — MCU & Sensor Chips",
    "cyber_pureplay":            "AI Cybersecurity — Pure Plays",   # id 12 → 14
    "cyber_platform":            "AI Cybersecurity — Platform Vendors",  # id 13 → 15
}
```

## New tickers

| Ticker | Sub-layer | Exchange | Currency | Country | Rationale |
|---|---|---|---|---|---|
| EMR | robotics_industrial | US | USD | US | Process automation (Aspen Tech, Branson). Largest US industrial-automation pure-ish play missing from registry. |
| TSLA | robotics_medical_humanoid | US | USD | US | Optimus humanoid program; 2026 Fremont factory conversion. Only US humanoid public exposure. |
| 1683.HK | robotics_medical_humanoid | HK | HKD | HK | UBTECH Robotics — most commercially successful humanoid pure-play (Walker S). HK-listed Dec 2023. |
| 005380.KS | robotics_medical_humanoid | KS | KRW | KR | Hyundai Motor — owns ~80% of Boston Dynamics. Cleanest BD proxy until BD spin-out. |
| TXN | robotics_mcu_chips | US | USD | US | Largest industrial MCU vendor. |
| MCHP | robotics_mcu_chips | US | USD | US | Microcontroller leader for embedded robotics control. |
| 6723.T | robotics_mcu_chips | T | JPY | JP | Renesas — automotive/industrial MCU leader. |
| ADI | robotics_mcu_chips | US | USD | US | Analog Devices — sensor / motor-control analog for robotics. |

## New exchange codes & FX pairs

`ingestion/fx_ingestion.py` additions:

```python
_FX_SYMBOLS = {
    ...,
    "HKDUSD": "HKDUSD=X",
    "KRWUSD": "KRWUSD=X",
}
SUPPORTED_CURRENCIES = frozenset({"EUR", "CHF", "JPY", "DKK", "SEK", "NOK", "GBP", "HKD", "KRW"})
CURRENCY_TO_PAIR = {
    ...,
    "HKD": "HKDUSD",
    "KRW": "KRWUSD",
}
```

Total FX pairs: 7 → 9.

`processing/fx_features.py`: verify it reads currencies from the registry (no hard-coded currency lists). If it has a hard-coded list, switch it to derive from `SUPPORTED_CURRENCIES` or `non_usd_tickers()`.

## Pending-IPO watchlist

Add a metadata-only constant in `ticker_registry.py`:

```python
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

- Not added to `TICKERS_INFO`. Not fetched. Not joined into features.
- A dedicated test asserts every entry has `name` / `expected_symbol` / `layer` / `expected_date` keys, and that `layer` is a real layer name in `LAYER_IDS`.
- A single log line in `tools/run_refresh.sh` (or wherever ingestion startup logs) reports `"N pending-IPO tickers awaiting listing"` so the team is reminded.

## Test plan

### Modified

`tests/test_ticker_registry.py`
- Replace `test_layers_includes_robotics` with three tests: `_industrial`, `_medical_humanoid`, `_mcu_chips`. Each asserts:
  - `LAYER_IDS[<name>] == <expected_id>`
  - `tickers_in_layer(<name>)` contains the exact expected list (sorted-equality).
- Update `test_layers_includes_cyber_pureplay` / `_platform`: expected ids 14 / 15.
- Update `test_layer_count`: 13 → 15.
- Update `test_ticker_count`: 141 → 149.
- Replace any `result[-3]`-style index assertions with name-based or `LAYER_IDS`-based lookups (more robust to future reordering).
- New `test_pending_ipo_watchlist_structure`: every entry has the 4 required keys; `layer` value exists in `LAYER_IDS`.

`tests/test_fx_ingestion.py`
- Pair-count assertion: 7 → 9.
- New test (mocked yfinance): `fetch_fx_rates(["HKDUSD","KRWUSD"])` returns two non-empty DataFrames with the expected schema.
- Currency-set assertion: `SUPPORTED_CURRENCIES` includes `HKD` and `KRW`.

### New

`tests/test_robotics_sub_layers.py`
- One test per sub-layer asserting exact ticker membership.
- Cross-check: every ticker in `TICKERS_INFO` whose layer starts with `robotics_` is in exactly one of the three sub-layers (no orphans, no duplicates across sub-layers).
- Assertion: legacy literal `"robotics"` is **not** in `LAYER_IDS`.

### Acceptance gate

`pytest tests/ -m 'not integration'` — current baseline 422 passing. After this spec lands, expect ~425+ passing (3 new tests, none broken).

## Files touched

| File | Change |
|---|---|
| `ingestion/fx_ingestion.py` | +HKDUSD/KRWUSD pairs |
| `tests/test_fx_ingestion.py` | +HKD/KRW test, count update |
| `ingestion/ticker_registry.py` | layer split, +8 tickers, shift cyber ids, +PENDING_IPO_WATCHLIST |
| `tests/test_ticker_registry.py` | layer/ticker counts, split robotics test, watchlist test |
| `tests/test_robotics_sub_layers.py` | new file |
| `processing/fx_features.py` | inspect; switch to registry-driven if hard-coded |
| `tools/run_refresh.sh` | +1 startup log line for pending-IPO count (cosmetic) |

## Files NOT touched (verified)

- `models/train.py`, `models/inference.py` — read layers via `LAYER_IDS` / `tickers_in_layer()`; will pick up the new sub-layers transparently.
- `processing/supply_chain_features.py` — iterates via `all_layers()`.
- `ingestion/deal_ingestion.py` — uses `TICKER_LAYERS` mapping.

## Rollout

Single PR, atomic commit ordered:

1. `ingestion/fx_ingestion.py` — additive, no breakage
2. `tests/test_fx_ingestion.py`
3. `ingestion/ticker_registry.py` — layer split + new tickers + watchlist
4. `tests/test_ticker_registry.py`
5. `tests/test_robotics_sub_layers.py`
6. (If needed) `processing/fx_features.py` — registry-driven currency list
7. `tools/run_refresh.sh` — startup log
8. Run `pytest tests/ -m 'not integration'`
9. Commit: `feat: split robotics layer into 3 sub-layers, +8 tickers, HKD/KRW FX support`

## Risks & out-of-scope

- **Per-layer model checkpoints** keyed by old `"robotics"` name (under `data/models/per_layer/`, if any) become stale. Acceptable — next training run regenerates them. The model isn't trained at full scale yet.
- **Backtest result files** (e.g., `data/backtest/walk_forward_results.json`) referencing old layer ids will be stale. A fresh backtest after merging is required but out-of-scope for this spec.
- **Spec 2 (robotics signals ingestion)** is sequenced after this lands; that adds new feature columns and a dedicated ingestion module. None of that work belongs in this spec.

## Out-of-scope (deferred)

- Industrial automation order indices, IFR robot shipments, ISM PMI new orders → Spec 2.
- Robotics-specific cross-ticker features → Spec 2.
- Hyundai (`005380.KS`) decomposition into auto-only vs. BD-only signal → not feasible without segment financials.
- Adding Unitree / Boston Dynamics tickers — will be a small follow-up once each lists.
