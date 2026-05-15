# AI-PRED — Architecture v2

**State as of 2026-05-15, branch `experiment/architecture-v2`, 13 post-baseline phases shipped.**

This is the operator's reference: what the system does, what's validated, how to
run it, and what's known to be limited. It's meant to be read once when picking
up the project after a 30-60 day gap, without re-reading commit messages.

Master is locked at `v0.1-baseline-2026-04-28`. Architecture-v2 has not yet
been merged. The branch is in "watch forward" mode pending out-of-sample
evidence from the rolling realized log.

---

## What is deployed

Two cross-sectional alpha signals, served by default at the inference layer:

| Horizon | Ablation | Portfolio | Smoothed annualized return | Sharpe | Max drawdown | Sample (forward folds at branch state) |
|---|---|---|---:|---:|---:|---:|
| **65d** | `no_ai_infra` (51 features) | 5% global decile, L/S | **+70%** | **2.36** | −9.2% | 82 matured |
| **252d** | `deep_only` (20 features, sign-flipped per E5) | 5% global decile, L/S | **+12%** | **1.03** | −14.2% | 753 matured |

A regime overlay (`tools/macro_overlay.py`) scales gross notional in
`[0.3, 1.0]` based on a VIX / yield-curve / DXY composite. At current
conditions (2026-05-15) the overlay sizes at 1.00× (benign regime).

---

## What is NOT validated

| Horizon | Status |
|---|---|
| 5d | NEUTRAL (Phase D) — no alpha to deploy |
| 20d | NEUTRAL (Phase E1) — no alpha to deploy |
| **65d** | **deployed (primary signal)** |
| **252d** | **deployed (weaker, sign-flipped)** |
| 756d (3y) | no trained artifacts; never validated |
| 1260d (5y) | no trained artifacts; never validated |
| 2520d (10y) | no trained artifacts; never validated |
| 5040d (20y) | no trained artifacts; never validated |

Long horizons are blocked by three structural issues documented in Phase E2:
shallow feature history, basket curation bias (registry assembled in 2026,
projection back to 2010 captures survivors only), and the absence of
walk-forward retraining. They cannot be backtested credibly today.

---

## End-to-end flow

```
                   ┌─────────────────────────────────┐
                   │ tools/run_refresh.sh (25 steps) │  ingestion refresh
                   └─────────┬───────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ data/raw/                                                   │
   │   financials/ohlcv/{TICKER}/*.parquet           (1962+)     │
   │   financials/fundamentals/{TICKER}/*.parquet    (2008+)     │
   │   financials/13f_holdings/                      (2013+)     │
   │   financials/insider_trades/                    (1997+)     │
   │   robotics_signals/{FRED}.parquet               (2010+)     │
   │   financials/fx/, news/, cyber_threat/, etc.    (varies)    │
   └─────────────────────────┬───────────────────────────────────┘
                             ▼
              ┌──────────────────────────────────────┐
              │ models/train.py                       │
              │   per-layer × per-horizon ensembles:  │
              │   LGBM(q10,q50,q90) + RF + Ridge      │
              │   NNLS-weighted from 3-fold CV        │
              │   ablations: none / no_ai_infra /     │
              │              deep_only                 │
              └──────────────┬───────────────────────┘
                             ▼
       ┌────────────────────────────────────────────────────┐
       │ models/artifacts/walkforward/cutoff=*/             │
       │   ablation={none, no_ai_infra, deep_only}/         │
       │   layer_NN_<name>/horizon_<H>/*.pkl + .json        │
       └──────────────┬─────────────────────────────────────┘
                      ▼
   ┌────────────────────────────────────────────────────────────┐
   │ tools/daily_inference.py                                    │
   │   - resolves per-horizon ablation default (HORIZON_ABLATION │
   │     _DEFAULTS) and sign convention (HORIZON_SIGN_CONVENTION)│
   │   - batched feature build per holdout range                 │
   │   - writes predictions to                                   │
   │       data/predictions/walkforward/cutoff=*/ablation=*/     │
   │         date=YYYY-MM-DD/horizon=<H>/predictions.parquet     │
   └──────────────┬──────────────────────────────────────────────┘
                  ▼
   ┌────────────────────────────────────────────────────────────┐
   │ tools/rolling_score.py                                     │
   │   - finds matured predictions, joins to realized returns    │
   │   - applies HORIZON_PORTFOLIO_DEFAULTS (5% global @ 65/252) │
   │   - appends to data/scoring_log.parquet                    │
   └──────────────┬──────────────────────────────────────────────┘
                  ▼
   ┌────────────────────────────────────────────────────────────┐
   │ tools/macro_overlay.py                                      │
   │   - VIX / yield curve / DXY composite z-score              │
   │   - macro_risk_score → gross_scale ∈ [0.3, 1.0]            │
   │   - data/macro_overlay_log.parquet                         │
   └──────────────┬──────────────────────────────────────────────┘
                  ▼
   ┌────────────────────────────────────────────────────────────┐
   │ tools/daily_report.py                                      │
   │   prints: macro overlay + top long/short picks + rolling   │
   │   realized metrics to stdout                                │
   └────────────────────────────────────────────────────────────┘

       all five steps above run by tools/daily_scheduler.py
       (APScheduler BlockingScheduler, mon-fri 23:00 local)
```

---

## How to run it

### Continuous daemon (intended)

```bash
python -m tools.daily_scheduler                            # foreground, 23:00 local
python -m tools.daily_scheduler --time 16:30 --tz America/New_York
python -m tools.daily_scheduler --run-once                 # test chain immediately
```

Wrap in `systemd` / `launchd` / `nohup` for survivable deployment.

Logs land under `data/scheduler_logs/YYYY-MM-DD/<step>.log` per step; a
cumulative summary line is appended to `data/scheduler_logs/scheduler.log`
after each run.

### Manual catch-up (if the daemon was down)

```bash
python -m tools.daily_inference     # forward predictions for any missing dates
python -m tools.rolling_score       # score newly matured predictions
python -m tools.macro_overlay       # today's macro signal
python -m tools.daily_report        # operator view to stdout
```

All four are idempotent; re-running on the same day is a no-op.

### Re-running audits (manual, not part of daily cron)

```bash
python -m tools.feature_history_audit          # Phase E2
python -m tools.robustness_audit               # Phase 2.5
python -m tools.drawdown_audit                 # Phase 2.6
python -m tools.smoothed_portfolio_sim         # Phase 2.7
python -m tools.calibration_check              # Phase 2.11
python -m tools.timesfm_diagnostic_baseline    # Phase 2.12 (~45 min CPU)
python -m tools.inverted_prediction_check      # Phase E4
```

---

## Known limitations

| Area | Limitation | Source / detail |
|---|---|---|
| Quantile bands | Both q10/q90 miscalibrated; do NOT use for sizing | Phase 2.11. At 65d, real distribution is ~1.5× wider than predicted. At 252d, all coverages systematically too low (AI bull market exceeded training-period levels). |
| 252d / FULL post-flip | Collapses after E5 sign flip; only DEEP_ONLY is the validated 252d path | Phase 2.2 retest |
| Sector-neutral default in harness | Backtest harness defaults to sector-neutral 10% for A/B coherence — DO NOT confuse with production | Phase 2.8; production uses 5% global per `HORIZON_PORTFOLIO_DEFAULTS` |
| GDELT sentiment | Per-ticker fetcher exists but rate-limited; `sentiment_features` are still ~empty | Phase 2.9 |
| USPTO patent features | Source decommissioned (PatentsView v1 dead); features always-null | Phase 2.2-pivot; memory `project_uspto_migration_deferred.md` |
| Macro overlay informational | `gross_scale` printed in daily report but not auto-applied to position size | Phase 2.10; operator applies manually when sizing real positions |
| Forward-prediction storage at 5y+ | We don't write predictions for unvalidated long horizons — would just clutter disk with untestable numbers | by design |

---

## Key constants and where they live

All in `models/train.py`:

| Constant | Purpose |
|---|---|
| `FEATURE_COLS` | 112 features (kept intact; ablations subset at training time) |
| `HORIZON_CONFIGS` | 8 horizons + tier mapping (short / medium / long) |
| `TIER_FEATURE_COLS` | per-tier feature subset (short tier excludes ownership; long tier includes long-cycle features) |
| `HORIZON_SIGN_CONVENTION` | per-horizon ±1 applied to `expected_annual_return` at inference. 5d / 20d / 65d / 756d+ = +1; **252d = −1** (Phase E5) |
| `HORIZON_ABLATION_DEFAULTS` | per-horizon production ablation default. 65d → no_ai_infra, 252d+ → deep_only, others → none (Phase 2.3A) |
| `HORIZON_PORTFOLIO_DEFAULTS` | per-horizon (decile_pct, sector_neutral). 5d/20d → (10%, sector-neutral) for backtest stability; 65d/252d+ → (5%, global) for production deployment (Phase 2.8) |
| `_AI_INFRA_FEATURE_COLS` | 61 differentiated features (energy_geo, supply_chain, cyber_threat, gov_behavioral, patents, labor, census_trade, physical_ai, ai_economics) — excluded by `no_ai_infra` ablation |
| `_DEEP_FEATURE_COLS` | 20 deep-history features (price + fundamentals only) — `deep_only` ablation |
| `_ABLATION_FILTERS` | dict mapping ablation tag → feature filter function |

---

## Decision rules (pre-committed, before observing forward data)

These rules are pre-committed so future-you can't move the goalposts after
looking at the rolling log.

### Trigger to retrain or investigate

- **Rolling 20-fold IC drops below 0** for 4 consecutive weekly observations
  on the 65d signal → investigate; the model has degraded.
- **Rolling 60-fold IC drops below half** of the all-time baseline (currently
  +0.079 at 65d) → investigate; may indicate regime shift.
- **macro_risk_score > 0.7** for 5+ consecutive trading days → no manual
  action required; `gross_scale` already adjusts. Just notice and confirm
  the daemon is responding.

### Trigger to size up

- **60+ new forward-realized folds at 65d maintain Sharpe > 1.5** in
  smoothed simulation → consider increasing position sizing on the 65d
  signal. The all-time Sharpe of 2.36 is from in-sample-adjacent walk-
  forward; forward observation at Sharpe > 1.5 is real confirmation.
- **252d sample reaches >120 post-2024-06 matured folds** (~6 months from
  branch state) → re-run `tools.smoothed_portfolio_sim` to update the 252d
  deployment numbers with cleaner post-cutoff data.

### Trigger to merge architecture-v2 → master

- 60+ forward folds at 65d with rolling-60 IC > +0.05 AND Sharpe > 1.5
  → architecture-v2 has held up out-of-sample → merge.
- If rolling-60 IC drops below 0 in the same window → revert architecture-v2
  to baseline; the validation was curve-fit.

### No-action zone

- Single weekly observations, any direction. The standard error on a single
  fold's IC is ~0.10; single-week noise should not move decisions.

---

## Repo inventory

### `models/`

| File | Purpose |
|---|---|
| `train.py` | per-(layer, horizon) ensemble training; constants listed above |
| `inference.py` | per-date inference; applies `HORIZON_ABLATION_DEFAULTS` + `HORIZON_SIGN_CONVENTION` |

### `tools/` (in commit order)

| File | Phase | Purpose |
|---|---|---|
| `feature_history_audit.py` | E2 | data depth audit (which features actually have history) |
| `inverted_prediction_check.py` | E4 | check whether negating predictions improves IC (diagnoses sign issues) |
| `backtest_walk_forward.py` | C | walk-forward backtest harness (the canonical scorer) |
| `walkforward_inference.py` | C v2 | batched walkforward inference (multi-date) |
| `daily_inference.py` | 2.4 | daily catch-up forward predictions |
| `rolling_score.py` | 2.4b | rolling realized log |
| `daily_report.py` | 2.4c | operator view |
| `robustness_audit.py` | 2.5 | (decile %, sector_neutral, tc, borrow) variant grid |
| `drawdown_audit.py` | 2.6 | per-variant return distribution (p5, min, WLS, maxDD) |
| `smoothed_portfolio_sim.py` | 2.7 | smoothed daily-rebalance simulation (real deployment numbers) |
| `macro_overlay.py` | 2.10 | VIX/yield-curve/DXY composite → gross_scale |
| `calibration_check.py` | 2.11 | one-shot q10/q90 coverage |
| `timesfm_diagnostic_baseline.py` | 2.12 | per-cell TimesFM vs ensemble comparison |
| `daily_scheduler.py` | 2.13 | APScheduler daemon for the 5-step chain |

### `data/` (gitignored except where noted)

| Path | Purpose |
|---|---|
| `raw/financials/ohlcv/<TICKER>/*.parquet` | OHLCV (1962+) |
| `raw/financials/fundamentals/<TICKER>/*.parquet` | SEC fundamentals (2008+) |
| `raw/financials/13f_holdings/raw/<YYYYQQ>/*.parquet` | per-filer 13F filings (2013+) |
| `raw/financials/13f_holdings/features/<TICKER>/*.parquet` | aggregated 13F features |
| `raw/financials/insider_trades/<TICKER>/transactions.parquet` | SEC Form 4 P/S (1997+) |
| `raw/financials/insider_features/<TICKER>/daily.parquet` | aggregated insider features |
| `predictions/walkforward/cutoff=<DATE>/ablation=<NAME>/date=<D>/horizon=<H>/predictions.parquet` | per-date predictions |
| `scoring_log.parquet` | rolling realized performance log |
| `macro_overlay_log.parquet` | daily macro indicators |
| `scheduler_logs/YYYY-MM-DD/<step>.log` | per-step daemon logs |
| `scheduler_logs/scheduler.log` | cumulative chain summaries |

### `reports/` (committed)

| Path | Purpose |
|---|---|
| `timesfm/diagnostic_baseline.md` | Phase 2.12 per-cell verdict table |

---

## Phase chronicle (one sentence each)

| Phase | Headline finding |
|---|---|
| A | point-in-time correctness on all features; PIT regression tests in place |
| B | sector-residualized excess-return target as a second-class alongside raw |
| C v2 | walk-forward retraining with strict label cutoff (no future leak) |
| D | 5d ablation: AI-infra block adds zero incremental IC, NEUTRAL verdict |
| E1 | 65d emerges as the cleanest validated horizon; 20d NEUTRAL; 65d IC=+0.049 |
| E2 | feature-data history audit — most "differentiated" features have months not years |
| E3 | deep_only at 252d: both ablations have negative IC, anti-predictive at 1y |
| E4 | inverting predictions at 252d gives IC=+0.0351, real cross-sectional signal with wrong sign |
| E5 | productionize sign convention per horizon; 252d gets ×−1 at inference |
| 2.1 | 13F backfill (3 months → 13 years): 65d FULL +280 bps lift |
| 2.1 retest | confirmed lift; FULL beat NO_AI temporarily |
| 2.2 | Form 4 backfill (5 yr → 29 yr); helped NO_AI, hurt FULL |
| 2.2 retest | NO_AI/65d = +428 bps, the new production winner |
| 2.3A | per-horizon ablation defaults; production routes 65d→no_ai_infra, 252d→deep_only |
| 2.4 | daily forward inference accumulates new predictions |
| 2.4b | rolling realized log measures live performance |
| 2.4c | daily report combines picks + metrics |
| 2.5 | robustness audit: alpha is cost-robust; sector-neutral is the ×4 binding constraint |
| 2.6 | drawdown audit: 65d distribution clean (−4% maxDD), 252d heavy tails (−47% worst fold) |
| 2.7 | smoothed daily-rebalance: real-deployment numbers (65d +70% ann, 252d +12% ann) |
| 2.8 | production portfolio defaults: 5% global decile is the deployable config |
| 2.9 | GDELT per-ticker integration: opt-in only (rate limits prevent daily-default) |
| 2.10 | macro overlay (VIX/yield-curve/DXY composite) — applies at all horizons |
| 2.11 | quantile bands not calibrated; production correctly uses rank not level |
| 2.12 | TimesFM diagnostic: Outcome B (mixed), don't integrate; per-layer IC dispersion is the surprising finding |
| 2.13 | APScheduler daemon runs the 5-step chain weekday 23:00; no more manual operation |

---

## Last-mile checklist before relying on this

If/when the rolling log accumulates 60+ post-branch forward folds with
Sharpe > 1.5, before sizing real money:

1. Re-run `tools.smoothed_portfolio_sim` to refresh the deployment numbers
   on the full out-of-sample window.
2. Re-run `tools.calibration_check` to confirm bands still don't match
   reality (we expect that conclusion to hold; verify).
3. Re-run `tools.robustness_audit` with the new out-of-sample folds to
   confirm the sector-neutral-vs-global gap survives outside backtest data.
4. Tag the validated state on `experiment/architecture-v2`, then merge to
   `master`.
5. Pick execution path (manual vs broker API) — this is off-system work.

If the rolling log shows degradation instead — revert to baseline. The
validation was curve-fit, and the next investment should be a fresh
methodology, not another patch.
