# AI-PRED — decision log

Append-only log of when pre-committed decision rules trigger and what
action was taken. The point of the log is **discipline** — if you don't
write down "I changed my mind because of [X]," you'll forget that you
moved goalposts under noise.

Reference for the rules themselves: `docs/architecture-v2-design.md` →
**Decision rules** section. Rules below are reproduced for convenience
but the design doc is the source of truth.

---

## Pre-committed rules (do not edit retroactively)

These were set on **2026-05-15** at the `v0.2-arch-v2-frozen` tag, before
any forward observation period began.

### Investigate / retrain

| Rule | Action if triggered |
|---|---|
| Rolling 20-fold IC at 65d drops below 0 for **4 consecutive weeks** | Investigate the rolling log + recent regime conditions. Don't retrain yet. |
| Rolling 60-fold IC at 65d drops below **+0.040** (half of all-time +0.079) | Investigate; consider retraining at a more recent cutoff with the same architecture. |
| Rolling 60-fold Sharpe at 65d (smoothed) drops below 0.75 | Same as above; the alpha has materially degraded. |

### Promote / size up

| Rule | Action if triggered |
|---|---|
| 60+ new forward-realized folds at 65d maintain Sharpe > 1.5 in smoothed simulation | Consider increasing position sizing on the 65d signal. Real out-of-sample evidence has confirmed the validated state. |
| 252d sample reaches >120 post-2024-06 matured folds | Re-run `tools.smoothed_portfolio_sim` to update 252d numbers; decide whether 252d deserves more weight. |

### Merge architecture-v2 → master

| Rule | Action |
|---|---|
| 60+ forward folds at 65d with rolling-60 IC > +0.05 AND Sharpe > 1.5 | Merge architecture-v2 → master. Bump tag to v0.3 or v1.0 depending on real-money status. |
| Rolling-60 IC drops below 0 in the same observation window | Revert architecture-v2. Don't merge. The validation was curve-fit. |

### No-action zone (do not respond)

- Single weekly IC fluctuations either direction. Std error on a single
  fold's IC ≈ 0.10; a single bad week is meaningless.
- One bad day on the macro overlay (e.g. spike in VIX) — `gross_scale`
  already adjusts. No manual override needed.
- A bad single-fold L/S realized return — the L/S basket has noise; the
  rolling mean is the metric, not any single fold.

---

## Baseline observations at tag time

For reference, here's the state at `v0.2-arch-v2-frozen` (2026-05-15). All
future entries should compare against these.

| Metric | Value at tag | Source |
|---|---|---|
| 65d rolling all-time IC | +0.0785 | `data/scoring_log.parquet`, n=82 |
| 65d rolling last-60 IC | +0.0877 | same, last 60 folds |
| 65d rolling last-20 IC | +0.1861 | same, last 20 folds |
| 65d smoothed ann return | +70% | `tools.smoothed_portfolio_sim` |
| 65d smoothed Sharpe | 2.36 | same |
| 252d rolling all-time IC | +0.0351 | scoring_log, n=753 |
| 252d smoothed ann return | +12% | smoothed sim |
| 252d smoothed Sharpe | 1.03 | same |
| macro_risk_score (today) | 0.43 (neutral) | `tools.macro_overlay` |
| gross_scale (today) | 1.00 (full size) | same |

---

## Entries

Append below this line. Each entry must include date, metric that
triggered, action taken, and a one-line reason. Never edit existing
entries; corrections go in new entries.

Template:

```
### YYYY-MM-DD — <short title>

**Trigger**: <which rule>
**Observation**: <metric value when triggered>
**Action**: <what you did, or explicitly: "no action, monitor">
**Reason**: <one sentence; the WHY>
```

---

### 2026-05-15 — pause begins

**Trigger**: tag `v0.2-arch-v2-frozen` created; forward observation period starts.
**Observation**: baseline metrics above. 82 65d folds, 753 252d folds, both with positive validated alpha.
**Action**: launch `tools.daily_scheduler` daemon; no model changes; wait.
**Reason**: every additional research-grade investment has lower expected value than letting the rolling log accumulate forward observations. Re-evaluate at +60 forward folds (~2-3 months calendar time at current matured-fold rate).
