# TimesFM diagnostic baseline (Phase 2.12)

Generated: 2026-05-15

Restricted to post-TimesFM-2.0-training-cutoff dates (>= 2024-06-01) to avoid lookahead bias in TimesFM's input sequences.

Verdict rules (pre-committed):
- **TimesFM wins**: Δ IC ≥ +0.010 AND TimesFM mean IC > 0
- **Ensemble wins**: Δ IC ≤ -0.010
- **Comparable**: otherwise

## Per-(layer, horizon) results

| layer | horizon | n_dates | TimesFM IC | Ensemble IC | Δ IC | Verdict |
|---|---|---:|---:|---:|---:|---|
| cloud | 252d | 119 | -0.0550 | -0.2113 | +0.1563 | Comparable |
| compute | 252d | 119 | +0.1625 | +0.0638 | +0.0987 | TimesFM wins |
| cooling | 252d | 119 | -0.0794 | -0.2400 | +0.1606 | Comparable |
| cyber_platform | 252d | 119 | +0.2271 | +0.1467 | +0.0804 | TimesFM wins |
| cyber_pureplay | 252d | 119 | +0.0214 | +0.0238 | -0.0024 | Comparable |
| datacenter | 252d | 119 | -0.3297 | -0.3339 | +0.0042 | Comparable |
| enterprise_saas | 252d | 119 | -0.0336 | +0.2464 | -0.2801 | Ensemble wins |
| grid | 252d | 119 | +0.1317 | -0.0287 | +0.1604 | TimesFM wins |
| metals | 252d | 119 | -0.1132 | -0.0471 | -0.0661 | Ensemble wins |
| networking | 252d | 119 | +0.0368 | +0.3893 | -0.3525 | Ensemble wins |
| power | 252d | 119 | +0.0742 | +0.2702 | -0.1959 | Ensemble wins |
| robotics_industrial | 252d | 119 | -0.1992 | +0.0845 | -0.2837 | Ensemble wins |
| robotics_mcu_chips | 252d | 119 | +0.0957 | -0.0987 | +0.1944 | TimesFM wins |
| robotics_medical_humanoid | 252d | 119 | +0.0044 | +0.0476 | -0.0432 | Ensemble wins |
| semi_equipment | 252d | 119 | +0.1057 | +0.0331 | +0.0726 | TimesFM wins |
| servers | 252d | 119 | -0.1481 | +0.3087 | -0.4569 | Ensemble wins |
| cloud | 65d | 82 | -0.1918 | +0.1198 | -0.3116 | Ensemble wins |
| compute | 65d | 82 | +0.3167 | -0.2029 | +0.5196 | TimesFM wins |
| cooling | 65d | 82 | -0.2187 | +0.3572 | -0.5760 | Ensemble wins |
| cyber_platform | 65d | 82 | +0.3318 | +0.1558 | +0.1761 | TimesFM wins |
| cyber_pureplay | 65d | 82 | +0.0167 | -0.2495 | +0.2662 | TimesFM wins |
| datacenter | 65d | 82 | +0.1812 | +0.0812 | +0.1000 | TimesFM wins |
| enterprise_saas | 65d | 82 | +0.0358 | +0.0309 | +0.0049 | Comparable |
| grid | 65d | 82 | +0.0575 | +0.4445 | -0.3871 | Ensemble wins |
| metals | 65d | 82 | +0.0627 | -0.2568 | +0.3195 | TimesFM wins |
| networking | 65d | 82 | +0.2382 | +0.0469 | +0.1913 | TimesFM wins |
| power | 65d | 82 | -0.0044 | -0.3883 | +0.3839 | Comparable |
| robotics_industrial | 65d | 82 | +0.2853 | +0.0572 | +0.2282 | TimesFM wins |
| robotics_mcu_chips | 65d | 82 | -0.1089 | +0.4590 | -0.5679 | Ensemble wins |
| robotics_medical_humanoid | 65d | 82 | +0.6188 | -0.2624 | +0.8812 | TimesFM wins |
| semi_equipment | 65d | 82 | -0.0336 | +0.3328 | -0.3663 | Ensemble wins |
| servers | 65d | 82 | +0.2850 | +0.0752 | +0.2098 | TimesFM wins |

## Verdict counts

| verdict | n cells |
|---|---:|
| Comparable | 6 |
| Ensemble wins | 12 |
| TimesFM wins | 14 |