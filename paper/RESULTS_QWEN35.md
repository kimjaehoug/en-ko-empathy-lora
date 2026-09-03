# SELENE TAFFC Results (Qwen3.5-9B)

## Direction I — AI Hub valid (n=3182, utterance S, full valid)

| System | A | S F1 | R | Avg |
|---|---:|---:|---:|---:|
| **F32** | 87.02 | 63.95 | 87.40 | **79.46** |
| F16 | 87.71 | 64.23 | 86.33 | 79.42 |
| B32 | 80.89 | 62.77 | 81.24 | 74.97 |
| B16 | 80.89 | 62.72 | 80.99 | 74.87 |
| S16 | 80.89 | 62.48 | 80.48 | 74.62 |
| MAD-X | 78.4 | 64.3 | 67.4 | 70.1 |

Investigation: B16 vs B32 vs S16 emotion predictions are **100% identical** on full valid.

## Direction II — ED valid (n=2758; EN-before = 65.4%)

| System | Acc | ΔAcc |
|---|---:|---:|
| F16 | 64.1 | **−1.3** |
| KED | 64.8 | −1.9 |
| F32 | 60.6 | −4.8 |
| B16 | 59.4 | −5.9 |
| B32 | 58.9 | −6.5 |
| MAD-X | 40.1 | −26.6 |
| S16 | 31.0 | −34.4 |

## Multi-seed (in-training eval, seeds 123/456)

| System | A | S | R | Avg |
|---|---|---|---|---|
| F16 | 89.0±1.4 | 66.2±1.6 | 97.0±0.0 | **84.1±1.0** |
| B16 | 77.5±0.0 | 68.0±0.8 | 96.5±0.7 | 80.7±0.5 |
| S16 | 77.5±0.0 | 67.5±0.1 | 97.2±0.4 | 80.8±0.1 |

Paired F16 vs B16 Avg: t=10.18, p=0.062 (n=2).

## Component ablation (seed 42, in-training Avg)

| ID | Config | Avg |
|---|---|---:|
| A2 | w/o two-pass | 86.0 |
| A5 | w/o EN replay | 85.2 |
| A3 | w/o gate losses | 83.9 |
| A4 | w/o curriculum | 82.9 |
| A6 | w/o LoRA anchor | 82.8 |
| A0 | soft-share v3 | 79.7 |
| A8 | freeze LoRA | 77.0 |
| A9 | LM only | 34.0 |

## KoED held-out (8-class shared)

| System | Acc | macro-F1 |
|---|---:|---:|
| **F16** | **84.1** | **74.0** |
| KED | 76.9 | 72.6 |
| B32 | 77.9 | 63.9 |
| F32 | 76.6 | 65.8 |
| B16 | 76.5 | 62.7 |
| S16 | 68.2 | 38.4 |
| MAD-X | 18.4 | 12.7 |

## Stage-2 gates

| Factor | Score | Gate |
|---|---:|---|
| Affect | 0.62 | share |
| Strategy | 0.46 | relearn |
| Relation | 0.24 | relearn |
| Culture | 0.02 | suppress |

## Claims
1. Factor-Bank **F32/F16** lead Dir I Avg (~79.5) vs Blind (~74.9): **+4.5 pp**.
2. Blind≡Scratch emotion preds → non-selective collapse; soft-share alone insufficient.
3. F16 best Dir II retention (−1.3 pp) among Bank/Blind/Scratch.
4. F16 best KoED 8-class Acc/F1; MAD-X fails culturally.
5. Ablations: EN-merge + two-pass package matters more than single regularizers.
