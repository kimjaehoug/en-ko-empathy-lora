# SELENE TAFFC Results (Qwen3.5-9B)

## Direction I — AI Hub valid (n=3182, utterance S, full valid)

| System | A | S F1 | R | Avg |
|---|---:|---:|---:|---:|
| **F32** | 87.02 | 63.95 | 87.40 | **79.46** |
| F16 | 87.71 | 64.23 | 86.33 | 79.42 |
| B32 | 80.89 | 62.77 | 81.24 | 74.97 |
| B16 | 80.89 | 62.72 | 80.99 | 74.87 |
| S16 | 80.89 | 62.48 | 80.48 | 74.62 |

## Claims (TAFFC)
1. Factor-Bank **F32** leads Dir I Avg (79.5) vs B16 (74.9).
2. v3 soft-share ≈ Blind (negative result) → structural Factor-Bank required.
3. F16 best Dir II among bank variants (see eval log).
4. KoED + confusion + ablation: see outputs/taffc/.
