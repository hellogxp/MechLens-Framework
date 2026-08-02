# TruthfulQA prompt-template sensitivity

Controlled Qwen2.5-7B reruns on all 817 TruthfulQA questions. These runs use
the same model, targets, threshold (`k=10`), dtype, and sample order as the
canonical `Q: ... / A:` run. Only the prompt wording changes.

| Template | Ever top-10 | Never top-10 | Final top-10 | Dropout | Mean first / persistent depth |
|---|---:|---:|---:|---:|---:|
| `Q: ... / A:` | 68.91% | 31.09% | 64.75% | 18.85% | .824 / .876 |
| `Question: ... / Answer:` | 69.65% | 30.35% | 65.73% | 22.28% | .818 / .865 |
| Truthfulness instruction + `Question/Answer` | 77.23% | 22.77% | 74.42% | 16.65% | .811 / .842 |

The abbreviated and expanded neutral templates are close (70 paired final-
visibility gains versus 62 losses; exact McNemar `p=0.543`). Adding a
truthfulness instruction increases final top-10 visibility by 9.67 percentage
points (110 gains versus 31 losses; `p=1.44e-11`). FEP is therefore not a
model-only property. All runs have 817 results and zero failures.

The original `Q/A` artifact is stored in `../corrected_fep/`. The two files in
this directory are deterministic gzip archives of the additional templates.
