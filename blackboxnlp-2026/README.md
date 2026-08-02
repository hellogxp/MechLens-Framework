# BlackboxNLP 2026 artifact

This directory contains every submission-specific component of the anonymous artifact:

```text
blackboxnlp-2026/
├── paper/          Anonymous manuscript, figures, and compiled PDF
├── experiments/    Corrected measurement and analysis drivers
└── results/        Canonical compressed artifacts, checksums, and tables
```

Reusable implementation and tests remain at repository level under `src/` and `tests/`.

From the repository root, regenerate tables and figures with:

```bash
python blackboxnlp-2026/experiments/analyze_corrected_fep.py
```

The canonical runner writes new outputs to `blackboxnlp-2026/results/corrected_fep/` by default. Build the paper with `blackboxnlp-2026/paper/compile.sh`.
