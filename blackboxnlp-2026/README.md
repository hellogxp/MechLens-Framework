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

In addition to the paper summaries, this writes per-layer leave-one-readout-out
sensitivity, paired final-visibility and first-entry-depth comparisons,
TruthfulQA target-composition audits, and logical dataset hashes under
`results/analysis/`.

Run a new corrected measurement with, for example:

```bash
python blackboxnlp-2026/experiments/run_corrected_fep.py \
  --model qwen7 --dataset truthfulqa --max-samples 50
```

New runner outputs default to `blackboxnlp-2026/results/corrected_fep/`.

## Build the paper

```bash
cd blackboxnlp-2026/paper
./compile.sh
```

`paper/main.pdf` is the current anonymous submission artifact. Before
uploading, verify that the PDF and repository remain anonymous.
