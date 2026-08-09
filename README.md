# Layerwise Vocabulary Readout Audit — Reproducibility Artifact

This repository is the supplementary artifact for an anonymous BlackboxNLP
2026 submission. The manuscript is submitted separately through OpenReview and
is intentionally not included here.

The artifact contains the corrected, trajectory-aware measurement code,
canonical compressed outputs, analysis tables, and generated figures used in
the submission. Never-observed top-k events are treated as censored rather than
as final-layer entries; first entry, persistent entry, later disappearance, and
final presence are reported separately.

## Layout

- `src/mechlens/`: unit-tested trajectory and statistical helpers.
- `blackboxnlp-2026/experiments/`: experiment and analysis entry points.
- `blackboxnlp-2026/results/corrected_fep/`: canonical compressed runs.
- `blackboxnlp-2026/results/prompt_sensitivity/`: controlled prompt reruns.
- `blackboxnlp-2026/results/analysis/`: released analysis CSVs.
- `blackboxnlp-2026/generated/`: tables and figures derived from the runs.
- `tests/unit/`: CPU-only tests for the measurement contract.

## Install and verify

Python 3.10 or newer is required. The canonical runs used Python 3.12,
Transformers 4.48.3, and PyTorch 2.10.0.

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests blackboxnlp-2026/experiments
```

Verify the released compressed artifacts:

```bash
(cd blackboxnlp-2026/results/corrected_fep && sha256sum -c checksums.sha256)
(cd blackboxnlp-2026/results/prompt_sensitivity && sha256sum -c checksums.sha256)
```

Regenerate all analysis CSVs, tables, and figures:

```bash
python blackboxnlp-2026/experiments/analyze_corrected_fep.py
```

The experiment runner exposes `--model`, `--dataset`, sampling, model-revision,
dataset-revision, and output options:

```bash
python blackboxnlp-2026/experiments/run_corrected_fep.py --help
```

Large model weights and public benchmark datasets are downloaded from their
original providers and are not redistributed in this artifact.

