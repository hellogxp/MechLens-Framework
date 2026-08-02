# First Is Not Stable: MechLens audit artifact

This anonymous artifact accompanies the BlackboxNLP 2026 manuscript **“First Is Not Stable: Auditing Layerwise Vocabulary Readout in Language Models.”** It contains the corrected, trajectory-aware measurement code and canonical paper results.

## Measurement contract

- Jointly tokenize prompt and continuation and verify the exact prompt prefix.
- Store never-observed top-k events as censored (`null`), not as final-layer entry.
- Separate first entry, persistent entry, dropout, final presence, and non-entry.
- Validate the final projected state against native model logits.
- Keep candidate-set accuracy separate from full-vocabulary rank.

## Canonical evidence

All submission-specific material is isolated under `blackboxnlp-2026/`:

- `blackboxnlp-2026/results/corrected_fep/`: six-model TruthfulQA, Qwen MMLU, and five-model SST-2 trajectories with checksums.
- `blackboxnlp-2026/results/prompt_sensitivity/`: two additional full TruthfulQA prompt-template reruns.
- `blackboxnlp-2026/results/analysis/`: paper tables, intervals, threshold sensitivity, and paired tests.
- `blackboxnlp-2026/paper/`: anonymous source and compiled six-page PDF.

The main finding is methodological: vocabulary readout is sensitive to censoring, trajectory stability, prompt wording, and answer-space format. It is not evidence of where knowledge is stored or first used causally.

## Reproduction

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q tests/unit/test_fep.py tests/unit/test_fep_analysis.py
python blackboxnlp-2026/experiments/analyze_corrected_fep.py
```

GPU reruns use `blackboxnlp-2026/experiments/run_corrected_fep.py`. Canonical runs used Python 3.12.13, PyTorch 2.10.0, Transformers 4.48.3, CUDA 12.8, bfloat16, one NVIDIA H20, and seed 42.

## License

This project is licensed under the MIT License.
