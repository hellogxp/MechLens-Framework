# First Is Not Stable: MechLens audit artifact

This anonymous artifact accompanies the BlackboxNLP 2026 manuscript **“First Is
Not Stable: Auditing Layerwise Vocabulary Readout in Language Models.”** It
contains the corrected, trajectory-aware measurement code and the canonical
results used by the paper.

## Measurement contract

- Extract the first continuation token by jointly tokenizing prompt and answer
  and verifying the exact prompt prefix.
- Store targets that never enter vocabulary top-k as censored (`null`), not as
  final-layer events.
- Report first entry, persistent entry, dropout after entry, final presence, and
  non-entry separately.
- Validate the final projected state against native model logits.
- Keep candidate-set accuracy separate from full-vocabulary rank.

## Canonical evidence

- `results/corrected_fep/`: six-model TruthfulQA, Qwen MMLU, and five-model SST-2
  per-sample trajectories with checksums.
- `results/prompt_sensitivity/`: two additional full TruthfulQA prompt-template
  reruns; the original Q/A template is in `results/corrected_fep/`.
- `results/analysis/`: paper tables, Wilson intervals, top-k sensitivity, and
  paired exact McNemar statistics.
- `paper-blackboxnlp/`: anonymous source and compiled six-page PDF.

The paper's main finding is methodological: vocabulary readout is sensitive to
censoring, trajectory stability, prompt wording, and answer-space format. It is
not evidence of where knowledge is stored or first used causally.

## Reproduction

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q tests/unit/test_fep.py tests/unit/test_fep_analysis.py
PYTHONPATH=src python experiments/analyze_corrected_fep.py
```

GPU reruns use `experiments/run_corrected_fep.py`; see `--help` for model,
dataset, top-k, and prompt-template options. Canonical experiments used Python
3.12.13, PyTorch 2.10.0, Transformers 4.48.3, CUDA 12.8, bfloat16, and one NVIDIA
H20. All random sampling uses seed 42.

## License

This project is licensed under the MIT License. See `LICENSE`.
