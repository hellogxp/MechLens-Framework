# Canonical run manifest

This manifest documents the commands that generated the August 2, 2026
BlackboxNLP artifacts. Run them from the repository root. All runs used
`--top-k 10 --dtype bfloat16 --sample-strategy first --seed 42` on one NVIDIA
H20 with Python 3.12.13, PyTorch 2.10.0, Transformers 4.48.3, and CUDA 12.8.

## Historical provenance boundary

The original jobs loaded local snapshots under `/mnt/workspace/models/` and
did not capture immutable Hugging Face revisions or a Git commit. The exact
per-sample outputs, prompts, aligned token IDs, ranks, probabilities, and
checksums are retained, but a bitwise GPU rerun cannot be promised from the
historical metadata alone. The public model IDs corresponding to those local
snapshots were:

| Alias | Public model ID |
|---|---|
| `qwen7` | `Qwen/Qwen2.5-7B` |
| `qwen14` | `Qwen/Qwen2.5-14B` |
| `llama` | `meta-llama/Llama-3.1-8B` |
| `mistral` | `mistralai/Mistral-7B-v0.1` |
| `pythia` | `EleutherAI/pythia-6.9b` |
| `gemma` | `google/gemma-7b` |

The revised runner accepts `--model-revision` and `--dataset-revision` and
records requested and resolved revisions. New archival runs should always pin
both when a remote dataset is used.

## TruthfulQA

The input is the tracked `data/truthfulqa.json`. Replace `$MODEL` by each of
`qwen7 qwen14 llama mistral pythia gemma`:

```bash
python blackboxnlp-2026/experiments/run_corrected_fep.py \
  --model "$MODEL" --dataset truthfulqa --max-samples 817
```

The two prompt controls use Qwen2.5-7B:

```bash
python blackboxnlp-2026/experiments/run_corrected_fep.py \
  --model qwen7 --dataset truthfulqa --prompt-template question_answer \
  --max-samples 817
python blackboxnlp-2026/experiments/run_corrected_fep.py \
  --model qwen7 --dataset truthfulqa --prompt-template instruction \
  --max-samples 817
```

## MMLU

The run uses the first 50 test examples from each of the 24 subjects listed in
the runner, for 1,200 examples total:

```bash
python blackboxnlp-2026/experiments/run_corrected_fep.py \
  --model qwen7 --dataset mmlu --max-samples 1200 \
  --dataset-revision "$CAIS_MMLU_REVISION"
```

The historical run did not record `$CAIS_MMLU_REVISION`. Its exact logical
inputs remain embedded in the canonical artifact and are hashed in
`../analysis/dataset_sample_manifest.csv`.

## SST-2

The experiment uses all 872 validation rows with `sentence` and `label`
columns. Replace `$MODEL` by each of `qwen7 mistral llama pythia gemma`:

```bash
python blackboxnlp-2026/experiments/run_corrected_fep.py \
  --model "$MODEL" --dataset sst2 --max-samples 872 \
  --sst2-data-file "$SST2_VALIDATION_TSV"
```

The original TSV checksum was not captured. The exact sentence, label, prompt,
and sample order are retained in every canonical SST-2 artifact and are hashed
in `../analysis/dataset_sample_manifest.csv`.

## Analysis and verification

```bash
python blackboxnlp-2026/experiments/analyze_corrected_fep.py
PYTHONPATH=src pytest -q tests/unit/test_fep.py tests/unit/test_fep_analysis.py
ruff check blackboxnlp-2026/experiments src/mechlens/fep.py \
  src/mechlens/fep_analysis.py
cd blackboxnlp-2026/results/corrected_fep
shasum -a 256 -c checksums.sha256
```
