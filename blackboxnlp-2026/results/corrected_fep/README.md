# Corrected FEP audit for BlackboxNLP 2026

This directory contains the canonical reruns performed on August 2, 2026. The
results supersede the legacy FEP, MMLU, and SST-2 numbers currently quoted in
the manuscript. Do not update the paper by mixing the two result families.

## Measurement contract

- The target is the first continuation token obtained by tokenizing the full
  `prompt + continuation` string and verifying exact prompt-prefix alignment.
- A target that never enters the full-vocabulary top-10 is stored as `null`,
  not as a final-layer event.
- First FEP and persistent FEP are separate. Persistent FEP is the first layer
  after which the target stays in the top-10 through the output.
- The final trajectory point uses the model's native logits. Reprojected final
  logits must preserve top-10 membership and have maximum absolute error at
  most 0.25; exact-rank equality is retained as an audit field.
- MMLU and SST-2 task accuracy is computed over the explicit candidate labels
  (A/B/C/D or negative/positive), separately from full-vocabulary rank.
- Layer numbers in this directory are one-based.

## Canonical results

### TruthfulQA

| Model | First observed | Never observed | Final top-10 | Dropout after entry | Mean first FEP | Mean persistent FEP |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 68.91% | 31.09% | 64.75% | 18.85% | 23.06/28 | 24.53/28 |
| Qwen2.5-14B | 71.24% | 28.76% | 67.69% | 16.77% | 41.87/48 | 43.21/48 |
| Llama-3.1-8B | 73.81% | 26.19% | 71.73% | 15.18% | 26.65/32 | 27.67/32 |
| Mistral-7B-v0.1 | 73.19% | 26.81% | 67.93% | 18.48% | 25.22/32 | 26.71/32 |
| Pythia-6.9B | 78.09% | 21.91% | 71.11% | 28.40% | 14.53/32 | 19.67/32 |
| Gemma-7B | 76.50% | 23.50% | 71.73% | 47.98% | 23.77/28 | 27.99/28 |

The cross-model result is not a single ordering of "late crystallization."
Pythia becomes visible early but often leaves the top-10. Gemma becomes visible
later, and its raw instability is dominated by one penultimate-layer projection:
the any-gap rate falls from 47.98% to 23.38% when that readout is treated as
missing. This is a raw-lens sensitivity diagnosis, not evidence that the model
destroys and reconstructs information. First visibility, persistence, and final
visibility are distinct properties.

### TruthfulQA target audit

The experiment follows the manuscript's first-continuation-token contract, but
the reference answers are almost always longer: 97.92%--99.14% are multi-token
under the six tokenizers. Across models, about 63.8% of lexical first-token
targets belong to the ten most frequent types. See
`../analysis/truthfulqa_target_audit.csv` and
`../analysis/truthfulqa_first_token_frequencies.csv`. These results audit
answer-prefix vocabulary readout, not whole-answer semantics.

### MMLU

Qwen2.5-7B on 1,200 balanced examples from 24 subjects reaches 72.0% candidate
accuracy. Every correct A/B/C/D label enters and ends in the full-vocabulary
top-10. Mean first FEP is 18.08/28 and mean persistent FEP is 20.47/28. The old
claim that 98.2% of MMLU answers crystallize only at the final layer was caused
by tracking the answer text rather than the correct option label.

| Group | n | Accuracy | Mean first FEP | Mean persistent FEP | Dropout |
|---|---:|---:|---:|---:|---:|
| STEM | 450 | 63.11% | 17.80 | 20.45 | 52.44% |
| Humanities | 250 | 75.20% | 18.96 | 20.72 | 48.40% |
| Social Sciences | 250 | 80.00% | 17.40 | 20.49 | 63.60% |
| Other | 250 | 76.80% | 18.37 | 20.24 | 47.60% |

### SST-2

| Model | Candidate accuracy | Ever top-10 | Never top-10 | Final top-10 | Mean first FEP |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 91.63% | 99.54% | 0.46% | 94.27% | 24.50/28 |
| Mistral-7B-v0.1 | 86.93% | 48.85% | 51.15% | 0.00% | 21.95/32 |
| Llama-3.1-8B | 73.17% | 44.50% | 55.50% | 35.09% | 26.45/32 |
| Pythia-6.9B | 80.85% | 0.00% | 100.00% | 0.00% | n/a |
| Gemma-7B | 69.84% | 2.06% | 97.94% | 2.06% | 28.00/28 |

Candidate accuracy can be high while the label never becomes a likely free-form
continuation. This shows that FEP measures full-vocabulary readout and prompt
calibration, not the mere presence of task-discriminative information.

## Manuscript consequences

1. Remove the legacy 26.8%--93.4% strict-final-layer range, the 85.9% Qwen
   headline, and the 98.2% MMLU claim.
2. Replace the final-layer sentinel definition with explicit right censoring.
3. Reframe the contribution around trajectory types: first decodability,
   persistent decodability, dropout, and never-observed targets.
4. Treat MMLU as evidence that answer format and candidate calibration can
   dominate FEP, not as replication of late factual readout.
5. Treat SST-2 as a readout-format diagnostic. Do not infer that a model cannot
   perform sentiment classification merely because the label is outside the
   free-vocabulary top-10.
6. Re-run any tuned-lens, category-spectrum, causal-boundary, LayerNorm, or
   FEP-guided intervention analysis that consumed the legacy FEP values before
   retaining its numerical claims.

## Reproduction

The canonical runner is `blackboxnlp-2026/experiments/run_corrected_fep.py`; pure trajectory and
aggregation logic is in `src/mechlens/fep.py` and covered by unit tests. Raw
JSON files are committed as deterministic `.json.gz` archives. They contain
per-sample ranks, probabilities, trajectories, alignment metadata, candidate
predictions where applicable, and failure records.

`RUN_MANIFEST.md` records the canonical commands, environment, public model
IDs, and the historical provenance boundary. In particular, the August 2 GPU
runs recorded local model paths but not immutable hub revisions or a Git
commit. The revised runner now accepts and records model and dataset revisions;
the missing historical values are disclosed rather than reconstructed.

The analysis additionally releases:

- `pairwise_first_entry_depth.csv`: paired exact sign tests on jointly observed
  TruthfulQA items, with Holm correction;
- `pairwise_final_visibility.csv`: exact McNemar tests with Holm and BH
  corrections;
- `truthfulqa_target_audit.csv` and `truthfulqa_first_token_frequencies.csv`;
- `dataset_sample_manifest.csv`: SHA-256 digests over logical sample inputs;
- `single_layer_sensitivity.csv` and `layer_influence.csv`.

When interpreting `layer_influence.csv`, masking the final layer changes the
last included observation to layer $L-1$; those rows are diagnostic only. The
paper's selected layers are all non-final.

Verify the raw artifacts from this directory with:

```bash
gunzip -k *.json.gz
shasum -a 256 -c checksums.sha256
```
