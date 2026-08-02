# MechLens: Layerwise Factual Decodability Profiles and Their Relationship to Intervention Effectiveness

MechLens is a mechanistic-interpretability toolkit for measuring when factual
answers become decodable across transformer layers and for comparing those
profiles with factuality interventions.

## Scope and findings

The paper introduces the **Factual Emergence Point (FEP)**: the first layer at
which the correct answer token enters the top-*k* vocabulary projection. FEP is
a vocabulary-space decodability measurement; it is not, by itself, a claim
about where knowledge is stored or which layer is causally responsible for a
model's answer.

Across the tested base-model families, many factual answers become decodable
late in the network, with substantial variation between models and examples.
The accompanying experiments report:

- exploratory cross-model associations between FEP profiles and the relative
  effectiveness of CAA and DoLa, without claiming a universal intervention
  selection rule;
- causal-tracing results analyzed separately from vocabulary-space FEP;
- descriptive category-level computability--memorization patterns; and
- LayerNorm and attention-head analyses that constrain possible explanations
  but do not uniquely identify a mechanism.

## Supported models

| Model | Layers | Heads | d_model | Architecture |
|-------|--------|-------|---------|-------------|
| Qwen2.5-0.5B | 24 | 14 | 896 | GQA + SwiGLU |
| Qwen2.5-7B | 28 | 28 | 3584 | GQA + SwiGLU |
| Qwen2.5-14B | 48 | 40 | 5120 | GQA + SwiGLU |
| Llama-3.1-8B | 32 | 32 | 4096 | GQA + SwiGLU |
| Llama-2-7B | 32 | 32 | 4096 | Standard + SwiGLU |
| Mistral-7B | 32 | 32 | 4096 | Sliding Window + SwiGLU |
| Pythia-1.4B | 24 | 16 | 2048 | Standard MLP |

## Installation

Download the anonymous artifact linked in the paper, unpack it, and run:

```bash
cd MechLens-Framework
pip install -e ".[dev]"
```

To include all optional dependencies:

```bash
pip install -e ".[all]"
```

Requirements:

- Python 3.10 or later
- PyTorch 2.0 or later
- CUDA 12.1+ recommended for GPU acceleration
- An NVIDIA A100-40GB or comparable accelerator for 7B+ models

## Quick start

### Interactive UI

```bash
mechlens
# or
python -m mechlens.app
```

This starts a local interface for FEP measurement and visualization, activation
analysis, causal tracing, intervention experiments, and circuit analysis.

### Programmatic usage

```python
from mechlens.models.model_loader import load_model
from mechlens.analysis.activation import compute_fep

model = load_model("Qwen/Qwen2.5-7B")
result = compute_fep(
    model,
    "The capital of France is",
    answer="Paris",
    k=10,
)
print(f"FEP layer: {result.fep_layer}, depth: {result.fep_depth:.1%}")
```

## Reproducing the paper

Experiment drivers are in `experiments/`, and precomputed outputs are in
`results/`. Representative commands include:

```bash
python experiments/run_fep_detection.py
python experiments/run_cross_architecture_interventions.py
python experiments/run_layernorm_ablation.py
python experiments/run_head_attribution.py
python experiments/run_mmlu_fep_validation.py
python experiments/run_tuned_lens_comparison.py
```

The anonymous manuscript source and compiled PDF are in
`paper-blackboxnlp/`. See that directory's README for build instructions.

## Project structure

```text
MechLens-Framework/
├── src/mechlens/          # Analysis, interventions, models, and visualization
├── experiments/           # Paper experiment drivers
├── results/               # Precomputed experiment outputs
├── data/                  # Evaluation data
├── tests/                 # Unit and integration tests
├── paper-blackboxnlp/     # Anonymous manuscript source and compiled PDF
└── pyproject.toml         # Package configuration
```

## Testing

```bash
pytest tests/unit/ -v
```

## Citation

Citation metadata will be added after publication.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
