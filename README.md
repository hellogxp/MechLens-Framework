# MechLens: Late Crystallization of Factual Knowledge in Language Models

A mechanistic interpretability framework for analyzing factual knowledge emergence in large language models.

## Overview

MechLens discovers **Late Crystallization**: factual knowledge in LLMs does not gradually emerge across layers but "crystallizes" abruptly at the final layers. In Qwen2.5-7B, 85.9% of correct answers *never* enter top-10 predictions at any intermediate layer.

Key findings:
- **Factual Emergence Point (FEP)**: A formal metric identifying the critical layer where correct answers first appear in logit predictions
- **Architecture-dependent intervention patterns**: CAA outperforms DoLa on moderate-crystallization models; DoLa dominates on high-crystallization models
- **Computability-Memorization Spectrum**: Computable knowledge crystallizes earlier than memorized facts
- **LayerNorm as crystallization amplifier**: LN scaling (x1.2) yields +11.8% MC1 with zero inference overhead

## Supported Models

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

```bash
# Clone the repository
git clone https://github.com/anonymous/MechLens-Framework.git
cd MechLens-Framework

# Install in development mode
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA 12.1+ (recommended for GPU acceleration)
- NVIDIA A100-40GB or equivalent (for 7B+ models)

## Quick Start

### Interactive Gradio UI

```bash
mechlens
# or
python -m mechlens.app
```

This launches an interactive web interface at `http://localhost:7860` for:
- FEP detection and visualization
- Activation analysis (logit lens / tuned lens)
- Causal tracing
- Intervention experiments (CAA, DoLa, ITI, ablation, scaling)
- Circuit analysis

### Programmatic Usage

```python
from mechlens.models.model_loader import load_model
from mechlens.analysis.activation import compute_fep

# Load model
model = load_model("Qwen/Qwen2.5-7B")

# Detect Factual Emergence Point
fep_result = compute_fep(model, "The capital of France is", answer="Paris", k=10)
print(f"FEP Layer: {fep_result.fep_layer}, Depth: {fep_result.fep_depth:.1%}")
```

## Reproducing Paper Results

All experiment scripts are in the `experiments/` directory:

```bash
# FEP detection across models
python experiments/run_fep_detection.py

# Cross-architecture intervention comparison
python experiments/run_cross_architecture_interventions.py

# CrystalBoost validation
python experiments/run_crystalboost.py

# 14B scale validation
python experiments/run_14b_scale_validation.py

# MMLU cross-benchmark validation
python experiments/run_mmlu_fep_validation.py

# Tuned lens comparison
python experiments/run_tuned_lens_comparison.py

# Full Round 2 experiments
bash experiments/run_round2_all.sh
```

Pre-computed results are available in the `results/` directory.

## Project Structure

```
MechLens-Framework/
├── src/mechlens/
│   ├── analysis/          # FEP detection, activation analysis, causal tracing, circuit analysis
│   ├── intervention/      # CAA, DoLa, ITI, ablation, scaling, CrystalBoost
│   ├── models/            # Model loading, hook management
│   ├── benchmark/         # TruthfulQA, Chinese hallucination bench
│   ├── editing/           # ROME, MEMIT knowledge editing
│   ├── experiments/       # Cross-model experiment orchestration
│   ├── visualization/     # Plotly-based interactive visualizations
│   ├── app.py             # Gradio web interface
│   ├── config.py          # Model registry and configuration
│   └── types.py           # Type definitions
├── experiments/           # Experiment scripts for paper reproduction
├── results/               # Pre-computed experiment results
├── data/                  # Evaluation datasets
├── tests/                 # Unit and integration tests
├── paper/                 # LaTeX source and figures
└── pyproject.toml         # Project configuration
```

## Testing

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src/mechlens --cov-report=term-missing

# Run specific test
pytest tests/unit/test_intervention.py -v
```

## Hardware Requirements

| Model | FP16 VRAM | INT8 VRAM |
|-------|-----------|-----------|
| Pythia-1.4B | ~4 GB | ~2 GB |
| Qwen2.5-0.5B | ~2 GB | ~1 GB |
| Qwen2.5-7B | ~16 GB | ~8 GB |
| Qwen2.5-14B | ~30 GB | ~16 GB |
| Llama-3.1-8B | ~18 GB | ~10 GB |

## Citation

```bibtex
@inproceedings{
  anonymous2026mechlens,
  title={MechLens: Late Crystallization of Factual Knowledge Explains Intervention Effectiveness in Language Models},
  author={Anonymous},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026},
  url={https://anonymous.4open.science/r/MechLens-COLM2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
