# MechLens

A mechanistic interpretability framework for large language models (LLMs).

## Overview

MechLens provides tools for analyzing and understanding the internal mechanisms of transformer-based language models. It offers:

- **Analysis Tools**: Logit lens, causal tracing, activation analysis, attention pattern analysis
- **Intervention Strategies**: DoLa, CAA, ITI, activation scaling, ablation, injection
- **Visualization**: Interactive visualizations for activations, attention, circuits, and interventions
- **Benchmarks**: TruthfulQA and Chinese Hallucination Benchmark support

## Installation

```bash
# Clone the repository
git clone https://github.com/hellogxp/MechLens-Framework.git
cd MechLens-Framework

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.37
- transformer-lens >= 2.0

## Quick Start

```python
from mechlens.models import ModelLoader
from mechlens.analysis import LogitLensAnalyzer

# Load a model
loader = ModelLoader()
model = loader.load("pythia-70m")

# Run logit lens analysis
analyzer = LogitLensAnalyzer(model)
results = analyzer.analyze("The capital of France is")

# Visualize results
from mechlens.visualization import plot_logit_lens
plot_logit_lens(results)
```

## Features

### Analysis Methods

- **Logit Lens**: Project intermediate activations to vocabulary space
- **Causal Tracing**: Identify important components through activation patching
- **Contrastive Analysis**: Compare activations between correct and incorrect predictions
- **Circuit Analysis**: Discover and analyze computational circuits

### Intervention Strategies

- **DoLa (Decoding by Contrasting Layers)**: Contrast early and late layer predictions
- **CAA (Contrastive Activation Addition)**: Add steering vectors to modify behavior
- **ITI (Inference-Time Intervention)**: Apply learned intervention directions
- **Activation Scaling/Ablation**: Modify specific component activations

### Supported Models

MechLens supports models compatible with TransformerLens:
- Pythia family (70M to 12B)
- GPT-2 family
- Llama family
- Qwen family
- And more...

## Project Structure

```
src/mechlens/
├── analysis/          # Analysis methods
├── intervention/      # Intervention strategies
├── visualization/     # Visualization tools
├── models/           # Model loading and hooks
├── benchmark/        # Benchmark implementations
└── experiments/      # Experiment utilities
```

## Citation

If you use MechLens in your research, please cite:

```bibtex
@inproceedings{mechlens2026,
  title={MechLens: A Mechanistic Interpretability Framework for LLMs},
  author={Anonymous},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
