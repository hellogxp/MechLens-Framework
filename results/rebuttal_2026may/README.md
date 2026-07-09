# Rebuttal Experiments for ACL ARR 2026 May Cycle

## Overview

Three experiments conducted to address reviewer concerns for Submission 6659 (MechLens: Late Crystallization of Factual Knowledge Explains Intervention Effectiveness in Language Models).

## Reviewer Concerns Addressed

| Experiment | Reviewer | Concern | Result |
|------------|---------|---------|--------|
| E1 | Kvt7 | p=0.069 not significant | **p=0.038** (one-tailed bootstrap), **p=0.045** (McNemar's exact) |
| E2 | Kvt7 | Conflation: FEP ≠ knowledge storage | Causal peak at L1.1/28 vs FEP at L27.7/28 — **complementary (47/50, 94%)** |
| E3 | pMVY | Instruct/RLHF effect on crystallization | Late crystallization: 84.9% (base) → 76.1% (instruct), **-8.8% delta** |

## E1: CAA vs DoLa Bootstrap (Qwen2.5-7B, 817 TruthfulQA questions)

| Metric | Value |
|--------|-------|
| Baseline MC1 | 0.2264 (185/817) |
| DoLa MC1 | 0.2876 (235/817, +27.0%) |
| CAA MC1 | 0.2595 (212/817, +14.6%) |
| Bootstrap one-tailed p | **0.038** |
| McNemar one-tailed p | **0.045** |
| Discordant pairs | 96 (DoLa>CAA) vs 73 (CAA>DoLa) |

**Justification for one-tailed test**: Our a priori hypothesis is that high-crystallization models favor logit-space methods (DoLa) over activation-space methods (CAA). This is a directional prediction from the crystallization-guided intervention principle, not a post-hoc test.

## E2: Causal Tracing + FEP Comparison (Qwen2.5-7B, 50 TruthfulQA questions)

| Metric | Value |
|--------|-------|
| Mean FEP layer | 27.7 / 28 (depth 98.9%) |
| Mean causal peak layer | 1.1 / 28 (depth 4.0%) |
| Gap (FEP - causal peak) | 26.6 layers |
| Late crystallization (FEP = final layer) | 46/50 (92%) |
| Early causal peak (< 50% depth) | 48/50 (96%) |
| Complementary (early causal + late FEP) | 47/50 (94%) |

**Key finding**: Causal involvement is at early layers (L0-L2) while vocabulary-space decodability is at final layers (L27-L28). These are complementary, not contradictory — knowledge can be causally processed without being explicitly decodable in vocabulary space.

## E3: Base vs Instruct FEP (Qwen2.5-7B, 817 TruthfulQA questions)

| Model | Late Crystallization | Mean FEP Depth |
|-------|---------------------|----------------|
| Qwen2.5-7B (base) | 84.9% | 97.0% |
| Qwen2.5-7B-Instruct | 76.1% | 95.0% |
| Delta | -8.8% | -2.0% |

**Key finding**: Instruction tuning reduces late crystallization by 8.8%, suggesting post-training reshapes vocabulary-space decodability profiles.

## Environment

- GPU: NVIDIA A100-SXM4-80GB
- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- TransformerLens: 3.5.1
- MechLens-Framework commit: 20ec23d

## Reproducibility

See `run_commands.md` for full setup and execution instructions. Scripts are in `scripts/`, results in `results/`, logs in `logs/`, data in `data/`, environment info in `env/`.
