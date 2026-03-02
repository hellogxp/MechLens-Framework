"""Contrastive Activation Addition (CAA) for truthfulness steering.

Extracts direction vectors from contrastive activation analysis
(avg_correct - avg_hallucination), normalizes them, and adds
scaled direction vectors to the residual stream during inference.

Unlike ITI which learns directions from separate contrastive prompt pairs
via dedicated collection, CAA directly reuses directions from the existing
ContrastiveResult produced by run_contrastive_analysis(). This makes CAA
a lightweight, zero-additional-training steering method.

Integration:
  - Direction source: ContrastiveResult.avg_correct / avg_hallucination
  - Steering mechanism: reuses create_iti_steering_hook() from iti.py
  - Evaluation: compatible with MC1/MC2 via fwd_hooks parameter
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformer_lens import HookedTransformer

from mechlens.analysis.contrastive import ContrastiveResult
from mechlens.intervention.iti import create_iti_steering_hook

logger = logging.getLogger(__name__)


@dataclass
class CAADirections:
    """CAA direction vectors extracted from contrastive analysis."""

    directions: dict[int, torch.Tensor]  # layer -> unit direction [d_model]
    magnitudes: dict[int, float]  # layer -> pre-normalization magnitude
    layers: list[int] = field(default_factory=list)


def learn_caa_directions(
    contrastive_result: ContrastiveResult,
    layers: list[int] | None = None,
) -> CAADirections:
    """Extract CAA direction vectors from a ContrastiveResult.

    For each layer, computes:
        direction = normalize(avg_correct - avg_hallucination)

    This is the mean activation difference, normalized to a unit vector,
    pointing from the hallucinated subspace toward the truthful subspace.

    Args:
        contrastive_result: Result from run_contrastive_analysis()
        layers: Specific layers to extract (None = all available)

    Returns:
        CAADirections with per-layer unit direction vectors
    """
    directions = {}
    magnitudes = {}

    available_hooks = set(contrastive_result.avg_correct.keys())

    if layers is None:
        # Extract layer indices from hook names
        layers = sorted(
            int(h.split(".")[1])
            for h in available_hooks
            if h.startswith("blocks.")
        )

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        if hook_name not in available_hooks:
            logger.warning(f"Layer {layer}: no contrastive data, skipping")
            continue

        avg_correct = contrastive_result.avg_correct[hook_name]  # [1, d_model]
        avg_hallucination = contrastive_result.avg_hallucination[hook_name]

        diff = (avg_correct - avg_hallucination).squeeze(0)  # [d_model]
        magnitude = diff.norm(p=2).item()

        if magnitude > 1e-8:
            directions[layer] = (diff / diff.norm(p=2)).cpu()
        else:
            directions[layer] = torch.zeros_like(diff).cpu()
            logger.warning(f"Layer {layer}: near-zero CAA direction (mag={magnitude:.6f})")

        magnitudes[layer] = magnitude

    # Log top layers by magnitude
    sorted_layers = sorted(layers, key=lambda l: magnitudes.get(l, 0), reverse=True)
    top_5 = sorted_layers[:5]
    logger.info(
        "CAA top layers by magnitude: "
        + ", ".join(f"L{l}={magnitudes.get(l, 0):.4f}" for l in top_5)
    )

    return CAADirections(
        directions=directions,
        magnitudes=magnitudes,
        layers=layers,
    )


def select_top_layers(
    caa_directions: CAADirections,
    top_k: int = 5,
) -> list[int]:
    """Select the top-K layers by direction magnitude for steering.

    Args:
        caa_directions: Learned CAA directions
        top_k: Number of layers to select

    Returns:
        List of layer indices sorted by magnitude (descending)
    """
    sorted_layers = sorted(
        caa_directions.layers,
        key=lambda l: caa_directions.magnitudes.get(l, 0),
        reverse=True,
    )
    return sorted_layers[:top_k]


def build_caa_hooks(
    caa_directions: CAADirections,
    coefficient: float = 1.0,
    layers: list[int] | None = None,
) -> list[tuple[str, Any]]:
    """Build TransformerLens forward hooks for CAA steering.

    Creates hooks that add coefficient * direction to the residual stream
    at each specified layer. Compatible with model.run_with_hooks() and
    the fwd_hooks parameter of evaluate_truthfulqa_mc1/mc2.

    Args:
        caa_directions: CAA directions from learn_caa_directions()
        coefficient: Steering strength (higher = more intervention)
        layers: Which layers to steer (None = all available)

    Returns:
        List of (hook_name, hook_fn) tuples for TransformerLens
    """
    if layers is None:
        layers = caa_directions.layers

    hooks = []
    for layer in layers:
        if layer not in caa_directions.directions:
            continue
        direction = caa_directions.directions[layer]
        hook_point = f"blocks.{layer}.hook_resid_post"
        hook_fn = create_iti_steering_hook(direction, coefficient)
        hooks.append((hook_point, hook_fn))

    logger.info(f"Built {len(hooks)} CAA steering hooks (coeff={coefficient})")
    return hooks


def generate_with_caa(
    model: HookedTransformer,
    input_text: str,
    caa_directions: CAADirections,
    coefficient: float = 1.0,
    layers: list[int] | None = None,
    max_new_tokens: int = 100,
) -> tuple[str, str]:
    """Generate text with CAA steering applied.

    Args:
        model: HookedTransformer model
        input_text: Input prompt
        caa_directions: CAA directions from learn_caa_directions()
        coefficient: Steering strength
        layers: Which layers to steer (None = all)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (original_output, steered_output)
    """
    tokens = model.to_tokens(input_text)

    # Generate original (greedy)
    with torch.no_grad():
        orig_ids = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    original_output = model.to_string(orig_ids[0, tokens.shape[1]:]).strip()

    # Build CAA hooks
    hooks = build_caa_hooks(caa_directions, coefficient, layers)

    if not hooks:
        logger.warning("No valid CAA hooks, returning original output")
        return original_output, original_output

    # Generate with CAA steering (token-by-token)
    generated = tokens.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(generated, fwd_hooks=hooks)
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

            if model.tokenizer.eos_token_id is not None:
                if next_token.item() == model.tokenizer.eos_token_id:
                    break

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    steered_output = model.to_string(generated[0, tokens.shape[1]:]).strip()
    return original_output, steered_output


def grid_search_caa(
    model: HookedTransformer,
    dataset: list[dict],
    caa_directions: CAADirections,
    coefficients: list[float] | None = None,
    top_k_values: list[int] | None = None,
    max_samples: int = 50,
) -> list[dict]:
    """Grid search over CAA hyperparameters using MC1 evaluation.

    Args:
        model: HookedTransformer model
        dataset: TruthfulQA samples
        caa_directions: CAA directions
        coefficients: Steering coefficients to try
        top_k_values: Number of top layers to steer
        max_samples: Samples per configuration

    Returns:
        List of result dicts sorted by mc1_score (descending)
    """
    from mechlens.benchmark.truthfulqa import evaluate_truthfulqa_mc1

    if coefficients is None:
        coefficients = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    if top_k_values is None:
        top_k_values = [3, 5, 10]

    results = []
    total = len(coefficients) * len(top_k_values)
    idx = 0

    for top_k in top_k_values:
        top_layers = select_top_layers(caa_directions, top_k)

        for coeff in coefficients:
            idx += 1
            logger.info(f"Grid search [{idx}/{total}]: coeff={coeff}, top_k={top_k}")

            hooks = build_caa_hooks(caa_directions, coeff, top_layers)
            mc1_result = evaluate_truthfulqa_mc1(
                model, dataset, fwd_hooks=hooks, max_samples=max_samples
            )

            results.append({
                "coefficient": coeff,
                "top_k": top_k,
                "layers": top_layers,
                "mc1_score": mc1_result["mc1_score"],
                "n_samples": mc1_result["n_samples"],
            })

            logger.info(
                f"  -> MC1={mc1_result['mc1_score']:.4f} "
                f"(coeff={coeff}, top_k={top_k})"
            )

    results.sort(key=lambda r: r["mc1_score"], reverse=True)
    return results
