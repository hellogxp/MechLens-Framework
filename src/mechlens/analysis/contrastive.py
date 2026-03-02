"""Contrastive Activation Analysis.

Compares activation patterns between correct and hallucinated model outputs
to identify layers and neurons most associated with hallucination.

Uses last-token representation to handle variable sequence lengths across
different prompts.
"""
import logging
from dataclasses import dataclass, field

import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveResult:
    """Result of contrastive activation analysis."""

    layer_importance: list[float]
    top_neurons: dict[int, list[dict]]
    avg_correct: dict[str, torch.Tensor] = field(default_factory=dict)
    avg_hallucination: dict[str, torch.Tensor] = field(default_factory=dict)


def run_contrastive_analysis(
    model: HookedTransformer,
    prompts_with_labels: list[tuple[str, str, bool]],
    max_new_tokens: int = 1,
    layers_to_analyze: list[int] | None = None,
) -> ContrastiveResult:
    """Run contrastive analysis comparing correct vs hallucinated outputs.

    For each labeled prompt, collects activations at every residual stream
    position, then computes per-layer importance as the L2 norm of the
    mean difference between correct and hallucinated activation vectors.

    Uses last-token representation to handle variable sequence lengths.

    Args:
        model: HookedTransformer model
        prompts_with_labels: List of (prompt_text, answer, is_correct) tuples
        max_new_tokens: Not used for activation collection, kept for API compat
        layers_to_analyze: Specific layers to analyze (None = all)

    Returns:
        ContrastiveResult with layer importance scores and top neurons
    """
    n_layers = model.cfg.n_layers
    if layers_to_analyze is None:
        layers_to_analyze = list(range(n_layers))

    correct_prompts = [(p, a) for p, a, c in prompts_with_labels if c]
    incorrect_prompts = [(p, a) for p, a, c in prompts_with_labels if not c]

    if not correct_prompts or not incorrect_prompts:
        raise ValueError(
            f"Need both correct and incorrect examples. "
            f"Got {len(correct_prompts)} correct, {len(incorrect_prompts)} incorrect."
        )

    logger.info(
        f"Contrastive analysis: {len(correct_prompts)} correct, "
        f"{len(incorrect_prompts)} incorrect examples"
    )

    # Collect activations
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers_to_analyze]

    correct_acts_list = _collect_activations(model, correct_prompts, hook_names)
    incorrect_acts_list = _collect_activations(model, incorrect_prompts, hook_names)

    # Compute average activations using last-token representation
    # This handles variable sequence lengths across different prompts
    avg_correct = {}
    avg_hallucination = {}

    for hook_name in hook_names:
        # Extract last-token representation from each example: shape [1, d_model]
        correct_last = torch.stack(
            [a[hook_name][:, -1, :] for a in correct_acts_list]
        )  # [n_correct, 1, d_model]
        incorrect_last = torch.stack(
            [a[hook_name][:, -1, :] for a in incorrect_acts_list]
        )  # [n_incorrect, 1, d_model]

        avg_correct[hook_name] = correct_last.mean(dim=0)  # [1, d_model]
        avg_hallucination[hook_name] = incorrect_last.mean(dim=0)

    # Compute layer importance as L2 norm of mean activation difference
    layer_importance = [0.0] * n_layers
    for layer in layers_to_analyze:
        hook_name = f"blocks.{layer}.hook_resid_post"
        diff = avg_correct[hook_name] - avg_hallucination[hook_name]
        layer_importance[layer] = diff.norm(p=2).item()

    # Normalize to [0, 1] range
    max_imp = max(layer_importance) if max(layer_importance) > 0 else 1.0
    layer_importance = [v / max_imp for v in layer_importance]

    # Find top neurons per layer (by absolute difference)
    top_neurons = {}
    for layer in layers_to_analyze:
        hook_name = f"blocks.{layer}.hook_resid_post"
        diff = (avg_correct[hook_name] - avg_hallucination[hook_name]).squeeze(0)  # [d_model]
        abs_diff = diff.abs()
        top_k = min(20, abs_diff.numel())
        top_vals, top_idxs = torch.topk(abs_diff, top_k)
        top_neurons[layer] = [
            {"neuron_idx": idx.item(), "diff": diff[idx].item(), "abs_diff": val.item()}
            for val, idx in zip(top_vals, top_idxs)
        ]

    top_layers = sorted(
        layers_to_analyze,
        key=lambda l: layer_importance[l],
        reverse=True,
    )[:5]
    logger.info(f"Top contrastive layers: {top_layers}")

    return ContrastiveResult(
        layer_importance=layer_importance,
        top_neurons=top_neurons,
        avg_correct=avg_correct,
        avg_hallucination=avg_hallucination,
    )


def _collect_activations(
    model: HookedTransformer,
    prompts: list[tuple[str, str]],
    hook_names: list[str],
) -> list[dict[str, torch.Tensor]]:
    """Collect activations for a list of prompts.

    Args:
        model: HookedTransformer model
        prompts: List of (prompt_text, answer) tuples
        hook_names: List of hook names to cache

    Returns:
        List of dicts mapping hook_name -> activation tensor
    """
    all_activations = []
    for prompt_text, _ in prompts:
        tokens = model.to_tokens(prompt_text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        acts = {name: cache[name].detach().cpu() for name in hook_names}
        all_activations.append(acts)
    return all_activations
