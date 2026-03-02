"""MechLens intervention modules.

Provides ablation, scaling, and injection interventions for activation manipulation.
PRIMARY intervention paradigm as defined in the API contracts.
"""

import logging
from typing import Any, Literal

import torch
from transformer_lens import HookedTransformer

from mechlens.intervention import ablation, caa, dola, injection, iti, scaling, strategy
from mechlens.intervention.ablation import ablate
from mechlens.intervention.caa import build_caa_hooks, generate_with_caa, learn_caa_directions
from mechlens.intervention.dola import generate_with_dola, create_dola_score_fn
from mechlens.intervention.injection import inject
from mechlens.intervention.iti import generate_with_iti, learn_iti_directions
from mechlens.intervention.scaling import scale
from mechlens.types import InterventionResult, InterventionTarget

logger = logging.getLogger(__name__)

__all__ = [
    # Submodules
    "ablation",
    "scaling",
    "injection",
    "iti",
    "dola",
    "caa",
    "strategy",
    # Main functions
    "ablate",
    "scale",
    "inject",
    "learn_iti_directions",
    "generate_with_iti",
    "generate_with_dola",
    "create_dola_score_fn",
    "learn_caa_directions",
    "build_caa_hooks",
    "generate_with_caa",
    "batch_intervene",
]


def batch_intervene(
    model: HookedTransformer,
    samples: list[dict[str, str]],
    intervention_type: Literal["ablation", "scaling", "injection"],
    targets: list[InterventionTarget],
    **kwargs: Any,
) -> list[InterventionResult]:
    """Run the same intervention across multiple samples for batch evaluation.

    Args:
        model: HookedTransformer model
        samples: List of {"input_text": str, "expected": str}
        intervention_type: "ablation" | "scaling" | "injection"
        targets: Shared intervention targets
        **kwargs: Type-specific params:
            - ablation: (none)
            - scaling: factor (float)
            - injection: source_activations (dict[str, Tensor])

    Returns:
        List of InterventionResult, one per sample
    """
    results = []

    for i, sample in enumerate(samples):
        input_text = sample["input_text"]

        logger.debug(f"Processing sample {i+1}/{len(samples)}")

        if intervention_type == "ablation":
            result = ablate(model, input_text, targets, **kwargs)
        elif intervention_type == "scaling":
            factor = kwargs.get("factor", 0.5)
            result = scale(model, input_text, targets, factor, **kwargs)
        elif intervention_type == "injection":
            source_activations = kwargs.get("source_activations", {})
            result = inject(model, input_text, targets, source_activations, **kwargs)
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        # Add expected output to metrics for evaluation
        result.metrics["expected"] = sample.get("expected", "")
        result.metrics["sample_idx"] = i

        results.append(result)

    logger.info(f"Batch intervention complete: {len(results)} samples")
    return results


def run_from_strategy(
    model: HookedTransformer,
    input_text: str,
    strategy_id_or_name: str,
    **kwargs: Any,
) -> InterventionResult:
    """Run intervention using a saved strategy.

    Args:
        model: HookedTransformer model
        input_text: Input text
        strategy_id_or_name: Strategy ID or name
        **kwargs: Additional parameters to override strategy params

    Returns:
        InterventionResult
    """
    strat = strategy.load(strategy_id_or_name)
    targets = strategy.deserialize_targets(strat["targets"])
    intervention_type = strat["intervention_type"]

    # Merge strategy params with kwargs
    params = {**strat.get("params", {}), **kwargs}

    if intervention_type == "ablation":
        return ablate(model, input_text, targets)
    elif intervention_type == "scaling":
        factor = params.get("factor", 0.5)
        return scale(model, input_text, targets, factor)
    elif intervention_type == "injection":
        source_activations = params.get("source_activations", {})
        return inject(model, input_text, targets, source_activations)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


def evaluate_intervention(
    results: list[InterventionResult],
    metric: str = "kl_divergence",
) -> dict[str, Any]:
    """Evaluate batch intervention results.

    Args:
        results: List of InterventionResult from batch_intervene
        metric: Primary metric for evaluation

    Returns:
        Evaluation summary dict
    """
    if not results:
        return {"error": "No results to evaluate"}

    # Aggregate metrics
    metric_values = [r.metrics.get(metric, 0) for r in results]

    # Compute statistics
    avg = sum(metric_values) / len(metric_values)
    min_val = min(metric_values)
    max_val = max(metric_values)
    std = (sum((v - avg) ** 2 for v in metric_values) / len(metric_values)) ** 0.5

    # Count output changes
    output_changes = sum(
        1 for r in results
        if r.original_output != r.intervened_output
    )

    return {
        "n_samples": len(results),
        "metric": metric,
        "mean": avg,
        "std": std,
        "min": min_val,
        "max": max_val,
        "output_change_rate": output_changes / len(results),
    }
