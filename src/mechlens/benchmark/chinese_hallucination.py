"""MechLens benchmark loader and evaluator.

Load and evaluate on ChineseHallucinationBench dataset.
Per contract section 11 - Qwen2.5 models only.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

from mechlens.config import get_model_metadata
from mechlens.types import HallucinationDomain, HallucinationSample, HallucinationType, UnsupportedModelError

logger = logging.getLogger(__name__)


def load_dataset(
    path: str | Path,
) -> list[HallucinationSample]:
    """Load ChineseHallucinationBench dataset.

    Args:
        path: Path to dataset JSON file

    Returns:
        List of HallucinationSample
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        sample = HallucinationSample(
            id=item["id"],
            question=item["question"],
            ground_truth=item["ground_truth"],
            hallucination_type=HallucinationType(item["hallucination_type"]),
            domain=HallucinationDomain(item["domain"]),
            should_refuse=item.get("should_refuse", False),
            reference_sources=item.get("reference_sources", []),
        )
        samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def evaluate(
    model: Any,
    dataset: list[HallucinationSample],
    intervention_fn: Callable[[Any, str], tuple[str, str]] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate model on ChineseHallucinationBench.

    Args:
        model: HookedTransformer model
        dataset: List of HallucinationSample
        intervention_fn: Optional intervention function (model, question) -> (original_output, intervened_output)
        model_name: Model name for validation (extracted from model if not provided)

    Returns:
        Dict with hallucination_rate, hallucination_rate_reduction, per_type_rates, per_domain_rates, per_sample_results
    """
    # Validate model support (Qwen only)
    if model_name is None:
        model_name = _get_model_name(model)

    _validate_model_support(model_name)

    # Run evaluation
    results = []
    hallucinations = 0
    hallucinations_after = 0

    per_type_counts = {t: {"total": 0, "hallucinated": 0, "hallucinated_after": 0}
                       for t in HallucinationType}
    per_domain_counts = {d: {"total": 0, "hallucinated": 0, "hallucinated_after": 0}
                         for d in HallucinationDomain}

    for sample in dataset:
        # Get model response
        original_output = _generate_response(model, sample.question)

        # Check for hallucination (simple heuristic: ground truth not in response)
        is_hallucinated = not _check_correct(original_output, sample.ground_truth)

        if is_hallucinated:
            hallucinations += 1

        # Apply intervention if provided
        intervened_output = None
        is_hallucinated_after = is_hallucinated

        if intervention_fn is not None:
            _, intervened_output = intervention_fn(model, sample.question)
            is_hallucinated_after = not _check_correct(intervened_output, sample.ground_truth)

            if is_hallucinated_after:
                hallucinations_after += 1

        # Update per-type and per-domain counts
        per_type_counts[sample.hallucination_type]["total"] += 1
        per_domain_counts[sample.domain]["total"] += 1

        if is_hallucinated:
            per_type_counts[sample.hallucination_type]["hallucinated"] += 1
            per_domain_counts[sample.domain]["hallucinated"] += 1

        if is_hallucinated_after:
            per_type_counts[sample.hallucination_type]["hallucinated_after"] += 1
            per_domain_counts[sample.domain]["hallucinated_after"] += 1

        results.append({
            "id": sample.id,
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "original_output": original_output,
            "intervened_output": intervened_output,
            "is_hallucinated": is_hallucinated,
            "is_hallucinated_after": is_hallucinated_after,
        })

    # Compute rates
    n_samples = len(dataset)
    hallucination_rate = hallucinations / n_samples if n_samples > 0 else 0
    hallucination_rate_after = hallucinations_after / n_samples if n_samples > 0 else 0

    if intervention_fn is not None:
        hallucination_rate_reduction = hallucination_rate - hallucination_rate_after
    else:
        hallucination_rate_reduction = 0

    # Compute per-type rates
    per_type_rates = {}
    for t, counts in per_type_counts.items():
        if counts["total"] > 0:
            per_type_rates[t.value] = {
                "rate": counts["hallucinated"] / counts["total"],
                "rate_after": counts["hallucinated_after"] / counts["total"],
                "reduction": (counts["hallucinated"] - counts["hallucinated_after"]) / counts["total"],
            }

    # Compute per-domain rates
    per_domain_rates = {}
    for d, counts in per_domain_counts.items():
        if counts["total"] > 0:
            per_domain_rates[d.value] = {
                "rate": counts["hallucinated"] / counts["total"],
                "rate_after": counts["hallucinated_after"] / counts["total"],
                "reduction": (counts["hallucinated"] - counts["hallucinated_after"]) / counts["total"],
            }

    logger.info(
        f"Evaluation complete: hallucination_rate={hallucination_rate:.3f}, "
        f"reduction={hallucination_rate_reduction:.3f}"
    )

    return {
        "hallucination_rate": hallucination_rate,
        "hallucination_rate_after": hallucination_rate_after,
        "hallucination_rate_reduction": hallucination_rate_reduction,
        "per_type_rates": per_type_rates,
        "per_domain_rates": per_domain_rates,
        "per_sample_results": results,
        "n_samples": n_samples,
        "model_name": model_name,
    }


def _validate_model_support(model_name: str) -> None:
    """Validate that model supports ChineseHallucinationBench evaluation."""
    try:
        metadata = get_model_metadata(model_name)
        if not metadata.supports_chinese_bench:
            raise UnsupportedModelError(
                f"ChineseHallucinationBench not supported for {model_name}. "
                "Only Qwen2.5 models are supported per R8."
            )
    except ValueError:
        # Check by name pattern
        if "qwen" not in model_name.lower():
            raise UnsupportedModelError(
                f"ChineseHallucinationBench not supported for {model_name}. "
                "Only Qwen2.5 models are supported per R8."
            )


def _get_model_name(model: Any) -> str:
    """Extract model name from model."""
    if hasattr(model, "cfg"):
        if hasattr(model.cfg, "model_name") and model.cfg.model_name:
            return model.cfg.model_name
        if hasattr(model.cfg, "tokenizer_name"):
            return str(model.cfg.tokenizer_name)
    return "unknown"


def _generate_response(model: Any, question: str) -> str:
    """Generate model response for a question."""
    import torch

    tokens = model.to_tokens(question)

    with torch.no_grad():
        output_ids = model.generate(
            tokens,
            max_new_tokens=100,
            do_sample=False,
        )

    # Decode only the generated part
    response = model.to_string(output_ids[0, tokens.shape[1]:])
    return response.strip()


def _check_correct(output: str, ground_truth: str) -> bool:
    """Check if output contains the correct answer.

    Uses flexible matching to handle variations in response format.
    """
    output_lower = output.lower()
    ground_truth_lower = ground_truth.lower()

    # Direct containment
    if ground_truth_lower in output_lower:
        return True

    # Check key parts (for longer ground truths)
    # Split by common separators and check if key parts match
    gt_parts = ground_truth_lower.replace("（", "(").replace("）", ")").split("(")
    main_answer = gt_parts[0].strip()

    if main_answer and main_answer in output_lower:
        return True

    # For numeric answers, be more flexible
    import re
    gt_numbers = re.findall(r"\d+\.?\d*", ground_truth)
    if gt_numbers:
        for num in gt_numbers:
            if num in output:
                return True

    return False


def load_counterfact(
    path: str | Path,
) -> list[dict[str, str]]:
    """Load CounterFact sample dataset.

    Args:
        path: Path to counterfact JSON file

    Returns:
        List of sample dicts with subject, target_old, target_new, prompt, ground_truth
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    logger.info(f"Loaded {len(samples)} CounterFact samples from {path}")
    return samples


def evaluate_counterfact(
    model: Any,
    samples: list[dict[str, str]],
    intervention_fn: Callable[[Any, str], tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate model on CounterFact samples.

    Unlike ChineseHallucinationBench, this is supported on all 4 model families.

    Args:
        model: HookedTransformer model
        samples: List of CounterFact samples
        intervention_fn: Optional intervention function

    Returns:
        Evaluation results dict
    """
    results = []
    correct = 0
    correct_after = 0

    for sample in samples:
        prompt = sample["prompt"]
        ground_truth = sample["ground_truth"]

        # Get model response
        original_output = _generate_response(model, prompt)
        is_correct = _check_correct(original_output, ground_truth)

        if is_correct:
            correct += 1

        # Apply intervention if provided
        intervened_output = None
        is_correct_after = is_correct

        if intervention_fn is not None:
            _, intervened_output = intervention_fn(model, prompt)
            is_correct_after = _check_correct(intervened_output, ground_truth)

            if is_correct_after:
                correct_after += 1

        results.append({
            "id": sample.get("id", ""),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "original_output": original_output,
            "intervened_output": intervened_output,
            "is_correct": is_correct,
            "is_correct_after": is_correct_after,
        })

    n_samples = len(samples)
    accuracy = correct / n_samples if n_samples > 0 else 0
    accuracy_after = correct_after / n_samples if n_samples > 0 else 0

    return {
        "accuracy": accuracy,
        "accuracy_after": accuracy_after,
        "accuracy_change": accuracy_after - accuracy,
        "per_sample_results": results,
        "n_samples": n_samples,
    }
