"""
Case Study 1: Hallucination Suppression Pipeline

Pipeline:
1. Load model and benchmark dataset
2. Run causal tracing to identify hallucination-critical layers
3. Apply targeted intervention (ablation/scaling)
4. Evaluate hallucination rate reduction
5. Optionally apply ROME/MEMIT editing
6. Compute ES/PS/NS metrics
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from mechlens.config import SUPPORTED_MODELS
from mechlens.models.model_loader import load_model
from mechlens.types import (
    InterventionTarget, ComponentType, InterventionType,
    HallucinationType, HallucinationDomain, HallucinationSample
)

# Analysis
from mechlens.analysis.activation import causal_trace
from mechlens.analysis.circuit import discover as discover_circuit

# Intervention
from mechlens.intervention.ablation import ablate
from mechlens.intervention.scaling import scale

# Editing
from mechlens.editing.rome import edit as rome_edit
from mechlens.editing.memit import edit as memit_edit, verify_model_support

# Benchmark
from mechlens.benchmark.chinese_hallucination import (
    load_dataset, evaluate, load_counterfact, evaluate_counterfact
)

logger = logging.getLogger(__name__)


@dataclass
class HallucinationStudyConfig:
    """Configuration for hallucination study."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "float16"
    
    # Causal tracing
    trace_num_samples: int = 10
    trace_noise_std: float = 0.1
    
    # Intervention
    intervention_type: str = "ablation"  # ablation or scaling
    scale_factor: float = 0.0  # for scaling
    top_k_layers: int = 3  # top-k layers to intervene
    
    # Editing (optional)
    apply_editing: bool = False
    editing_method: str = "rome"  # rome or memit
    
    # Evaluation
    max_new_tokens: int = 100
    
    # Output
    output_dir: str = "results/hallucination_study"


@dataclass 
class HallucinationStudyResult:
    """Results from hallucination study."""
    model_name: str
    config: HallucinationStudyConfig
    
    # Causal tracing results
    critical_layers: List[int] = field(default_factory=list)
    layer_importance_scores: Dict[int, float] = field(default_factory=dict)
    
    # Intervention results
    baseline_hallucination_rate: float = 0.0
    intervened_hallucination_rate: float = 0.0
    hallucination_reduction: float = 0.0
    
    # Per-type breakdown
    per_type_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_domain_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Editing results (if applied)
    editing_applied: bool = False
    edit_metrics: Optional[Dict[str, float]] = None
    edited_hallucination_rate: Optional[float] = None
    
    # Sample-level results
    sample_results: List[Dict[str, Any]] = field(default_factory=list)


def run_hallucination_study(
    config: HallucinationStudyConfig,
    dataset_path: str = "data/chinese_hallucination_bench/dataset.json"
) -> HallucinationStudyResult:
    """
    Run the complete hallucination suppression study.
    
    Pipeline:
    1. Load model and dataset
    2. Causal trace to find critical layers
    3. Apply intervention
    4. Evaluate hallucination reduction
    5. Optionally apply editing
    """
    logger.info(f"Starting hallucination study with model {config.model_name}")
    
    # Initialize result
    result = HallucinationStudyResult(
        model_name=config.model_name,
        config=config
    )
    
    # Step 1: Load model
    logger.info("Loading model...")
    model = load_model(config.model_name, dtype=config.dtype)
    
    # Step 2: Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    # Step 3: Causal tracing on sample subset
    logger.info("Running causal tracing...")
    critical_layers, importance_scores = _run_causal_tracing(
        model=model,
        dataset=dataset[:config.trace_num_samples],
        noise_std=config.trace_noise_std
    )
    result.critical_layers = critical_layers
    result.layer_importance_scores = importance_scores
    logger.info(f"Identified critical layers: {critical_layers}")
    
    # Step 4: Create intervention targets
    targets = _create_intervention_targets(
        critical_layers=critical_layers[:config.top_k_layers],
        model_name=config.model_name
    )
    
    # Step 5: Define intervention function
    def intervention_fn(m, text, targets=targets):
        if config.intervention_type == "ablation":
            return ablate(m, text, targets, max_new_tokens=config.max_new_tokens)
        else:
            return scale(m, text, targets, factor=config.scale_factor, max_new_tokens=config.max_new_tokens)
    
    # Step 6: Evaluate with intervention
    logger.info("Evaluating hallucination rates...")
    eval_results = evaluate(
        model=model,
        dataset=dataset,
        intervention_fn=lambda m, text: intervention_fn(m, text).modified_output,
        model_name=config.model_name
    )
    
    result.baseline_hallucination_rate = eval_results["hallucination_rate"]
    result.intervened_hallucination_rate = eval_results["hallucination_rate_after"]
    result.hallucination_reduction = eval_results["hallucination_rate_reduction"]
    result.per_type_results = eval_results["per_type_rates"]
    result.per_domain_results = eval_results["per_domain_rates"]
    result.sample_results = eval_results["per_sample_results"]
    
    logger.info(f"Baseline hallucination rate: {result.baseline_hallucination_rate:.2%}")
    logger.info(f"Intervened hallucination rate: {result.intervened_hallucination_rate:.2%}")
    logger.info(f"Reduction: {result.hallucination_reduction:.2%}")
    
    # Step 7: Optional editing
    if config.apply_editing and verify_model_support(config.model_name):
        logger.info("Applying weight editing...")
        result = _apply_editing(
            model=model,
            result=result,
            config=config,
            dataset=dataset
        )
    
    # Save results
    _save_results(result, config.output_dir)
    
    return result


def _run_causal_tracing(
    model,
    dataset: List[HallucinationSample],
    noise_std: float = 0.1
) -> Tuple[List[int], Dict[int, float]]:
    """
    Run causal tracing on sample subset to identify critical layers.
    
    Returns:
        critical_layers: List of layer indices sorted by importance
        importance_scores: Dict mapping layer -> importance score
    """
    n_layers = model.cfg.n_layers
    layer_scores = {i: 0.0 for i in range(n_layers)}
    
    for sample in dataset:
        try:
            # Use first word of question as subject for tracing
            subject = sample.question.split()[0] if sample.question else sample.question[:5]
            
            trace_result = causal_trace(
                model=model,
                input_text=sample.question,
                subject=subject,
                component_type=ComponentType.RESID
            )
            
            # Accumulate importance scores
            for layer, score in enumerate(trace_result.patch_results):
                layer_scores[layer] += score
                
        except Exception as e:
            logger.warning(f"Causal tracing failed for sample {sample.id}: {e}")
            continue
    
    # Normalize by number of samples
    n_samples = len(dataset)
    if n_samples > 0:
        layer_scores = {k: v / n_samples for k, v in layer_scores.items()}
    
    # Sort layers by importance
    sorted_layers = sorted(layer_scores.keys(), key=lambda x: layer_scores[x], reverse=True)
    
    return sorted_layers, layer_scores


def _create_intervention_targets(
    critical_layers: List[int],
    model_name: str
) -> List[InterventionTarget]:
    """Create intervention targets for critical layers."""
    targets = []
    
    for layer in critical_layers:
        # Target residual stream at each critical layer
        target = InterventionTarget(
            layer=layer,
            component_type=ComponentType.RESID,
            component_idx=None
        )
        targets.append(target)
    
    return targets


def _apply_editing(
    model,
    result: HallucinationStudyResult,
    config: HallucinationStudyConfig,
    dataset: List[HallucinationSample]
) -> HallucinationStudyResult:
    """Apply ROME/MEMIT editing and evaluate."""
    
    # Select a factual sample for editing demonstration
    factual_samples = [
        s for s in dataset 
        if s.hallucination_type == HallucinationType.FACTUAL_FABRICATION
    ]
    
    if not factual_samples:
        logger.warning("No factual samples found for editing")
        return result
    
    sample = factual_samples[0]
    
    # Parse subject and target from question/ground_truth
    # This is a simplified heuristic
    subject = sample.question.split("？")[0] if "？" in sample.question else sample.question[:20]
    target_new = sample.ground_truth
    target_old = "unknown"  # Placeholder
    
    try:
        if config.editing_method == "rome":
            edited_model, metrics = rome_edit(
                model=model,
                subject=subject,
                target_old=target_old,
                target_new=target_new,
                layers=result.critical_layers[:3]  # Use top-3 critical layers
            )
        else:  # memit
            edits = [{
                "subject": subject,
                "target_old": target_old,
                "target_new": target_new
            }]
            edited_model, metrics_list = memit_edit(
                model=model,
                edits=edits,
                layers=result.critical_layers[:3]
            )
            metrics = metrics_list[0] if metrics_list else None
        
        result.editing_applied = True
        if metrics:
            result.edit_metrics = {
                "efficacy_score": metrics.efficacy_score,
                "paraphrase_score": metrics.paraphrase_score,
                "neighborhood_score": metrics.neighborhood_score
            }
        
        # Re-evaluate with edited model
        edited_eval = evaluate(
            model=edited_model,
            dataset=dataset,
            intervention_fn=None,  # No intervention, just edited model
            model_name=config.model_name
        )
        result.edited_hallucination_rate = edited_eval["hallucination_rate"]
        
    except Exception as e:
        logger.error(f"Editing failed: {e}")
    
    return result


def _save_results(result: HallucinationStudyResult, output_dir: str):
    """Save study results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable dict
    result_dict = {
        "model_name": result.model_name,
        "config": {
            "model_name": result.config.model_name,
            "dtype": result.config.dtype,
            "intervention_type": result.config.intervention_type,
            "scale_factor": result.config.scale_factor,
            "top_k_layers": result.config.top_k_layers,
            "apply_editing": result.config.apply_editing,
            "editing_method": result.config.editing_method
        },
        "critical_layers": result.critical_layers,
        "layer_importance_scores": result.layer_importance_scores,
        "baseline_hallucination_rate": result.baseline_hallucination_rate,
        "intervened_hallucination_rate": result.intervened_hallucination_rate,
        "hallucination_reduction": result.hallucination_reduction,
        "per_type_results": result.per_type_results,
        "per_domain_results": result.per_domain_results,
        "editing_applied": result.editing_applied,
        "edit_metrics": result.edit_metrics,
        "edited_hallucination_rate": result.edited_hallucination_rate,
        "sample_results": result.sample_results
    }
    
    output_file = output_path / f"hallucination_study_{result.model_name.replace('/', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def run_counterfact_study(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dtype: str = "float16",
    dataset_path: str = "data/counterfact_sample.json",
    intervention_type: str = "ablation",
    output_dir: str = "results/counterfact_study"
) -> Dict[str, Any]:
    """
    Run CounterFact evaluation study (supports all 4 models).
    
    Similar pipeline but uses CounterFact dataset.
    """
    logger.info(f"Starting CounterFact study with model {model_name}")
    
    # Load model
    model = load_model(model_name, dtype=dtype)
    
    # Load dataset
    samples = load_counterfact(dataset_path)
    
    # Create simple intervention function
    def intervention_fn(m, text):
        # Ablate middle layer
        mid_layer = m.cfg.n_layers // 2
        target = InterventionTarget(
            layer=mid_layer,
            component_type=ComponentType.RESID,
            component_idx=None
        )
        result = ablate(m, text, [target], max_new_tokens=50)
        return result.modified_output
    
    # Evaluate
    results = evaluate_counterfact(
        model=model,
        samples=samples,
        intervention_fn=intervention_fn
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"counterfact_study_{model_name.replace('/', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"CounterFact results saved to {output_file}")
    
    return results


def compare_models(
    model_names: List[str],
    dataset_path: str = "data/counterfact_sample.json",
    output_dir: str = "results/model_comparison"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on CounterFact benchmark.
    """
    results = {}
    
    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")
        try:
            result = run_counterfact_study(
                model_name=model_name,
                dataset_path=dataset_path,
                output_dir=output_dir
            )
            results[model_name] = result
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run hallucination study on Qwen
    config = HallucinationStudyConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        intervention_type="ablation",
        apply_editing=True
    )
    
    result = run_hallucination_study(config)
    
    print(f"\n=== Hallucination Study Results ===")
    print(f"Model: {result.model_name}")
    print(f"Critical layers: {result.critical_layers[:5]}")
    print(f"Baseline rate: {result.baseline_hallucination_rate:.2%}")
    print(f"Intervened rate: {result.intervened_hallucination_rate:.2%}")
    print(f"Reduction: {result.hallucination_reduction:.2%}")
