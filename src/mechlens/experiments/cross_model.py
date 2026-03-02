"""
Cross-Model Experiment Runner

Utilities for running experiments across all 4 supported models:
- Qwen2.5-0.5B
- Qwen2.5-7B
- Llama-3.1-8B
- Pythia-1.4B
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from mechlens.config import SUPPORTED_MODELS
from mechlens.models.model_loader import load_model
from mechlens.types import InterventionTarget, ComponentType

# Analysis
from mechlens.analysis.attention import analyze as analyze_attention
from mechlens.analysis.activation import analyze as analyze_activation, causal_trace
from mechlens.analysis.logit_lens import compute_logit_lens
from mechlens.analysis.circuit import discover as discover_circuit

# Intervention
from mechlens.intervention.ablation import ablate
from mechlens.intervention.scaling import scale

# Benchmark
from mechlens.benchmark.chinese_hallucination import (
    load_dataset as load_chinese_dataset,
    evaluate as evaluate_chinese,
    load_counterfact,
    evaluate_counterfact
)

logger = logging.getLogger(__name__)


ALL_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",
    "EleutherAI/pythia-1.4b"
]

QWEN_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B"
]

ROME_MEMIT_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B",
    "EleutherAI/pythia-1.4b"
]


@dataclass
class ExperimentConfig:
    """Configuration for cross-model experiments."""
    experiment_name: str
    models: List[str] = field(default_factory=lambda: ALL_MODELS)
    dtype: str = "float16"
    
    # Analysis options
    run_attention_analysis: bool = True
    run_activation_analysis: bool = True
    run_logit_lens: bool = True
    run_causal_trace: bool = True
    run_circuit_discovery: bool = True
    
    # Intervention options
    run_ablation: bool = True
    run_scaling: bool = True
    scale_factors: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.5, 2.0])
    
    # Evaluation options
    run_counterfact: bool = True
    run_chinese_bench: bool = True  # Only runs on Qwen models
    
    # Output
    output_dir: str = "results/cross_model"
    save_intermediate: bool = True


@dataclass
class ModelExperimentResult:
    """Results from a single model experiment."""
    model_name: str
    load_time_seconds: float = 0.0
    
    # Analysis results
    attention_analysis: Optional[Dict[str, Any]] = None
    activation_analysis: Optional[Dict[str, Any]] = None
    logit_lens_analysis: Optional[Dict[str, Any]] = None
    causal_trace_results: Optional[Dict[str, Any]] = None
    circuit_discovery: Optional[Dict[str, Any]] = None
    
    # Intervention results
    ablation_results: Optional[Dict[str, Any]] = None
    scaling_results: Optional[Dict[str, Any]] = None
    
    # Benchmark results
    counterfact_results: Optional[Dict[str, Any]] = None
    chinese_bench_results: Optional[Dict[str, Any]] = None
    
    # Errors
    errors: List[str] = field(default_factory=list)


@dataclass
class CrossModelResult:
    """Aggregated results across all models."""
    experiment_name: str
    config: ExperimentConfig
    model_results: Dict[str, ModelExperimentResult] = field(default_factory=dict)
    
    # Comparison metrics
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    total_time_seconds: float = 0.0


def run_cross_model_experiment(
    config: ExperimentConfig,
    test_prompts: Optional[List[str]] = None
) -> CrossModelResult:
    """
    Run experiments across all specified models.
    
    Args:
        config: Experiment configuration
        test_prompts: Optional list of test prompts (defaults to standard set)
    
    Returns:
        CrossModelResult with all model results
    """
    logger.info(f"Starting cross-model experiment: {config.experiment_name}")
    
    start_time = time.time()
    
    # Default test prompts
    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "In 1969, the first human walked on",
            "The chemical formula for water is",
            "Einstein developed the theory of",
        ]
    
    result = CrossModelResult(
        experiment_name=config.experiment_name,
        config=config
    )
    
    # Run experiments for each model
    for model_name in config.models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiments on: {model_name}")
        logger.info(f"{'='*50}")
        
        model_result = _run_single_model_experiment(
            model_name=model_name,
            config=config,
            test_prompts=test_prompts
        )
        
        result.model_results[model_name] = model_result
        
        # Save intermediate results
        if config.save_intermediate:
            _save_model_result(model_result, config.output_dir)
    
    # Compute comparison metrics
    result.comparison_metrics = _compute_comparison_metrics(result)
    
    result.total_time_seconds = time.time() - start_time
    
    # Save final results
    _save_cross_model_result(result, config.output_dir)
    
    logger.info(f"\nExperiment completed in {result.total_time_seconds:.1f}s")
    
    return result


def _run_single_model_experiment(
    model_name: str,
    config: ExperimentConfig,
    test_prompts: List[str]
) -> ModelExperimentResult:
    """Run all experiments on a single model."""
    
    result = ModelExperimentResult(model_name=model_name)
    
    try:
        # Load model
        load_start = time.time()
        model = load_model(model_name, dtype=config.dtype)
        result.load_time_seconds = time.time() - load_start
        logger.info(f"Model loaded in {result.load_time_seconds:.1f}s")
        
        # Use first prompt for analysis
        test_prompt = test_prompts[0]
        
        # Analysis
        if config.run_attention_analysis:
            result.attention_analysis = _run_attention_analysis(model, test_prompt)
        
        if config.run_activation_analysis:
            result.activation_analysis = _run_activation_analysis(model, test_prompt)
        
        if config.run_logit_lens:
            result.logit_lens_analysis = _run_logit_lens_analysis(model, test_prompt)
        
        if config.run_causal_trace:
            result.causal_trace_results = _run_causal_trace_analysis(model, test_prompt)
        
        if config.run_circuit_discovery:
            result.circuit_discovery = _run_circuit_discovery(model, test_prompt)
        
        # Intervention
        if config.run_ablation:
            result.ablation_results = _run_ablation_experiments(model, test_prompts)
        
        if config.run_scaling:
            result.scaling_results = _run_scaling_experiments(
                model, test_prompts, config.scale_factors
            )
        
        # Benchmarks
        if config.run_counterfact:
            result.counterfact_results = _run_counterfact_benchmark(model, model_name)
        
        if config.run_chinese_bench and model_name in QWEN_MODELS:
            result.chinese_bench_results = _run_chinese_benchmark(model, model_name)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        import traceback
        error_msg = f"Error running experiment on {model_name}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        result.errors.append(error_msg)
    
    return result


def _run_attention_analysis(model, prompt: str) -> Dict[str, Any]:
    """Run attention analysis."""
    logger.info("Running attention analysis...")
    try:
        attn_data = analyze_attention(model, prompt)
        
        return {
            "n_layers": attn_data.patterns.shape[0],
            "n_heads": attn_data.patterns.shape[1],
            "seq_len": attn_data.patterns.shape[2],
            "mean_entropy": float(attn_data.entropy.mean()) if attn_data.entropy is not None else None
        }
    except Exception as e:
        logger.warning(f"Attention analysis failed: {e}")
        return {"error": str(e)}


def _run_activation_analysis(model, prompt: str) -> Dict[str, Any]:
    """Run activation analysis."""
    logger.info("Running activation analysis...")
    try:
        act_data = analyze_activation(model, prompt)
        
        return {
            "n_layers": act_data.residual_stream.shape[0],
            "d_model": act_data.residual_stream.shape[2],
            "mean_residual_norm": float(act_data.residual_stream.norm(dim=-1).mean()),
            "mean_mlp_norm": float(act_data.mlp_output.norm(dim=-1).mean()) if act_data.mlp_output is not None else None,
            "mean_attn_norm": float(act_data.attn_output.norm(dim=-1).mean()) if act_data.attn_output is not None else None
        }
    except Exception as e:
        logger.warning(f"Activation analysis failed: {e}")
        return {"error": str(e)}


def _run_logit_lens_analysis(model, prompt: str) -> Dict[str, Any]:
    """Run logit lens analysis."""
    logger.info("Running logit lens analysis...")
    try:
        logit_lens = compute_logit_lens(model, prompt)
        
        # Get top prediction at each layer for last position
        n_layers = logit_lens.shape[0]
        top_probs = []
        
        for layer in range(n_layers):
            probs = logit_lens[layer, -1]  # Last position
            top_prob = float(probs.max())
            top_probs.append(top_prob)
        
        return {
            "n_layers": n_layers,
            "top_probs_per_layer": top_probs,
            "final_top_prob": top_probs[-1] if top_probs else None,
            "confidence_growth": (top_probs[-1] - top_probs[0]) if len(top_probs) > 1 else None
        }
    except Exception as e:
        logger.warning(f"Logit lens analysis failed: {e}")
        return {"error": str(e)}


def _run_causal_trace_analysis(model, prompt: str) -> Dict[str, Any]:
    """Run causal tracing analysis."""
    logger.info("Running causal trace analysis...")
    try:
        # Use first word as subject
        subject = prompt.split()[0] if prompt.split() else prompt[:5]
        
        trace_result = causal_trace(
            model=model,
            input_text=prompt,
            subject=subject,
            component_type=ComponentType.RESID
        )
        
        # Find most important layers
        layer_scores = list(enumerate(trace_result.patch_results))
        top_layers = sorted(layer_scores, key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "n_layers": len(trace_result.patch_results),
            "top_5_layers": [(l, float(s)) for l, s in top_layers],
            "max_recovery": float(max(trace_result.patch_results)),
            "mean_recovery": float(sum(trace_result.patch_results) / len(trace_result.patch_results))
        }
    except Exception as e:
        logger.warning(f"Causal trace failed: {e}")
        return {"error": str(e)}


def _run_circuit_discovery(model, prompt: str) -> Dict[str, Any]:
    """Run circuit discovery."""
    logger.info("Running circuit discovery...")
    try:
        circuit = discover_circuit(
            model=model,
            input_text=prompt,
            target_token_idx=-1,
            method="activation_patching",
            threshold=0.1
        )
        
        return {
            "n_nodes": len(circuit.nodes),
            "n_edges": len(circuit.edges),
            "faithfulness": float(circuit.faithfulness),
            "completeness": float(circuit.completeness),
            "top_nodes": [
                {"layer": n.layer, "type": n.component_type.value, "importance": float(n.importance)}
                for n in sorted(circuit.nodes, key=lambda x: x.importance, reverse=True)[:10]
            ]
        }
    except Exception as e:
        logger.warning(f"Circuit discovery failed: {e}")
        return {"error": str(e)}


def _run_ablation_experiments(model, prompts: List[str]) -> Dict[str, Any]:
    """Run ablation experiments."""
    logger.info("Running ablation experiments...")
    try:
        results = []
        n_layers = model.cfg.n_layers
        mid_layer = n_layers // 2
        
        # Ablate middle layer residual
        target = InterventionTarget(
            layer=mid_layer,
            component_type=ComponentType.RESID,
            component_idx=None
        )
        
        for prompt in prompts[:3]:  # Limit to 3 prompts
            ablation_result = ablate(model, prompt, [target], max_new_tokens=20)
            
            results.append({
                "prompt": prompt[:50],
                "original_output": ablation_result.original_output[:100],
                "modified_output": ablation_result.modified_output[:100],
                "kl_divergence": float(ablation_result.kl_divergence) if ablation_result.kl_divergence else None
            })
        
        return {
            "ablated_layer": mid_layer,
            "component": "residual",
            "results": results
        }
    except Exception as e:
        logger.warning(f"Ablation experiments failed: {e}")
        return {"error": str(e)}


def _run_scaling_experiments(
    model,
    prompts: List[str],
    scale_factors: List[float]
) -> Dict[str, Any]:
    """Run scaling experiments."""
    logger.info("Running scaling experiments...")
    try:
        n_layers = model.cfg.n_layers
        mid_layer = n_layers // 2
        
        target = InterventionTarget(
            layer=mid_layer,
            component_type=ComponentType.RESID,
            component_idx=None
        )
        
        prompt = prompts[0]
        results = []
        
        for factor in scale_factors:
            scaling_result = scale(model, prompt, [target], factor=factor, max_new_tokens=20)
            
            results.append({
                "scale_factor": factor,
                "modified_output": scaling_result.modified_output[:100],
                "kl_divergence": float(scaling_result.kl_divergence) if scaling_result.kl_divergence else None
            })
        
        return {
            "scaled_layer": mid_layer,
            "component": "residual",
            "prompt": prompt[:50],
            "original_output": results[0].get("modified_output", "") if results else "",
            "scaling_results": results
        }
    except Exception as e:
        logger.warning(f"Scaling experiments failed: {e}")
        return {"error": str(e)}


def _run_counterfact_benchmark(model, model_name: str) -> Dict[str, Any]:
    """Run CounterFact benchmark."""
    logger.info("Running CounterFact benchmark...")
    try:
        samples = load_counterfact("data/counterfact_sample.json")
        
        # Limit to subset for speed
        samples = samples[:20]
        
        results = evaluate_counterfact(
            model=model,
            samples=samples,
            intervention_fn=None  # No intervention, just baseline
        )
        
        return results
    except Exception as e:
        logger.warning(f"CounterFact benchmark failed: {e}")
        return {"error": str(e)}


def _run_chinese_benchmark(model, model_name: str) -> Dict[str, Any]:
    """Run Chinese hallucination benchmark (Qwen only)."""
    logger.info("Running Chinese hallucination benchmark...")
    try:
        dataset = load_chinese_dataset("data/chinese_hallucination_bench/dataset.json")
        
        # Limit to subset for speed
        dataset = dataset[:20]
        
        results = evaluate(
            model=model,
            dataset=dataset,
            intervention_fn=None,  # No intervention, just baseline
            model_name=model_name
        )
        
        return {
            "hallucination_rate": results["hallucination_rate"],
            "per_type_rates": results["per_type_rates"],
            "per_domain_rates": results["per_domain_rates"]
        }
    except Exception as e:
        logger.warning(f"Chinese benchmark failed: {e}")
        return {"error": str(e)}


def _compute_comparison_metrics(result: CrossModelResult) -> Dict[str, Any]:
    """Compute comparison metrics across models."""
    metrics = {
        "load_times": {},
        "attention_entropy": {},
        "activation_norms": {},
        "circuit_sizes": {},
        "counterfact_accuracy": {},
        "chinese_hallucination_rate": {}
    }
    
    for model_name, model_result in result.model_results.items():
        short_name = model_name.split("/")[-1]
        
        metrics["load_times"][short_name] = model_result.load_time_seconds
        
        if model_result.attention_analysis and "mean_entropy" in model_result.attention_analysis:
            metrics["attention_entropy"][short_name] = model_result.attention_analysis["mean_entropy"]
        
        if model_result.activation_analysis and "mean_residual_norm" in model_result.activation_analysis:
            metrics["activation_norms"][short_name] = model_result.activation_analysis["mean_residual_norm"]
        
        if model_result.circuit_discovery and "n_nodes" in model_result.circuit_discovery:
            metrics["circuit_sizes"][short_name] = model_result.circuit_discovery["n_nodes"]
        
        if model_result.counterfact_results and "accuracy" in model_result.counterfact_results:
            metrics["counterfact_accuracy"][short_name] = model_result.counterfact_results["accuracy"]
        
        if model_result.chinese_bench_results and "hallucination_rate" in model_result.chinese_bench_results:
            metrics["chinese_hallucination_rate"][short_name] = model_result.chinese_bench_results["hallucination_rate"]
    
    return metrics


def _save_model_result(result: ModelExperimentResult, output_dir: str):
    """Save individual model result."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name_safe = result.model_name.replace("/", "_")
    output_file = output_path / f"model_result_{model_name_safe}.json"
    
    # Convert to serializable dict
    result_dict = {
        "model_name": result.model_name,
        "load_time_seconds": result.load_time_seconds,
        "attention_analysis": result.attention_analysis,
        "activation_analysis": result.activation_analysis,
        "logit_lens_analysis": result.logit_lens_analysis,
        "causal_trace_results": result.causal_trace_results,
        "circuit_discovery": result.circuit_discovery,
        "ablation_results": result.ablation_results,
        "scaling_results": result.scaling_results,
        "counterfact_results": result.counterfact_results,
        "chinese_bench_results": result.chinese_bench_results,
        "errors": result.errors
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"Saved model result to {output_file}")


def _save_cross_model_result(result: CrossModelResult, output_dir: str):
    """Save cross-model experiment results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        "experiment_name": result.experiment_name,
        "total_time_seconds": result.total_time_seconds,
        "models": list(result.model_results.keys()),
        "comparison_metrics": result.comparison_metrics
    }
    
    with open(output_path / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved experiment summary to {output_path / 'experiment_summary.json'}")


def quick_comparison(
    models: Optional[List[str]] = None,
    prompt: str = "The capital of France is",
    output_dir: str = "results/quick_comparison"
) -> Dict[str, Any]:
    """
    Quick comparison across models with minimal analysis.
    
    Good for sanity checking all models work.
    """
    if models is None:
        models = ALL_MODELS
    
    results = {}
    
    for model_name in models:
        logger.info(f"\nTesting {model_name}...")
        
        try:
            model = load_model(model_name, dtype="float16")
            
            # Generate output
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                output = model.generate(
                    tokens,
                    max_new_tokens=20,
                    return_type="str"
                )
            
            if isinstance(output, list):
                output = output[0]
            
            results[model_name] = {
                "status": "success",
                "n_layers": model.cfg.n_layers,
                "n_heads": model.cfg.n_heads,
                "d_model": model.cfg.d_model,
                "output": output
            }
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "quick_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Quick sanity check
    print("Running quick comparison...")
    results = quick_comparison(
        models=["Qwen/Qwen2.5-0.5B"],  # Start with smallest model
        prompt="The capital of France is"
    )
    
    for model, result in results.items():
        print(f"\n{model}:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Output: {result['output'][:100]}")
