"""Comprehensive Experiment Runner for Qwen2.5-7B.

Extends the MechLens experimental pipeline to a larger model (7B) to test
whether findings from small models (0.5B/1.4B) generalize to larger scales.

Key questions:
- Does knowledge remain concentrated in late layers, or does it distribute?
- Do more layers make fixed scaling even less effective?
- How does the causal tracing profile change with 28 vs 24 layers?

Pipeline:
1. Causal tracing v2 (28 layers)
2. Contrastive activation analysis
3. Targeted intervention (20 scaling strategies on Chinese benchmark)
4. Knowledge distribution metrics (entropy, Gini coefficient)
"""
import json
import gc
import logging
import os
import sys
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qwen_7b_runner")

MODEL_NAME = "Qwen/Qwen2.5-7B"


def load_model(try_int8: bool = False):
    """Load Qwen2.5-7B, with int8 fallback for OOM.

    Args:
        try_int8: If True, attempt int8 quantization (requires bitsandbytes)

    Returns:
        HookedTransformer model
    """
    from mechlens.models.model_loader import load_model as ml_load
    from mechlens.config import DEFAULT_DEVICE

    dtype = "float16"
    if DEFAULT_DEVICE == "mps":
        dtype = "float32"

    if try_int8:
        logger.info(f"Loading {MODEL_NAME} with int8 quantization")
        dtype = "int8"

    logger.info(f"Loading model: {MODEL_NAME} (device={DEFAULT_DEVICE}, dtype={dtype})")

    try:
        model = ml_load(MODEL_NAME, dtype=dtype)
        logger.info(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, d_model={model.cfg.d_model}")

        # Validate hook points are available
        if hasattr(model, "hook_dict"):
            hook_names = sorted(model.hook_dict.keys())
            logger.info(f"Hook dict has {len(hook_names)} hook points")
            # Log a few representative hooks for debugging
            sample_hooks = [h for h in hook_names if "blocks.0." in h]
            logger.info(f"Layer 0 hooks: {sample_hooks}")
        else:
            logger.warning("Model has no hook_dict — hooks may not work!")

        # Log VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        return model
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and not try_int8:
            logger.warning(f"OOM with fp16, falling back to int8")
            torch.cuda.empty_cache()
            gc.collect()
            return load_model(try_int8=True)
        raise


# ============================================================
# Phase 1: Causal Tracing v2
# ============================================================
def run_causal_tracing(model) -> dict:
    """Run causal tracing on 28-layer Qwen2.5-7B."""
    from mechlens.analysis.causal_tracing_v2 import (
        run_causal_tracing_v2,
        run_head_level_tracing,
    )

    test_cases = [
        {"text": "The capital of France is", "subject": "France"},
        {"text": "The Yangtze River flows through", "subject": "Yangtze River"},
        {"text": "Albert Einstein developed the theory of", "subject": "Einstein"},
        {"text": "The CEO of Apple Inc is", "subject": "Apple"},
        {"text": "The chemical formula for water is", "subject": "water"},
    ]

    results = {"model": MODEL_NAME, "layer_level": [], "head_level": []}

    for case in test_cases:
        logger.info(f"Causal tracing: '{case['text']}'")
        try:
            for comp in ["mlp", "attn"]:
                ct_result = run_causal_tracing_v2(
                    model,
                    input_text=case["text"],
                    subject=case["subject"],
                    component_type=comp,
                    noise_factor=10.0,
                    n_runs=5,
                    use_kl=True,
                )
                scores = ct_result.patch_results.tolist()
                top_layer = ct_result.patch_results.argmax().item()
                max_score = ct_result.patch_results.max().item()

                results["layer_level"].append({
                    "text": case["text"],
                    "subject": case["subject"],
                    "component": comp,
                    "scores": scores,
                    "top_layer": top_layer,
                    "max_recovery": max_score,
                    "base_output": ct_result.base_output,
                    "corrupted_output": ct_result.corrupted_output,
                })
                logger.info(f"  {comp}: top_layer={top_layer}, max_recovery={max_score:.4f}")
        except Exception as e:
            logger.error(f"  Error: {e}")
            results["layer_level"].append({
                "text": case["text"],
                "subject": case["subject"],
                "error": str(e),
            })

    # Head-level tracing (first 2 cases - expensive for 28 layers x 28 heads)
    for case in test_cases[:2]:
        logger.info(f"Head-level tracing: '{case['text']}'")
        try:
            head_result = run_head_level_tracing(
                model,
                input_text=case["text"],
                subject=case["subject"],
                noise_factor=10.0,
                n_runs=3,
            )
            results["head_level"].append({
                "text": case["text"],
                "subject": case["subject"],
                "top_heads": head_result["top_heads"][:10],
                "n_layers": head_result["n_layers"],
                "n_heads": head_result["n_heads"],
            })
        except Exception as e:
            logger.error(f"  Head-level error: {e}")

    return results


# ============================================================
# Phase 2: Contrastive Analysis
# ============================================================
def run_contrastive(model) -> dict:
    """Run contrastive analysis."""
    from mechlens.analysis.contrastive import run_contrastive_analysis

    correct_prompts = [
        ("The capital of France is Paris", "Paris", True),
        ("The chemical formula for water is H2O", "H2O", True),
        ("The first president of the United States was George Washington", "Washington", True),
    ]
    incorrect_prompts = [
        ("The capital of France is Berlin", "Berlin", False),
        ("The chemical formula for water is CO2", "CO2", False),
        ("The first president of the United States was Abraham Lincoln", "Lincoln", False),
    ]

    prompts_with_labels = correct_prompts + incorrect_prompts

    try:
        result = run_contrastive_analysis(model, prompts_with_labels, max_new_tokens=1)

        top_layers = sorted(
            range(len(result.layer_importance)),
            key=lambda i: result.layer_importance[i],
            reverse=True,
        )[:5]

        findings = {
            "model": MODEL_NAME,
            "layer_importance": result.layer_importance,
            "top_layers": top_layers,
            "top_neurons_per_layer": {
                str(k): v[:10] for k, v in result.top_neurons.items()
            },
        }
        logger.info(f"Contrastive top layers: {top_layers}")
        return findings

    except Exception as e:
        logger.error(f"Contrastive analysis failed: {e}")
        return {"model": MODEL_NAME, "error": str(e)}


# ============================================================
# Phase 3: Targeted Intervention (20 strategies)
# ============================================================
def run_extended_intervention(model, causal_results: dict, contrastive_results: dict) -> dict:
    """Run 20 scaling strategies on Chinese hallucination benchmark.

    Mirrors the extended intervention from run_comprehensive.py but
    on the 7B model.
    """
    from mechlens.benchmark.chinese_hallucination import evaluate, load_dataset
    from mechlens.types import ComponentType, InterventionTarget

    dataset_path = PROJECT_ROOT / "data" / "chinese_hallucination_bench" / "dataset.json"
    if not dataset_path.exists():
        logger.warning("Chinese hallucination dataset not found")
        return {"error": "dataset not found"}

    dataset = load_dataset(dataset_path)

    # Extract top layers from analysis results
    mlp_results = [r for r in causal_results.get("layer_level", [])
                    if r.get("component") == "mlp" and "scores" in r]
    attn_results = [r for r in causal_results.get("layer_level", [])
                     if r.get("component") == "attn" and "scores" in r]

    n_layers = model.cfg.n_layers  # 28

    if mlp_results:
        avg_mlp = [0.0] * len(mlp_results[0]["scores"])
        for r in mlp_results:
            for i, s in enumerate(r["scores"]):
                avg_mlp[i] += s / len(mlp_results)
        top_mlp = sorted(range(len(avg_mlp)), key=lambda i: avg_mlp[i], reverse=True)[:3]
    else:
        top_mlp = [n_layers * 2 // 3, n_layers * 2 // 3 + 1, n_layers * 2 // 3 + 2]

    if attn_results:
        avg_attn = [0.0] * len(attn_results[0]["scores"])
        for r in attn_results:
            for i, s in enumerate(r["scores"]):
                avg_attn[i] += s / len(attn_results)
        top_attn = sorted(range(len(avg_attn)), key=lambda i: avg_attn[i], reverse=True)[:3]
    else:
        top_attn = [n_layers * 2 // 3, n_layers * 2 // 3 + 1, n_layers * 2 // 3 + 2]

    contrastive_layers = contrastive_results.get("top_layers", top_mlp)[:3]

    # Define 20 strategies covering different components, layers, and factors
    strategies = {}

    # MLP dampening at various factors
    for factor in [0.7, 0.85, 0.9, 0.95]:
        strategies[f"mlp_dampen_{factor}"] = {
            "layers": top_mlp, "component": ComponentType.MLP_NEURON,
            "factor": factor, "hook": "hook_mlp_out",
        }

    # MLP amplification
    for factor in [1.05, 1.10, 1.15, 1.20]:
        strategies[f"mlp_amplify_{factor}"] = {
            "layers": top_mlp, "component": ComponentType.MLP_NEURON,
            "factor": factor, "hook": "hook_mlp_out",
        }

    # Attention dampening
    for factor in [0.7, 0.85, 0.9]:
        strategies[f"attn_dampen_{factor}"] = {
            "layers": top_attn, "component": ComponentType.ATTN_HEAD,
            "factor": factor, "hook": "attn.hook_result",
        }

    # Attention amplification
    for factor in [1.05, 1.10, 1.15]:
        strategies[f"attn_amplify_{factor}"] = {
            "layers": top_attn, "component": ComponentType.ATTN_HEAD,
            "factor": factor, "hook": "attn.hook_result",
        }

    # Residual stream at contrastive layers
    for factor in [0.85, 0.9, 1.10, 1.15]:
        strategies[f"resid_contrastive_{factor}"] = {
            "layers": contrastive_layers, "component": ComponentType.RESID,
            "factor": factor, "hook": "hook_resid_post",
        }

    # Late-layer broad interventions (layers 20-27 for 7B)
    late_layers = list(range(n_layers - 8, n_layers))
    strategies["late_mlp_dampen_0.9"] = {
        "layers": late_layers, "component": ComponentType.MLP_NEURON,
        "factor": 0.9, "hook": "hook_mlp_out",
    }
    strategies["late_resid_dampen_0.9"] = {
        "layers": late_layers, "component": ComponentType.RESID,
        "factor": 0.9, "hook": "hook_resid_post",
    }

    all_results = {
        "model": MODEL_NAME,
        "top_mlp_layers": top_mlp,
        "top_attn_layers": top_attn,
        "contrastive_layers": contrastive_layers,
        "strategies": {},
    }

    def _quick_generate(mdl, text, max_tokens=100):
        tokens = mdl.to_tokens(text)
        with torch.no_grad():
            output_ids = mdl.generate(tokens, max_new_tokens=max_tokens, do_sample=False)
        return mdl.to_string(output_ids[0, tokens.shape[1]:]).strip()

    for name, strategy in strategies.items():
        logger.info(f"Strategy: {name} (layers={strategy['layers']}, factor={strategy['factor']})")

        # Validate hook point exists before attempting intervention
        first_layer = strategy['layers'][0]
        test_hook = f"blocks.{first_layer}.{strategy['hook']}"
        layer_prefix = f"blocks.{first_layer}."
        if hasattr(model, "hook_dict") and test_hook not in model.hook_dict:
            available = [h for h in model.hook_dict if layer_prefix in h]
            logger.error(
                f"  Hook point '{test_hook}' not found in model. "
                f"Available hooks at layer {first_layer}: {available}"
            )
            all_results["strategies"][name] = {
                "error": f"Hook point '{test_hook}' not found",
                "available_hooks": available,
            }
            continue

        targets = [
            InterventionTarget(
                layer=layer,
                component_type=strategy["component"],
                component_id=0,
                factor=strategy["factor"],
            )
            for layer in strategy["layers"]
        ]

        def make_intervention_fn(tgts, scaling_factor):
            def intervention_fn(mdl, question):
                from mechlens.intervention import scale
                original = _quick_generate(mdl, question)
                result = scale(mdl, question, targets=tgts, factor=scaling_factor)
                return original, result.intervened_output
            return intervention_fn

        try:
            eval_result = evaluate(
                model,
                dataset,
                intervention_fn=make_intervention_fn(targets, strategy["factor"]),
                model_name=MODEL_NAME,
            )

            all_results["strategies"][name] = {
                "layers": strategy["layers"],
                "factor": strategy["factor"],
                "hook": strategy["hook"],
                "hallucination_rate": eval_result["hallucination_rate"],
                "hallucination_rate_after": eval_result["hallucination_rate_after"],
                "reduction": eval_result["hallucination_rate_reduction"],
                "per_type": eval_result["per_type_rates"],
            }
            logger.info(
                f"  Rate: {eval_result['hallucination_rate']:.3f} -> "
                f"{eval_result['hallucination_rate_after']:.3f} "
                f"(reduction: {eval_result['hallucination_rate_reduction']:+.3f})"
            )
        except Exception as e:
            logger.error(f"  Strategy {name} failed: {e}")
            all_results["strategies"][name] = {"error": str(e)}

    return all_results


# ============================================================
# Phase 4: Knowledge Distribution Metrics
# ============================================================
def compute_knowledge_distribution(causal_results: dict) -> dict:
    """Compute knowledge distribution metrics from causal tracing scores.

    Measures how knowledge is distributed across layers using:
    - Shannon entropy (higher = more distributed)
    - Gini coefficient (higher = more concentrated)
    - Top-K concentration ratio
    """
    import math

    metrics = {"model": MODEL_NAME, "mlp": {}, "attn": {}}

    for comp in ["mlp", "attn"]:
        results = [r for r in causal_results.get("layer_level", [])
                    if r.get("component") == comp and "scores" in r]

        if not results:
            continue

        # Average scores across prompts
        n_layers = len(results[0]["scores"])
        avg_scores = [0.0] * n_layers
        for r in results:
            for i, s in enumerate(r["scores"]):
                avg_scores[i] += s / len(results)

        # Normalize to probability distribution
        total = sum(max(0, s) for s in avg_scores)
        if total > 0:
            probs = [max(0, s) / total for s in avg_scores]
        else:
            probs = [1.0 / n_layers] * n_layers

        # Shannon entropy
        entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
        max_entropy = math.log2(n_layers)
        normalized_entropy = entropy / max_entropy

        # Gini coefficient
        sorted_probs = sorted(probs)
        n = len(sorted_probs)
        gini_sum = sum((2 * (i + 1) - n - 1) * sorted_probs[i] for i in range(n))
        gini = gini_sum / (n * sum(sorted_probs)) if sum(sorted_probs) > 0 else 0

        # Top-K concentration ratios
        sorted_scores = sorted(probs, reverse=True)
        top3_ratio = sum(sorted_scores[:3]) / sum(sorted_scores) if sum(sorted_scores) > 0 else 0
        top5_ratio = sum(sorted_scores[:5]) / sum(sorted_scores) if sum(sorted_scores) > 0 else 0

        metrics[comp] = {
            "avg_scores": avg_scores,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized_entropy,
            "gini_coefficient": gini,
            "top3_concentration": top3_ratio,
            "top5_concentration": top5_ratio,
            "n_layers": n_layers,
            "top_layer": avg_scores.index(max(avg_scores)),
            "max_score": max(avg_scores),
        }

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    results_dir = PROJECT_ROOT / "results" / "qwen_7b"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MechLens Qwen2.5-7B Comprehensive Experiment Suite")
    logger.info("=" * 60)

    model = load_model()

    all_results = {"model": MODEL_NAME, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Phase 1: Causal Tracing
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Causal Tracing v2 (28 layers)")
    logger.info("=" * 60)
    causal_results = run_causal_tracing(model)
    all_results["causal_tracing"] = causal_results
    with open(results_dir / "causal_tracing_results.json", "w") as f:
        json.dump(causal_results, f, indent=2, default=str)
    logger.info("Causal tracing results saved.")

    # Phase 2: Contrastive Analysis
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Contrastive Activation Analysis")
    logger.info("=" * 60)
    contrastive_results = run_contrastive(model)
    all_results["contrastive"] = contrastive_results
    with open(results_dir / "contrastive_results.json", "w") as f:
        json.dump(contrastive_results, f, indent=2, default=str)
    logger.info("Contrastive results saved.")

    # Phase 3: Extended Intervention (20 strategies)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Extended Intervention (20 strategies)")
    logger.info("=" * 60)
    intervention_results = run_extended_intervention(model, causal_results, contrastive_results)
    all_results["extended_intervention"] = intervention_results
    with open(results_dir / "extended_intervention.json", "w") as f:
        json.dump(intervention_results, f, indent=2, default=str)
    logger.info("Intervention results saved.")

    # Phase 4: Knowledge Distribution Metrics
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Knowledge Distribution Metrics")
    logger.info("=" * 60)
    distribution_metrics = compute_knowledge_distribution(causal_results)
    all_results["knowledge_distribution"] = distribution_metrics
    with open(results_dir / "knowledge_distribution.json", "w") as f:
        json.dump(distribution_metrics, f, indent=2, default=str)
    logger.info("Distribution metrics saved.")

    # Save combined results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("ALL QWEN2.5-7B EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to {results_dir}")
    logger.info("=" * 60)

    # Print summary
    print("\n\n=== QWEN2.5-7B EXPERIMENT SUMMARY ===")
    print(f"Model: {MODEL_NAME}")

    ct = all_results.get("causal_tracing", {})
    n_layer = len([r for r in ct.get("layer_level", []) if "scores" in r])
    n_head = len(ct.get("head_level", []))
    print(f"Causal Tracing: {n_layer} layer-level, {n_head} head-level results")

    cont = all_results.get("contrastive", {})
    if "top_layers" in cont:
        print(f"Contrastive Top Layers: {cont['top_layers']}")

    ext = all_results.get("extended_intervention", {})
    strategies = ext.get("strategies", {})
    n_success = len([s for s in strategies.values() if "reduction" in s])
    print(f"Extended Intervention: {n_success}/{len(strategies)} strategies completed")

    for name, res in strategies.items():
        if "reduction" in res:
            print(f"  {name:30s}: reduction={res['reduction']:+.3f}")

    dist = all_results.get("knowledge_distribution", {})
    for comp in ["mlp", "attn"]:
        if comp in dist and dist[comp]:
            d = dist[comp]
            print(f"\nKnowledge Distribution ({comp}):")
            print(f"  Normalized entropy: {d.get('normalized_entropy', 0):.4f}")
            print(f"  Gini coefficient: {d.get('gini_coefficient', 0):.4f}")
            print(f"  Top-3 concentration: {d.get('top3_concentration', 0):.4f}")
            print(f"  Top-5 concentration: {d.get('top5_concentration', 0):.4f}")


if __name__ == "__main__":
    main()
