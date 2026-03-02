"""Cross-Architecture FEP Detection Experiment.

Validates Late Crystallization phenomenon across multiple architectures:
- Llama-3.1-8B (GQA, 32 layers)
- Mistral-7B (GQA + Sliding Window Attention, 32 layers)

Compares with Qwen2.5-7B baseline to establish architecture-independent
generalization of the Late Crystallization theory.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cross_arch_fep")

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_architecture"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to evaluate for cross-architecture validation
# Downloaded via ModelScope, loaded from local paths
CROSS_ARCH_MODELS = [
    "meta-llama/Llama-3.1-8B",    # GQA, 32 layers
    "mistralai/Mistral-7B-v0.1",  # GQA + SWA, 32 layers
]

# Local paths for models downloaded via ModelScope
LOCAL_MODEL_PATHS = {
    "meta-llama/Llama-3.1-8B": "/root/.cache/modelscope/LLM-Research/Meta-Llama-3___1-8B",
    "mistralai/Mistral-7B-v0.1": "/root/.cache/modelscope/AI-ModelScope/Mistral-7B-v0___1",
}


def load_model(model_name: str):
    """Load model from local path via TransformerLens HookedTransformer."""
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = LOCAL_MODEL_PATHS.get(model_name)
    if local_path and os.path.isdir(local_path):
        logger.info(f"Loading model from local path: {local_path}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_path, torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True,
        )
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, tokenizer=tokenizer,
            torch_dtype=torch.float16, device="cuda",
        )
    else:
        logger.info(f"Loading model via default loader: {model_name}")
        from mechlens.models.model_loader import load_model as ml_load
        model = ml_load(model_name, dtype="float16")

    logger.info(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    return model


def load_truthfulqa_dataset():
    """Load TruthfulQA dataset."""
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")
    return dataset


def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to vocabulary logits via ln_final + W_U."""
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def detect_fep_for_sample(
    model,
    question: str,
    correct_answer: str,
    top_k: int = 10,
) -> dict:
    """Detect Factual Emergence Point for a single sample.
    
    FEP = the first layer where the correct answer token enters top-k predictions.
    
    Returns:
        Dict with fep_layer, layer_ranks, layer_probs, answer_tokens info
    """
    n_layers = model.cfg.n_layers
    
    # Format as Q/A prompt
    prompt = f"Q: {question}\nA:"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    # Get answer's first token (for tracking)
    answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)[0]
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0].item()
    
    # Cache all layer residuals
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    # Track answer rank and probability at each layer (last position)
    layer_ranks = []
    layer_probs = []
    layer_in_topk = []
    
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # [d_model]
        logits = unembed_at_layer(model, resid)  # [vocab]
        probs = F.softmax(logits.float(), dim=-1)
        
        # Get rank of target token
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank = rank[0].item() if len(rank) > 0 else probs.shape[0]
        
        prob = probs[target_token].item()
        in_topk = rank < top_k
        
        layer_ranks.append(rank)
        layer_probs.append(prob)
        layer_in_topk.append(in_topk)
    
    # Find FEP: first layer where answer enters top-k
    fep_layer = None
    for layer in range(n_layers):
        if layer_in_topk[layer]:
            fep_layer = layer
            break
    
    # If never enters top-k, set to n_layers (beyond all layers)
    if fep_layer is None:
        fep_layer = n_layers
    
    return {
        "fep_layer": fep_layer,
        "layer_ranks": layer_ranks,
        "layer_probs": layer_probs,
        "layer_in_topk": layer_in_topk,
        "target_token": target_token,
        "target_token_str": model.to_single_str_token(target_token),
        "final_rank": layer_ranks[-1],
        "final_prob": layer_probs[-1],
    }


def compute_dola_premature_layer(
    model,
    question: str,
    mature_layer: int = None,
    premature_candidates: list = None,
) -> int:
    """Compute which premature layer DoLa would dynamically select.
    
    DoLa selects the premature layer with highest Jensen-Shannon divergence
    from the mature layer.
    """
    n_layers = model.cfg.n_layers
    if mature_layer is None:
        mature_layer = n_layers - 1
    if premature_candidates is None:
        premature_candidates = list(range(0, int(n_layers * 0.6)))
    
    prompt = f"Q: {question}\nA:"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    all_layers = set(premature_candidates) | {mature_layer}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    # Get mature layer distribution
    mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, -1, :]
    mature_logits = unembed_at_layer(model, mature_resid)
    mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)
    mature_probs = mature_log_probs.exp()
    
    # Find premature layer with max JSD
    best_layer = premature_candidates[0]
    best_jsd = -1.0
    
    for layer in premature_candidates:
        p_resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        p_logits = unembed_at_layer(model, p_resid)
        p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
        p_probs = p_log_probs.exp()
        
        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        m_probs = 0.5 * (mature_probs + p_probs)
        m_log_probs = m_probs.log()
        
        kl_pm = F.kl_div(m_log_probs, mature_probs, reduction="sum", log_target=False)
        kl_qm = F.kl_div(m_log_probs, p_probs, reduction="sum", log_target=False)
        jsd = 0.5 * (kl_pm + kl_qm).item()
        
        if jsd > best_jsd:
            best_jsd = jsd
            best_layer = layer
    
    return best_layer


def run_fep_detection_for_model(model, dataset, model_name: str, max_samples: int = None):
    """Run FEP detection on dataset for a single model."""
    if max_samples is not None:
        dataset = dataset[:max_samples]
    
    results = []
    fep_distribution = defaultdict(int)
    n_layers = model.cfg.n_layers
    
    logger.info(f"Running FEP detection on {len(dataset)} samples for {model_name}...")
    
    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            logger.info(f"[{model_name}] FEP detection: {i}/{len(dataset)}")
        
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        
        if not best_answer.strip():
            continue
        
        # Detect FEP
        fep_result = detect_fep_for_sample(model, question, best_answer)
        
        if "error" in fep_result:
            continue
        
        # Get DoLa's premature layer selection
        dola_premature = compute_dola_premature_layer(model, question)
        
        entry = {
            "id": sample["id"],
            "question": question,
            "best_answer": best_answer,
            "category": sample.get("category", "Unknown"),
            "fep_layer": fep_result["fep_layer"],
            "dola_premature_layer": dola_premature,
            "final_rank": fep_result["final_rank"],
            "final_prob": fep_result["final_prob"],
            "layer_ranks": fep_result["layer_ranks"],
            "layer_probs": fep_result["layer_probs"],
            "target_token_str": fep_result["target_token_str"],
        }
        results.append(entry)
        
        fep_distribution[fep_result["fep_layer"]] += 1
    
    return results, dict(fep_distribution)


def analyze_fep_dola_correlation(results: list, n_layers: int) -> dict:
    """Analyze correlation between FEP and DoLa's premature layer selection."""
    from scipy import stats
    
    fep_layers = []
    dola_layers = []
    
    for r in results:
        fep = r["fep_layer"]
        dola = r["dola_premature_layer"]
        
        # Only include samples where FEP is within model layers
        if fep <= n_layers:
            fep_layers.append(fep)
            dola_layers.append(dola)
    
    fep_arr = np.array(fep_layers)
    dola_arr = np.array(dola_layers)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(fep_arr, dola_arr)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(fep_arr, dola_arr)
    
    # How often is DoLa's selection close to FEP?
    within_2 = np.mean(np.abs(fep_arr - dola_arr) <= 2)
    within_5 = np.mean(np.abs(fep_arr - dola_arr) <= 5)
    
    # Late crystallization percentage (FEP at final layer)
    late_crystallization_pct = np.mean(fep_arr == n_layers)
    
    return {
        "n_samples": len(fep_layers),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "dola_within_2_of_fep": float(within_2),
        "dola_within_5_of_fep": float(within_5),
        "mean_fep": float(np.mean(fep_arr)),
        "std_fep": float(np.std(fep_arr)),
        "mean_dola": float(np.mean(dola_arr)),
        "std_dola": float(np.std(dola_arr)),
        "late_crystallization_pct": float(late_crystallization_pct),
    }


def analyze_fep_by_category(results: list) -> dict:
    """Analyze FEP distribution by question category."""
    category_feps = defaultdict(list)
    
    for r in results:
        cat = r["category"]
        fep = r["fep_layer"]
        if fep < 100:  # Valid FEP
            category_feps[cat].append(fep)
    
    category_stats = {}
    for cat, feps in category_feps.items():
        if len(feps) >= 5:
            category_stats[cat] = {
                "n": len(feps),
                "mean_fep": float(np.mean(feps)),
                "std_fep": float(np.std(feps)),
                "min_fep": min(feps),
                "max_fep": max(feps),
            }
    
    return category_stats


def run_single_model(model_name: str, dataset: list, max_samples: int = None) -> dict:
    """Run complete FEP analysis for a single model."""
    logger.info("=" * 60)
    logger.info(f"Starting FEP Detection for: {model_name}")
    logger.info("=" * 60)
    
    # Load model
    model = load_model(model_name)
    n_layers = model.cfg.n_layers
    
    # Run FEP detection
    results, fep_distribution = run_fep_detection_for_model(
        model, dataset, model_name, max_samples
    )
    
    logger.info(f"FEP detection complete: {len(results)} samples analyzed")
    
    # Analyze FEP-DoLa correlation
    correlation = analyze_fep_dola_correlation(results, n_layers)
    logger.info(f"Late Crystallization: {correlation['late_crystallization_pct']:.1%} at final layer")
    logger.info(f"FEP-DoLa correlation: Pearson r={correlation['pearson_r']:.4f}")
    
    # Analyze by category
    category_stats = analyze_fep_by_category(results)
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_samples": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fep_distribution": fep_distribution,
        "correlation_analysis": correlation,
        "category_analysis": category_stats,
        "per_sample_results": results,
    }


def compare_architectures(all_results: dict) -> dict:
    """Compare Late Crystallization across architectures."""
    comparison = {
        "models": [],
        "late_crystallization_pcts": [],
        "mean_feps": [],
        "std_feps": [],
    }
    
    for model_name, result in all_results.items():
        corr = result["correlation_analysis"]
        comparison["models"].append(model_name)
        comparison["late_crystallization_pcts"].append(corr["late_crystallization_pct"])
        comparison["mean_feps"].append(corr["mean_fep"])
        comparison["std_feps"].append(corr["std_fep"])
    
    # Statistical comparison
    if len(comparison["late_crystallization_pcts"]) >= 2:
        pcts = comparison["late_crystallization_pcts"]
        comparison["min_late_crystallization"] = min(pcts)
        comparison["max_late_crystallization"] = max(pcts)
        comparison["range_late_crystallization"] = max(pcts) - min(pcts)
        comparison["all_above_75pct"] = all(p >= 0.75 for p in pcts)
    
    return comparison


def main():
    logger.info("=" * 70)
    logger.info("Cross-Architecture Late Crystallization Validation")
    logger.info("Models: Llama-3.1-8B (GQA) + Mistral-7B (GQA+SWA)")
    logger.info("=" * 70)
    
    # Load dataset once
    dataset = load_truthfulqa_dataset()
    logger.info(f"Loaded TruthfulQA dataset: {len(dataset)} samples")
    
    # Run FEP detection for each model
    all_results = {}
    
    for model_name in CROSS_ARCH_MODELS:
        try:
            result = run_single_model(model_name, dataset, max_samples=None)
            all_results[model_name] = result
            
            # Save individual model results
            safe_name = model_name.replace("/", "_").replace("-", "_")
            output_path = RESULTS_DIR / f"fep_{safe_name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            continue
    
    # Cross-architecture comparison
    comparison = compare_architectures(all_results)
    
    # Save combined results
    combined_output = {
        "experiment": "cross_architecture_fep_validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_tested": list(all_results.keys()),
        "comparison": comparison,
        "individual_results": all_results,
    }
    
    combined_path = RESULTS_DIR / "cross_architecture_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(combined_output, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Saved combined results to {combined_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE LATE CRYSTALLIZATION SUMMARY")
    print("=" * 70)
    
    for model_name, result in all_results.items():
        corr = result["correlation_analysis"]
        n_layers = result["n_layers"]
        print(f"\n{model_name}:")
        print(f"  Layers: {n_layers}")
        print(f"  Late Crystallization (FEP=final): {corr['late_crystallization_pct']:.1%}")
        print(f"  Mean FEP: {corr['mean_fep']:.1f} ± {corr['std_fep']:.1f}")
        print(f"  DoLa-FEP Correlation: r={corr['pearson_r']:.4f}")
    
    if comparison.get("all_above_75pct"):
        print("\n*** VALIDATION SUCCESS: All models show >75% Late Crystallization ***")
    else:
        print(f"\nLate Crystallization range: {comparison.get('min_late_crystallization', 0):.1%} - {comparison.get('max_late_crystallization', 0):.1%}")
    
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
