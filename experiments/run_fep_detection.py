"""Factual Emergence Point (FEP) Detection Experiment.

Core hypothesis: For each factual query, there exists a critical layer L_FEP
where the correct answer first emerges in the logit distribution (enters top-k).

This experiment:
1. Detects FEP for each TruthfulQA sample using logit lens
2. Records DoLa's dynamic premature layer selection
3. Computes correlation between FEP and DoLa's choice
4. Validates that DoLa succeeds by automatically finding FEP boundaries
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fep_detection")

RESULTS_DIR = PROJECT_ROOT / "results" / "fep_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str = "Qwen/Qwen2.5-7B"):
    """Load model via MechLens loader."""
    from mechlens.models.model_loader import load_model as ml_load
    from mechlens.config import DEFAULT_DEVICE
    logger.info(f"Loading model: {model_name} (device={DEFAULT_DEVICE})")
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


def run_fep_detection(model, dataset, max_samples: int = None):
    """Run FEP detection on dataset."""
    if max_samples is not None:
        dataset = dataset[:max_samples]
    
    results = []
    fep_distribution = defaultdict(int)
    
    logger.info(f"Running FEP detection on {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            logger.info(f"FEP detection: {i}/{len(dataset)}")
        
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


def analyze_fep_dola_correlation(results: list) -> dict:
    """Analyze correlation between FEP and DoLa's premature layer selection."""
    from scipy import stats
    
    fep_layers = []
    dola_layers = []
    
    n_layers = max(r["fep_layer"] for r in results if r["fep_layer"] < 100)
    
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
    
    return {
        "n_samples": len(fep_layers),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "dola_within_2_of_fep": within_2,
        "dola_within_5_of_fep": within_5,
        "mean_fep": float(np.mean(fep_arr)),
        "std_fep": float(np.std(fep_arr)),
        "mean_dola": float(np.mean(dola_arr)),
        "std_dola": float(np.std(dola_arr)),
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


def main():
    logger.info("=" * 60)
    logger.info("Factual Emergence Point (FEP) Detection Experiment")
    logger.info("=" * 60)
    
    model = load_model("Qwen/Qwen2.5-7B")
    dataset = load_truthfulqa_dataset()
    
    n_layers = model.cfg.n_layers
    logger.info(f"Model has {n_layers} layers")
    
    # Run FEP detection
    results, fep_distribution = run_fep_detection(model, dataset, max_samples=None)
    
    logger.info(f"FEP detection complete: {len(results)} samples analyzed")
    
    # Analyze FEP-DoLa correlation
    correlation = analyze_fep_dola_correlation(results)
    logger.info(f"FEP-DoLa correlation: Pearson r={correlation['pearson_r']:.4f}, "
                f"Spearman r={correlation['spearman_r']:.4f}")
    logger.info(f"DoLa within 2 layers of FEP: {correlation['dola_within_2_of_fep']:.2%}")
    logger.info(f"DoLa within 5 layers of FEP: {correlation['dola_within_5_of_fep']:.2%}")
    
    # Analyze by category
    category_stats = analyze_fep_by_category(results)
    
    # Save results
    output = {
        "model": "Qwen/Qwen2.5-7B",
        "n_layers": n_layers,
        "n_samples": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fep_distribution": fep_distribution,
        "correlation_analysis": correlation,
        "category_analysis": category_stats,
        "per_sample_results": results,
    }
    
    output_path = RESULTS_DIR / "fep_detection_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEP DETECTION SUMMARY")
    print("=" * 60)
    print(f"Model: Qwen/Qwen2.5-7B ({n_layers} layers)")
    print(f"Samples analyzed: {len(results)}")
    print(f"\nFEP Distribution:")
    for layer in sorted(fep_distribution.keys()):
        count = fep_distribution[layer]
        pct = count / len(results) * 100
        bar = "#" * int(pct / 2)
        print(f"  Layer {layer:2d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\nFEP-DoLa Correlation:")
    print(f"  Pearson r:  {correlation['pearson_r']:.4f} (p={correlation['pearson_p']:.2e})")
    print(f"  Spearman r: {correlation['spearman_r']:.4f} (p={correlation['spearman_p']:.2e})")
    print(f"  DoLa within ±2 of FEP: {correlation['dola_within_2_of_fep']:.1%}")
    print(f"  DoLa within ±5 of FEP: {correlation['dola_within_5_of_fep']:.1%}")
    
    print(f"\nMean FEP: {correlation['mean_fep']:.1f} ± {correlation['std_fep']:.1f}")
    print(f"Mean DoLa premature: {correlation['mean_dola']:.1f} ± {correlation['std_dola']:.1f}")
    
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
