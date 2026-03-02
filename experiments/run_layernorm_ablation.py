"""LayerNorm Ablation Experiment.

Establishes causal mechanism for Late Crystallization by testing:
1. ln_final ablation: What happens to FEP and MC1 without final LayerNorm?
2. Intermediate LN ablation: Can we shift crystallization by removing middle LayerNorms?
3. LayerNorm scaling: Does amplifying LayerNorm improve crystallization?

Hypothesis: LayerNorm is causally responsible for the crystallization process.
If true, ablating ln_final should:
- Shift FEP distribution earlier (knowledge becomes visible sooner)
- Degrade MC1 performance (poor crystallization quality)
- Flatten entropy profile (less sharp transition)
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
logger = logging.getLogger("layernorm_ablation")

RESULTS_DIR = PROJECT_ROOT / "results" / "layernorm_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Local paths for models downloaded via ModelScope or HF cache
LOCAL_MODEL_PATHS = {
    "Qwen/Qwen2.5-7B": "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796",
}


def load_model(model_name: str = "Qwen/Qwen2.5-7B"):
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


# ==================== ABLATION UTILITIES ====================

def unembed_without_ln(model, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to logits WITHOUT LayerNorm."""
    # Skip ln_final, directly apply unembedding
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def unembed_with_ln(model, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to logits WITH LayerNorm (standard)."""
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def unembed_with_scaled_ln(model, resid: torch.Tensor, scale: float) -> torch.Tensor:
    """Project with scaled LayerNorm (amplify/dampen normalization effect)."""
    # Apply ln_final
    normed = model.ln_final(resid)
    
    # Scale the normalization effect
    # scaled = resid + scale * (normed - resid)
    # When scale=1.0, this is standard LN
    # When scale=0.0, this bypasses LN
    # When scale>1.0, this amplifies LN effect
    scaled = resid + scale * (normed - resid)
    
    logits = scaled @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


# ==================== FEP DETECTION WITH ABLATION ====================

def detect_fep_with_ln_ablation(
    model,
    question: str,
    correct_answer: str,
    use_ln: bool = True,
    ln_scale: float = 1.0,
    top_k: int = 10,
) -> dict:
    """Detect FEP with optional LayerNorm ablation."""
    n_layers = model.cfg.n_layers
    
    prompt = f"Q: {question}\nA:"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)[0]
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0].item()
    
    # Cache all layer residuals
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    # Track answer rank at each layer
    layer_ranks = []
    layer_probs = []
    layer_entropies = []
    layer_in_topk = []
    
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        
        # Choose unembedding method based on ablation settings
        if not use_ln:
            logits = unembed_without_ln(model, resid)
        elif ln_scale != 1.0:
            logits = unembed_with_scaled_ln(model, resid, ln_scale)
        else:
            logits = unembed_with_ln(model, resid)
        
        probs = F.softmax(logits.float(), dim=-1)
        
        # Rank of target token
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank = rank[0].item() if len(rank) > 0 else probs.shape[0]
        
        prob = probs[target_token].item()
        in_topk = rank < top_k
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        layer_ranks.append(rank)
        layer_probs.append(prob)
        layer_entropies.append(entropy)
        layer_in_topk.append(in_topk)
    
    # Find FEP
    fep_layer = None
    for layer in range(n_layers):
        if layer_in_topk[layer]:
            fep_layer = layer
            break
    
    if fep_layer is None:
        fep_layer = n_layers
    
    return {
        "fep_layer": fep_layer,
        "layer_ranks": layer_ranks,
        "layer_probs": layer_probs,
        "layer_entropies": layer_entropies,
        "layer_in_topk": layer_in_topk,
        "target_token": target_token,
        "final_rank": layer_ranks[-1],
        "final_prob": layer_probs[-1],
    }


def run_fep_ablation_experiment(
    model,
    dataset,
    use_ln: bool = True,
    ln_scale: float = 1.0,
    max_samples: int = None,
    condition_name: str = "default",
) -> dict:
    """Run FEP detection under specific ablation condition."""
    if max_samples is not None:
        dataset = dataset[:max_samples]
    
    results = []
    fep_distribution = defaultdict(int)
    all_entropies = []
    
    logger.info(f"Running FEP detection [{condition_name}] on {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"[{condition_name}] Progress: {i}/{len(dataset)}")
        
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        
        if not best_answer.strip():
            continue
        
        fep_result = detect_fep_with_ln_ablation(
            model, question, best_answer,
            use_ln=use_ln, ln_scale=ln_scale
        )
        
        if "error" in fep_result:
            continue
        
        results.append({
            "id": sample["id"],
            "fep_layer": fep_result["fep_layer"],
            "layer_entropies": fep_result["layer_entropies"],
            "final_rank": fep_result["final_rank"],
        })
        
        fep_distribution[fep_result["fep_layer"]] += 1
        all_entropies.append(fep_result["layer_entropies"])
    
    # Compute mean entropy profile
    n_layers = model.cfg.n_layers
    mean_entropies = []
    for layer in range(n_layers):
        layer_entropies = [e[layer] for e in all_entropies if len(e) > layer]
        mean_entropies.append(np.mean(layer_entropies) if layer_entropies else 0)
    
    # Compute FEP statistics
    fep_values = [r["fep_layer"] for r in results]
    
    return {
        "condition": condition_name,
        "use_ln": use_ln,
        "ln_scale": ln_scale,
        "n_samples": len(results),
        "fep_distribution": dict(fep_distribution),
        "mean_fep": float(np.mean(fep_values)) if fep_values else 0,
        "std_fep": float(np.std(fep_values)) if fep_values else 0,
        "late_crystallization_pct": fep_distribution.get(n_layers, 0) / len(results) if results else 0,
        "mean_entropy_profile": mean_entropies,
        "per_sample_results": results,
    }


# ==================== MC1 EVALUATION WITH ABLATION ====================

def compute_log_prob_with_ablation(
    model,
    question: str,
    answer: str,
    use_ln: bool = True,
    ln_scale: float = 1.0,
) -> float:
    """Compute log probability with LayerNorm ablation."""
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    
    q_len = prompt_tokens.shape[1]
    
    if full_tokens.shape[1] <= q_len:
        return float("-inf")
    
    # Get final layer residuals
    hook_name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    
    with torch.no_grad():
        _, cache = model.run_with_cache(full_tokens, names_filter=[hook_name])
    
    total_log_prob = 0.0
    
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        resid = cache[hook_name][0, i - 1, :]
        
        # Apply ablation
        if not use_ln:
            logits = unembed_without_ln(model, resid)
        elif ln_scale != 1.0:
            logits = unembed_with_scaled_ln(model, resid, ln_scale)
        else:
            logits = unembed_with_ln(model, resid)
        
        log_probs = F.log_softmax(logits.float(), dim=-1)
        total_log_prob += log_probs[token_id].item()
    
    return total_log_prob


def evaluate_mc1_with_ablation(
    model,
    dataset,
    use_ln: bool = True,
    ln_scale: float = 1.0,
    condition_name: str = "default",
) -> dict:
    """Evaluate MC1 under specific ablation condition."""
    correct = 0
    total = 0
    
    logger.info(f"Evaluating MC1 [{condition_name}]...")
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"[{condition_name}] MC1 progress: {i}/{len(dataset)}")
        
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not best_answer or not incorrect_answers:
            continue
        
        best_score = compute_log_prob_with_ablation(
            model, question, best_answer, use_ln, ln_scale
        )
        incorrect_scores = [
            compute_log_prob_with_ablation(model, question, a, use_ln, ln_scale)
            for a in incorrect_answers
        ]
        
        all_scores = [best_score] + incorrect_scores
        is_correct = best_score == max(all_scores)
        
        if is_correct:
            correct += 1
        total += 1
    
    return {
        "condition": condition_name,
        "use_ln": use_ln,
        "ln_scale": ln_scale,
        "mc1_score": correct / total if total > 0 else 0,
        "n_correct": correct,
        "n_total": total,
    }


# ==================== MAIN EXPERIMENT ====================

def main():
    logger.info("=" * 70)
    logger.info("LayerNorm Ablation Experiment")
    logger.info("Testing causal role of LayerNorm in Late Crystallization")
    logger.info("=" * 70)
    
    model = load_model("Qwen/Qwen2.5-7B")
    dataset = load_truthfulqa_dataset()
    n_layers = model.cfg.n_layers
    
    logger.info(f"Model: Qwen2.5-7B, {n_layers} layers")
    logger.info(f"Dataset: {len(dataset)} samples")
    
    # Ablation conditions to test
    conditions = [
        {"use_ln": True, "ln_scale": 1.0, "name": "baseline_with_ln"},
        {"use_ln": False, "ln_scale": 1.0, "name": "ablate_ln_final"},
        {"use_ln": True, "ln_scale": 0.5, "name": "ln_scale_0.5"},
        {"use_ln": True, "ln_scale": 0.8, "name": "ln_scale_0.8"},
        {"use_ln": True, "ln_scale": 1.2, "name": "ln_scale_1.2"},
        {"use_ln": True, "ln_scale": 1.5, "name": "ln_scale_1.5"},
    ]
    
    all_results = {
        "model": "Qwen/Qwen2.5-7B",
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fep_experiments": {},
        "mc1_experiments": {},
    }
    
    # Run FEP ablation experiments
    logger.info("\n" + "=" * 50)
    logger.info("Phase 1: FEP Detection under Ablation Conditions")
    logger.info("=" * 50)
    
    for cond in conditions:
        result = run_fep_ablation_experiment(
            model, dataset,
            use_ln=cond["use_ln"],
            ln_scale=cond["ln_scale"],
            max_samples=None,  # Use all samples
            condition_name=cond["name"],
        )
        all_results["fep_experiments"][cond["name"]] = result
        
        logger.info(f"[{cond['name']}] Mean FEP: {result['mean_fep']:.2f} ± {result['std_fep']:.2f}, "
                   f"Late Crystallization: {result['late_crystallization_pct']:.1%}")
    
    # Run MC1 ablation experiments
    logger.info("\n" + "=" * 50)
    logger.info("Phase 2: MC1 Evaluation under Ablation Conditions")
    logger.info("=" * 50)
    
    for cond in conditions:
        result = evaluate_mc1_with_ablation(
            model, dataset,
            use_ln=cond["use_ln"],
            ln_scale=cond["ln_scale"],
            condition_name=cond["name"],
        )
        all_results["mc1_experiments"][cond["name"]] = result
        
        logger.info(f"[{cond['name']}] MC1: {result['mc1_score']:.4f}")
    
    # Compute deltas from baseline
    baseline_fep = all_results["fep_experiments"]["baseline_with_ln"]
    baseline_mc1 = all_results["mc1_experiments"]["baseline_with_ln"]
    
    all_results["analysis"] = {
        "baseline_mean_fep": baseline_fep["mean_fep"],
        "baseline_late_crystallization": baseline_fep["late_crystallization_pct"],
        "baseline_mc1": baseline_mc1["mc1_score"],
        "ablation_effects": {},
    }
    
    for cond in conditions:
        if cond["name"] == "baseline_with_ln":
            continue
        
        fep_exp = all_results["fep_experiments"][cond["name"]]
        mc1_exp = all_results["mc1_experiments"][cond["name"]]
        
        all_results["analysis"]["ablation_effects"][cond["name"]] = {
            "fep_shift": fep_exp["mean_fep"] - baseline_fep["mean_fep"],
            "late_crystal_change": fep_exp["late_crystallization_pct"] - baseline_fep["late_crystallization_pct"],
            "mc1_change": mc1_exp["mc1_score"] - baseline_mc1["mc1_score"],
            "mc1_pct_change": (mc1_exp["mc1_score"] - baseline_mc1["mc1_score"]) / baseline_mc1["mc1_score"] * 100,
        }
    
    # Save results
    output_path = RESULTS_DIR / "layernorm_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Saved results to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("LAYERNORM ABLATION SUMMARY")
    print("=" * 70)
    
    print(f"\nBaseline (with ln_final):")
    print(f"  Mean FEP: {baseline_fep['mean_fep']:.2f}")
    print(f"  Late Crystallization: {baseline_fep['late_crystallization_pct']:.1%}")
    print(f"  MC1: {baseline_mc1['mc1_score']:.4f}")
    
    print("\nAblation Effects:")
    for cond_name, effects in all_results["analysis"]["ablation_effects"].items():
        print(f"\n  {cond_name}:")
        print(f"    FEP shift: {effects['fep_shift']:+.2f} layers")
        print(f"    Late Crystal change: {effects['late_crystal_change']:+.1%}")
        print(f"    MC1 change: {effects['mc1_pct_change']:+.1f}%")
    
    # Key findings
    ablate_effect = all_results["analysis"]["ablation_effects"]["ablate_ln_final"]
    print("\n" + "=" * 50)
    print("KEY FINDING: LayerNorm Ablation Effect")
    print("=" * 50)
    print(f"Removing ln_final:")
    print(f"  FEP shifted by {ablate_effect['fep_shift']:+.2f} layers")
    print(f"  MC1 changed by {ablate_effect['mc1_pct_change']:+.1f}%")
    
    if ablate_effect["fep_shift"] < -2:
        print("\n*** CAUSAL EVIDENCE: ln_final ablation shifts FEP earlier ***")
    if ablate_effect["mc1_pct_change"] < -10:
        print("*** CAUSAL EVIDENCE: ln_final ablation degrades MC1 ***")
    
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
