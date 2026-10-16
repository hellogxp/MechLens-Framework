"""Cross-Architecture Intervention Experiments.

Validates intervention effectiveness hierarchy across architectures:
- Llama-3.1-8B (GQA, 32 layers)
- Mistral-7B (GQA + Sliding Window Attention, 32 layers)

Tests: Baseline, DoLa (dynamic), CAA (top_k=10), ITI (top_k=10)
Expected: DoLa > CAA > ITI > Simple Scaling (if Late Crystallization holds)
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

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
logger = logging.getLogger("cross_arch_interventions")

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_architecture"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to evaluate
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


# ==================== BASELINE EVALUATION ====================

def compute_answer_log_prob(model, question: str, answer: str) -> float:
    """Compute log probability of answer given question."""
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    
    q_len = prompt_tokens.shape[1]
    
    if full_tokens.shape[1] <= q_len:
        return float("-inf")
    
    with torch.no_grad():
        logits = model(full_tokens)
    
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    
    total_log_prob = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total_log_prob += log_probs[i - 1, token_id].item()
    
    return total_log_prob


def evaluate_mc1(model, dataset, score_fn=None) -> dict:
    """Evaluate MC1 (single-correct accuracy)."""
    if score_fn is None:
        score_fn = lambda m, q, a: compute_answer_log_prob(m, q, a)
    
    correct = 0
    total = 0
    per_sample = []
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"MC1 evaluation: {i}/{len(dataset)}")
        
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not best_answer or not incorrect_answers:
            continue
        
        # Compute log probs
        best_score = score_fn(model, question, best_answer)
        incorrect_scores = [score_fn(model, question, a) for a in incorrect_answers]
        
        # MC1: best answer should have highest score
        all_scores = [best_score] + incorrect_scores
        is_correct = best_score == max(all_scores)
        
        if is_correct:
            correct += 1
        total += 1
        
        per_sample.append({
            "id": sample["id"],
            "question": question,
            "is_correct": is_correct,
            "best_score": best_score,
            "max_incorrect_score": max(incorrect_scores) if incorrect_scores else None,
        })
    
    return {
        "mc1_score": correct / total if total > 0 else 0,
        "n_correct": correct,
        "n_total": total,
        "per_sample": per_sample,
    }


def evaluate_mc2(model, dataset, score_fn=None) -> dict:
    """Evaluate MC2 (normalized probability mass on correct answers)."""
    if score_fn is None:
        score_fn = lambda m, q, a: compute_answer_log_prob(m, q, a)
    
    mc2_scores = []
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"MC2 evaluation: {i}/{len(dataset)}")
        
        question = sample["question"]
        correct_answers = sample.get("correct_answers", [])
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not correct_answers:
            correct_answers = [sample.get("best_answer", "")]
        
        if not correct_answers[0] or not incorrect_answers:
            continue
        
        # Compute log probs for all answers
        correct_scores = [score_fn(model, question, a) for a in correct_answers]
        incorrect_scores = [score_fn(model, question, a) for a in incorrect_answers]
        
        # MC2: softmax normalized probability mass on correct answers
        all_scores = correct_scores + incorrect_scores
        all_scores_tensor = torch.tensor(all_scores)
        probs = F.softmax(all_scores_tensor, dim=0)
        
        mc2 = probs[:len(correct_scores)].sum().item()
        mc2_scores.append(mc2)
    
    return {
        "mc2_score": np.mean(mc2_scores) if mc2_scores else 0,
        "n_samples": len(mc2_scores),
    }


# ==================== DOLA IMPLEMENTATION ====================

def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to vocabulary logits."""
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def compute_dola_log_prob(model, question: str, answer: str, premature_layers: list = None) -> float:
    """Compute DoLa-adjusted log probability."""
    n_layers = model.cfg.n_layers
    mature_layer = n_layers - 1
    
    if premature_layers is None:
        premature_layers = list(range(0, int(n_layers * 0.6)))
    
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    
    q_len = prompt_tokens.shape[1]
    
    if full_tokens.shape[1] <= q_len:
        return float("-inf")
    
    # Cache all needed layers
    all_layers = set(premature_layers) | {mature_layer}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]
    
    with torch.no_grad():
        _, cache = model.run_with_cache(full_tokens, names_filter=hook_names)
    
    total_log_prob = 0.0
    
    for pos in range(q_len, full_tokens.shape[1]):
        target_token = full_tokens[0, pos].item()
        
        # Get mature layer logits
        mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, pos - 1, :]
        mature_logits = unembed_at_layer(model, mature_resid)
        mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)
        mature_probs = mature_log_probs.exp()
        
        # Dynamic selection: find premature layer with max JSD
        best_premature = premature_layers[0]
        best_jsd = -1.0
        
        for p_layer in premature_layers:
            p_resid = cache[f"blocks.{p_layer}.hook_resid_post"][0, pos - 1, :]
            p_logits = unembed_at_layer(model, p_resid)
            p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
            p_probs = p_log_probs.exp()
            
            # JSD
            m_probs = 0.5 * (mature_probs + p_probs)
            m_log_probs = m_probs.log()
            kl_pm = F.kl_div(m_log_probs, mature_probs, reduction="sum", log_target=False)
            kl_qm = F.kl_div(m_log_probs, p_probs, reduction="sum", log_target=False)
            jsd = 0.5 * (kl_pm + kl_qm).item()
            
            if jsd > best_jsd:
                best_jsd = jsd
                best_premature = p_layer
        
        # Compute DoLa contrasted log prob
        p_resid = cache[f"blocks.{best_premature}.hook_resid_post"][0, pos - 1, :]
        p_logits = unembed_at_layer(model, p_resid)
        p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
        
        # DoLa: log_softmax(mature) - log_softmax(premature)
        dola_logits = mature_log_probs - p_log_probs
        dola_log_probs = F.log_softmax(dola_logits, dim=-1)
        
        total_log_prob += dola_log_probs[target_token].item()
    
    return total_log_prob


# ==================== CAA IMPLEMENTATION ====================

def learn_caa_directions(model, dataset, top_k_layers: int = 10) -> dict:
    """Learn CAA directions from contrastive pairs."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # Use last top_k layers
    target_layers = list(range(n_layers - top_k_layers, n_layers))
    
    # Collect activations
    correct_acts = {l: [] for l in target_layers}
    incorrect_acts = {l: [] for l in target_layers}
    
    hook_names = [f"blocks.{l}.hook_resid_post" for l in target_layers]
    
    # Use first 100 samples for direction learning
    for sample in dataset[:100]:
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not best_answer or not incorrect_answers:
            continue
        
        # Correct answer activation
        correct_text = f"Q: {question}\nA: {best_answer}"
        correct_tokens = model.to_tokens(correct_text, prepend_bos=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(correct_tokens, names_filter=hook_names)
        
        for l in target_layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
            correct_acts[l].append(act)
        
        # Incorrect answer activation (use first incorrect)
        incorrect_text = f"Q: {question}\nA: {incorrect_answers[0]}"
        incorrect_tokens = model.to_tokens(incorrect_text, prepend_bos=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(incorrect_tokens, names_filter=hook_names)
        
        for l in target_layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
            incorrect_acts[l].append(act)
    
    # Compute directions: correct - incorrect
    directions = {}
    for l in target_layers:
        if correct_acts[l] and incorrect_acts[l]:
            correct_mean = torch.stack(correct_acts[l]).mean(dim=0)
            incorrect_mean = torch.stack(incorrect_acts[l]).mean(dim=0)
            direction = correct_mean - incorrect_mean
            direction = direction / direction.norm()
            directions[l] = direction
    
    return {
        "target_layers": target_layers,
        "directions": directions,
        "n_samples_used": min(len(correct_acts[target_layers[0]]), 100),
    }


def compute_caa_log_prob(model, question: str, answer: str, caa_info: dict, coeff: float = 5.0) -> float:
    """Compute CAA-steered log probability."""
    directions = caa_info["directions"]
    target_layers = caa_info["target_layers"]
    
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    
    q_len = prompt_tokens.shape[1]
    
    if full_tokens.shape[1] <= q_len:
        return float("-inf")
    
    # Create hooks for CAA steering
    def make_caa_hook(layer_idx):
        direction = directions[layer_idx].to(model.cfg.device, model.cfg.dtype)
        
        def hook_fn(activation, hook):
            # Add direction to residual stream
            steering = coeff * direction
            activation[:, :, :] = activation + steering.unsqueeze(0).unsqueeze(0)
            return activation
        
        return hook_fn
    
    hooks = [(f"blocks.{l}.hook_resid_post", make_caa_hook(l)) for l in target_layers if l in directions]
    
    with torch.no_grad():
        logits = model.run_with_hooks(full_tokens, fwd_hooks=hooks)
    
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    
    total_log_prob = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total_log_prob += log_probs[i - 1, token_id].item()
    
    return total_log_prob


# ==================== ITI IMPLEMENTATION ====================

def learn_iti_directions(model, dataset, top_k_heads: int = 10) -> dict:
    """Learn ITI directions from contrastive pairs."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # Collect head-level activations
    head_acts_correct = {}
    head_acts_incorrect = {}
    
    for l in range(n_layers):
        head_acts_correct[l] = []
        head_acts_incorrect[l] = []
    
    hook_names = [f"blocks.{l}.attn.hook_result" for l in range(n_layers)]
    
    # Use first 100 samples
    for sample in dataset[:100]:
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not best_answer or not incorrect_answers:
            continue
        
        # Correct
        correct_text = f"Q: {question}\nA: {best_answer}"
        correct_tokens = model.to_tokens(correct_text, prepend_bos=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(correct_tokens, names_filter=hook_names)
        
        for l in range(n_layers):
            act = cache[f"blocks.{l}.attn.hook_result"][0, -1, :].cpu()
            head_acts_correct[l].append(act)
        
        # Incorrect
        incorrect_text = f"Q: {question}\nA: {incorrect_answers[0]}"
        incorrect_tokens = model.to_tokens(incorrect_text, prepend_bos=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(incorrect_tokens, names_filter=hook_names)
        
        for l in range(n_layers):
            act = cache[f"blocks.{l}.attn.hook_result"][0, -1, :].cpu()
            head_acts_incorrect[l].append(act)
    
    # Compute per-layer directions and importance scores
    layer_directions = {}
    layer_importance = {}
    
    for l in range(n_layers):
        if head_acts_correct[l] and head_acts_incorrect[l]:
            correct_mean = torch.stack(head_acts_correct[l]).mean(dim=0)
            incorrect_mean = torch.stack(head_acts_incorrect[l]).mean(dim=0)
            direction = correct_mean - incorrect_mean
            importance = direction.norm().item()
            direction = direction / (direction.norm() + 1e-8)
            layer_directions[l] = direction
            layer_importance[l] = importance
    
    # Select top-k layers by importance
    sorted_layers = sorted(layer_importance.keys(), key=lambda l: layer_importance[l], reverse=True)
    top_layers = sorted_layers[:top_k_heads]
    
    return {
        "top_layers": top_layers,
        "directions": {l: layer_directions[l] for l in top_layers},
        "importance": {l: layer_importance[l] for l in top_layers},
    }


def compute_iti_log_prob(model, question: str, answer: str, iti_info: dict, coeff: float = 3.0) -> float:
    """Compute ITI-steered log probability."""
    directions = iti_info["directions"]
    top_layers = iti_info["top_layers"]
    
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    
    q_len = prompt_tokens.shape[1]
    
    if full_tokens.shape[1] <= q_len:
        return float("-inf")
    
    # Create ITI hooks
    def make_iti_hook(layer_idx):
        direction = directions[layer_idx].to(model.cfg.device, model.cfg.dtype)
        
        def hook_fn(activation, hook):
            steering = coeff * direction
            activation[:, :, :] = activation + steering.unsqueeze(0).unsqueeze(0)
            return activation
        
        return hook_fn
    
    hooks = [(f"blocks.{l}.attn.hook_result", make_iti_hook(l)) for l in top_layers]
    
    with torch.no_grad():
        logits = model.run_with_hooks(full_tokens, fwd_hooks=hooks)
    
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    
    total_log_prob = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total_log_prob += log_probs[i - 1, token_id].item()
    
    return total_log_prob


# ==================== MAIN EXPERIMENT ====================

def run_model_experiments(model_name: str, dataset: list) -> dict:
    """Run all intervention experiments for a single model."""
    logger.info("=" * 60)
    logger.info(f"Running Interventions for: {model_name}")
    logger.info("=" * 60)
    
    model = load_model(model_name)
    n_layers = model.cfg.n_layers
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Phase 1: Baseline
    logger.info("Phase 1: Baseline MC1/MC2")
    baseline_mc1 = evaluate_mc1(model, dataset)
    baseline_mc2 = evaluate_mc2(model, dataset)
    results["baseline"] = {
        "mc1": baseline_mc1["mc1_score"],
        "mc2": baseline_mc2["mc2_score"],
    }
    logger.info(f"Baseline MC1: {baseline_mc1['mc1_score']:.4f}, MC2: {baseline_mc2['mc2_score']:.4f}")
    
    # Phase 2: DoLa Dynamic
    logger.info("Phase 2: DoLa Dynamic")
    dola_score_fn = lambda m, q, a: compute_dola_log_prob(m, q, a)
    dola_mc1 = evaluate_mc1(model, dataset, score_fn=dola_score_fn)
    dola_mc2 = evaluate_mc2(model, dataset, score_fn=dola_score_fn)
    results["dola_dynamic"] = {
        "mc1": dola_mc1["mc1_score"],
        "mc2": dola_mc2["mc2_score"],
        "mc1_improvement": (dola_mc1["mc1_score"] - baseline_mc1["mc1_score"]) / baseline_mc1["mc1_score"] * 100,
    }
    logger.info(f"DoLa MC1: {dola_mc1['mc1_score']:.4f} (+{results['dola_dynamic']['mc1_improvement']:.1f}%)")
    
    # Phase 3: CAA (top_k=10, coeff=5.0)
    logger.info("Phase 3: CAA (top_k=10, coeff=5.0)")
    caa_info = learn_caa_directions(model, dataset, top_k_layers=10)
    caa_score_fn = lambda m, q, a: compute_caa_log_prob(m, q, a, caa_info, coeff=5.0)
    caa_mc1 = evaluate_mc1(model, dataset, score_fn=caa_score_fn)
    caa_mc2 = evaluate_mc2(model, dataset, score_fn=caa_score_fn)
    results["caa_top10_coeff5"] = {
        "mc1": caa_mc1["mc1_score"],
        "mc2": caa_mc2["mc2_score"],
        "mc1_improvement": (caa_mc1["mc1_score"] - baseline_mc1["mc1_score"]) / baseline_mc1["mc1_score"] * 100,
    }
    logger.info(f"CAA MC1: {caa_mc1['mc1_score']:.4f} (+{results['caa_top10_coeff5']['mc1_improvement']:.1f}%)")
    
    # Phase 4: ITI (top_k=10, coeff=3.0)
    logger.info("Phase 4: ITI (top_k=10, coeff=3.0)")
    iti_info = learn_iti_directions(model, dataset, top_k_heads=10)
    iti_score_fn = lambda m, q, a: compute_iti_log_prob(m, q, a, iti_info, coeff=3.0)
    iti_mc1 = evaluate_mc1(model, dataset, score_fn=iti_score_fn)
    iti_mc2 = evaluate_mc2(model, dataset, score_fn=iti_score_fn)
    results["iti_top10_coeff3"] = {
        "mc1": iti_mc1["mc1_score"],
        "mc2": iti_mc2["mc2_score"],
        "mc1_improvement": (iti_mc1["mc1_score"] - baseline_mc1["mc1_score"]) / baseline_mc1["mc1_score"] * 100,
    }
    logger.info(f"ITI MC1: {iti_mc1['mc1_score']:.4f} (+{results['iti_top10_coeff3']['mc1_improvement']:.1f}%)")
    
    # Summary
    results["summary"] = {
        "effectiveness_ranking": [],
        "late_crystallization_validated": False,
    }
    
    methods = [
        ("dola_dynamic", results["dola_dynamic"]["mc1"]),
        ("caa_top10_coeff5", results["caa_top10_coeff5"]["mc1"]),
        ("iti_top10_coeff3", results["iti_top10_coeff3"]["mc1"]),
        ("baseline", results["baseline"]["mc1"]),
    ]
    methods.sort(key=lambda x: x[1], reverse=True)
    results["summary"]["effectiveness_ranking"] = [m[0] for m in methods]
    
    # Check if DoLa > CAA > ITI (expected from Late Crystallization)
    if (results["dola_dynamic"]["mc1"] > results["caa_top10_coeff5"]["mc1"] > 
        results["iti_top10_coeff3"]["mc1"] > results["baseline"]["mc1"]):
        results["summary"]["late_crystallization_validated"] = True
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("Cross-Architecture Intervention Validation")
    logger.info("Testing: Baseline, DoLa, CAA, ITI")
    logger.info("Models: Llama-3.1-8B (GQA) + Mistral-7B (GQA+SWA)")
    logger.info("=" * 70)
    
    # Load dataset
    dataset = load_truthfulqa_dataset()
    logger.info(f"Loaded TruthfulQA dataset: {len(dataset)} samples")
    
    # Run experiments for each model
    all_results = {}
    
    for model_name in CROSS_ARCH_MODELS:
        try:
            result = run_model_experiments(model_name, dataset)
            all_results[model_name] = result
            
            # Save individual results
            safe_name = model_name.replace("/", "_").replace("-", "_")
            output_path = RESULTS_DIR / f"interventions_{safe_name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cross-architecture comparison
    comparison = {
        "models": list(all_results.keys()),
        "dola_improvements": [],
        "caa_improvements": [],
        "iti_improvements": [],
        "ranking_consistency": [],
    }
    
    for model_name, result in all_results.items():
        comparison["dola_improvements"].append(result["dola_dynamic"]["mc1_improvement"])
        comparison["caa_improvements"].append(result["caa_top10_coeff5"]["mc1_improvement"])
        comparison["iti_improvements"].append(result["iti_top10_coeff3"]["mc1_improvement"])
        comparison["ranking_consistency"].append(result["summary"]["late_crystallization_validated"])
    
    comparison["all_rankings_consistent"] = all(comparison["ranking_consistency"])
    
    # Save combined results
    combined = {
        "experiment": "cross_architecture_interventions",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": comparison,
        "individual_results": all_results,
    }
    
    combined_path = RESULTS_DIR / "cross_architecture_interventions.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE INTERVENTION SUMMARY")
    print("=" * 70)
    
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Baseline MC1: {result['baseline']['mc1']:.4f}")
        print(f"  DoLa MC1:     {result['dola_dynamic']['mc1']:.4f} (+{result['dola_dynamic']['mc1_improvement']:.1f}%)")
        print(f"  CAA MC1:      {result['caa_top10_coeff5']['mc1']:.4f} (+{result['caa_top10_coeff5']['mc1_improvement']:.1f}%)")
        print(f"  ITI MC1:      {result['iti_top10_coeff3']['mc1']:.4f} (+{result['iti_top10_coeff3']['mc1_improvement']:.1f}%)")
        print(f"  Ranking: {' > '.join(result['summary']['effectiveness_ranking'])}")
        print(f"  Late Crystallization Validated: {result['summary']['late_crystallization_validated']}")
    
    if comparison["all_rankings_consistent"]:
        print("\n*** SUCCESS: DoLa > CAA > ITI hierarchy holds across all architectures ***")
    else:
        print("\nRanking consistency:", comparison["ranking_consistency"])
    
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
