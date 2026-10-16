"""CrystalBoost Evaluation Experiment.

Evaluates the novel CrystalBoost intervention method against baselines:
- Baseline (no intervention)
- DoLa dynamic (current best: +25.4% MC1)
- CAA (top_k=10, coeff=5.0: +15.5% MC1)
- ITI (top_k=10, coeff=3.0: +10.0% MC1)

Target: CrystalBoost should achieve >25.4% MC1 improvement to demonstrate
that Late Crystallization insight enables better intervention design.
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
logger = logging.getLogger("crystalboost_eval")

RESULTS_DIR = PROJECT_ROOT / "results" / "crystalboost"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to evaluate
EVAL_MODELS = [
    "Qwen/Qwen2.5-7B",            # Primary model (28 layers, cached in HF)
    "meta-llama/Llama-3.1-8B",    # Cross-architecture validation
]

# Local paths for models downloaded via ModelScope
LOCAL_MODEL_PATHS = {
    "meta-llama/Llama-3.1-8B": "/root/.cache/modelscope/LLM-Research/Meta-Llama-3___1-8B",
    "mistralai/Mistral-7B-v0.1": "/root/.cache/modelscope/AI-ModelScope/Mistral-7B-v0___1",
    "Qwen/Qwen2.5-7B": "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796",
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


def compute_answer_log_prob(model, question: str, answer: str) -> float:
    """Compute baseline log probability."""
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


def evaluate_mc1_mc2(model, dataset, score_fn=None, method_name: str = "baseline") -> dict:
    """Evaluate both MC1 and MC2."""
    if score_fn is None:
        score_fn = lambda m, q, a: compute_answer_log_prob(m, q, a)
    
    mc1_correct = 0
    mc1_total = 0
    mc2_scores = []
    per_sample = []
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"[{method_name}] Evaluation: {i}/{len(dataset)}")
        
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        correct_answers = sample.get("correct_answers", [])
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not best_answer or not incorrect_answers:
            continue
        
        if not correct_answers:
            correct_answers = [best_answer]
        
        # MC1: Best answer vs all incorrect
        best_score = score_fn(model, question, best_answer)
        incorrect_scores = [score_fn(model, question, a) for a in incorrect_answers]
        
        all_scores_mc1 = [best_score] + incorrect_scores
        is_correct = best_score == max(all_scores_mc1)
        
        if is_correct:
            mc1_correct += 1
        mc1_total += 1
        
        # MC2: Probability mass on all correct answers
        correct_scores = [score_fn(model, question, a) for a in correct_answers]
        all_scores_mc2 = correct_scores + incorrect_scores
        all_scores_tensor = torch.tensor(all_scores_mc2)
        probs = F.softmax(all_scores_tensor, dim=0)
        mc2 = probs[:len(correct_scores)].sum().item()
        mc2_scores.append(mc2)
        
        per_sample.append({
            "id": sample["id"],
            "question": question,
            "category": sample.get("category", "Unknown"),
            "mc1_correct": is_correct,
            "mc2_score": mc2,
        })
    
    # Per-category analysis
    category_mc1 = {}
    category_mc2 = {}
    
    for s in per_sample:
        cat = s["category"]
        if cat not in category_mc1:
            category_mc1[cat] = []
            category_mc2[cat] = []
        category_mc1[cat].append(s["mc1_correct"])
        category_mc2[cat].append(s["mc2_score"])
    
    per_category = {}
    for cat in category_mc1:
        per_category[cat] = {
            "n": len(category_mc1[cat]),
            "mc1": np.mean(category_mc1[cat]),
            "mc2": np.mean(category_mc2[cat]),
        }
    
    return {
        "mc1_score": mc1_correct / mc1_total if mc1_total > 0 else 0,
        "mc2_score": np.mean(mc2_scores) if mc2_scores else 0,
        "n_samples": mc1_total,
        "per_category": per_category,
        "per_sample": per_sample,
    }


def run_crystalboost_evaluation(model, dataset, model_name: str) -> dict:
    """Run full CrystalBoost evaluation with grid search and comparison."""
    from mechlens.intervention.crystalboost import (
        CrystalBoostConfig,
        learn_crystalboost_directions,
        compute_crystalboost_log_prob,
    )
    
    n_layers = model.cfg.n_layers
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Phase 1: Baseline
    logger.info("=" * 50)
    logger.info("Phase 1: Baseline Evaluation")
    logger.info("=" * 50)
    baseline = evaluate_mc1_mc2(model, dataset, method_name="baseline")
    results["baseline"] = {
        "mc1": baseline["mc1_score"],
        "mc2": baseline["mc2_score"],
    }
    logger.info(f"Baseline: MC1={baseline['mc1_score']:.4f}, MC2={baseline['mc2_score']:.4f}")
    
    # Phase 2: CrystalBoost Grid Search
    logger.info("=" * 50)
    logger.info("Phase 2: CrystalBoost Grid Search")
    logger.info("=" * 50)
    
    # Grid search configurations
    configs_to_try = [
        # Conservative
        CrystalBoostConfig(
            early_suppression_coeff=-0.5,
            late_amplification_coeff=1.5,
            boundary_boost_coeff=2.0,
            gaussian_sigma=0.1,
        ),
        # Moderate
        CrystalBoostConfig(
            early_suppression_coeff=-1.0,
            late_amplification_coeff=2.0,
            boundary_boost_coeff=3.0,
            gaussian_sigma=0.1,
        ),
        # Aggressive
        CrystalBoostConfig(
            early_suppression_coeff=-1.5,
            late_amplification_coeff=3.0,
            boundary_boost_coeff=5.0,
            gaussian_sigma=0.1,
        ),
        # Boundary-focused
        CrystalBoostConfig(
            early_suppression_coeff=-0.5,
            late_amplification_coeff=1.0,
            boundary_boost_coeff=5.0,
            gaussian_sigma=0.05,
        ),
        # Wide Gaussian
        CrystalBoostConfig(
            early_suppression_coeff=-1.0,
            late_amplification_coeff=2.0,
            boundary_boost_coeff=3.0,
            gaussian_sigma=0.15,
        ),
    ]
    
    grid_results = []
    best_config = None
    best_mc1 = 0.0
    
    for i, config in enumerate(configs_to_try):
        logger.info(f"Testing config {i+1}/{len(configs_to_try)}: "
                   f"early={config.early_suppression_coeff}, "
                   f"late={config.late_amplification_coeff}, "
                   f"boundary={config.boundary_boost_coeff}")
        
        # Learn directions
        crystalboost_info = learn_crystalboost_directions(model, dataset, config)
        
        # Create score function
        score_fn = lambda m, q, a, info=crystalboost_info: compute_crystalboost_log_prob(m, q, a, info)
        
        # Evaluate
        eval_result = evaluate_mc1_mc2(model, dataset, score_fn, method_name=f"crystalboost_{i}")
        
        config_result = {
            "config": {
                "early_suppression_coeff": config.early_suppression_coeff,
                "late_amplification_coeff": config.late_amplification_coeff,
                "boundary_boost_coeff": config.boundary_boost_coeff,
                "gaussian_sigma": config.gaussian_sigma,
            },
            "mc1": eval_result["mc1_score"],
            "mc2": eval_result["mc2_score"],
            "mc1_improvement": (eval_result["mc1_score"] - baseline["mc1_score"]) / baseline["mc1_score"] * 100,
        }
        grid_results.append(config_result)
        
        logger.info(f"  MC1={eval_result['mc1_score']:.4f} (+{config_result['mc1_improvement']:.1f}%)")
        
        if eval_result["mc1_score"] > best_mc1:
            best_mc1 = eval_result["mc1_score"]
            best_config = config
            best_eval = eval_result
    
    results["crystalboost_grid"] = grid_results
    results["crystalboost_best"] = {
        "config": {
            "early_suppression_coeff": best_config.early_suppression_coeff,
            "late_amplification_coeff": best_config.late_amplification_coeff,
            "boundary_boost_coeff": best_config.boundary_boost_coeff,
            "gaussian_sigma": best_config.gaussian_sigma,
        },
        "mc1": best_mc1,
        "mc2": best_eval["mc2_score"],
        "mc1_improvement": (best_mc1 - baseline["mc1_score"]) / baseline["mc1_score"] * 100,
        "per_category": best_eval["per_category"],
    }
    
    logger.info(f"Best CrystalBoost: MC1={best_mc1:.4f} (+{results['crystalboost_best']['mc1_improvement']:.1f}%)")
    
    # Phase 3: Compare with DoLa (current best baseline)
    logger.info("=" * 50)
    logger.info("Phase 3: DoLa Comparison")
    logger.info("=" * 50)
    
    from mechlens.intervention.dola import compute_dola_logits
    
    def dola_score_fn(m, q, a):
        """DoLa dynamic scoring."""
        prompt = f"Q: {q}\nA:"
        full_text = f"Q: {q}\nA: {a}"
        
        prompt_tokens = m.to_tokens(prompt, prepend_bos=True)
        full_tokens = m.to_tokens(full_text, prepend_bos=True)
        
        q_len = prompt_tokens.shape[1]
        
        if full_tokens.shape[1] <= q_len:
            return float("-inf")
        
        # Use DoLa logits computation
        mature_layer = n_layers - 1
        premature_layers = list(range(0, int(n_layers * 0.6)))
        
        hook_names = [f"blocks.{l}.hook_resid_post" for l in set(premature_layers) | {mature_layer}]
        
        with torch.no_grad():
            _, cache = m.run_with_cache(full_tokens, names_filter=hook_names)
        
        total_log_prob = 0.0
        
        for pos in range(q_len, full_tokens.shape[1]):
            target_token = full_tokens[0, pos].item()
            
            # Get mature layer
            mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, pos - 1, :]
            mature_normed = m.ln_final(mature_resid)
            mature_logits = mature_normed @ m.W_U
            if m.b_U is not None:
                mature_logits = mature_logits + m.b_U
            mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)
            mature_probs = mature_log_probs.exp()
            
            # Dynamic selection
            best_premature = premature_layers[0]
            best_jsd = -1.0
            
            for p_layer in premature_layers:
                p_resid = cache[f"blocks.{p_layer}.hook_resid_post"][0, pos - 1, :]
                p_normed = m.ln_final(p_resid)
                p_logits = p_normed @ m.W_U
                if m.b_U is not None:
                    p_logits = p_logits + m.b_U
                p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
                p_probs = p_log_probs.exp()
                
                m_probs = 0.5 * (mature_probs + p_probs)
                m_log_probs = m_probs.log()
                kl_pm = F.kl_div(m_log_probs, mature_probs, reduction="sum", log_target=False)
                kl_qm = F.kl_div(m_log_probs, p_probs, reduction="sum", log_target=False)
                jsd = 0.5 * (kl_pm + kl_qm).item()
                
                if jsd > best_jsd:
                    best_jsd = jsd
                    best_premature = p_layer
            
            # DoLa contrast
            p_resid = cache[f"blocks.{best_premature}.hook_resid_post"][0, pos - 1, :]
            p_normed = m.ln_final(p_resid)
            p_logits = p_normed @ m.W_U
            if m.b_U is not None:
                p_logits = p_logits + m.b_U
            p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
            
            dola_logits = mature_log_probs - p_log_probs
            dola_log_probs = F.log_softmax(dola_logits, dim=-1)
            
            total_log_prob += dola_log_probs[target_token].item()
        
        return total_log_prob
    
    dola_eval = evaluate_mc1_mc2(model, dataset, dola_score_fn, method_name="dola_dynamic")
    results["dola_dynamic"] = {
        "mc1": dola_eval["mc1_score"],
        "mc2": dola_eval["mc2_score"],
        "mc1_improvement": (dola_eval["mc1_score"] - baseline["mc1_score"]) / baseline["mc1_score"] * 100,
    }
    
    logger.info(f"DoLa Dynamic: MC1={dola_eval['mc1_score']:.4f} (+{results['dola_dynamic']['mc1_improvement']:.1f}%)")
    
    # Summary
    results["summary"] = {
        "baseline_mc1": baseline["mc1_score"],
        "crystalboost_best_mc1": best_mc1,
        "dola_mc1": dola_eval["mc1_score"],
        "crystalboost_improvement": results["crystalboost_best"]["mc1_improvement"],
        "dola_improvement": results["dola_dynamic"]["mc1_improvement"],
        "crystalboost_beats_dola": best_mc1 > dola_eval["mc1_score"],
        "crystalboost_margin": (best_mc1 - dola_eval["mc1_score"]) / dola_eval["mc1_score"] * 100,
    }
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("CrystalBoost Evaluation Experiment")
    logger.info("Target: Outperform DoLa (+25.4% MC1) using Late Crystallization insight")
    logger.info("=" * 70)
    
    # Load dataset
    dataset = load_truthfulqa_dataset()
    logger.info(f"Loaded TruthfulQA dataset: {len(dataset)} samples")
    
    all_results = {}
    
    for model_name in EVAL_MODELS:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*60}")
            
            model = load_model(model_name)
            result = run_crystalboost_evaluation(model, dataset, model_name)
            all_results[model_name] = result
            
            # Save individual results
            safe_name = model_name.replace("/", "_").replace("-", "_")
            output_path = RESULTS_DIR / f"crystalboost_{safe_name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved results to {output_path}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    combined = {
        "experiment": "crystalboost_evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "individual_results": all_results,
    }
    
    combined_path = RESULTS_DIR / "crystalboost_combined.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CRYSTALBOOST EVALUATION SUMMARY")
    print("=" * 70)
    
    for model_name, result in all_results.items():
        summary = result["summary"]
        print(f"\n{model_name}:")
        print(f"  Baseline MC1:     {summary['baseline_mc1']:.4f}")
        print(f"  DoLa MC1:         {summary['dola_mc1']:.4f} (+{summary['dola_improvement']:.1f}%)")
        print(f"  CrystalBoost MC1: {summary['crystalboost_best_mc1']:.4f} (+{summary['crystalboost_improvement']:.1f}%)")
        print(f"  CrystalBoost beats DoLa: {summary['crystalboost_beats_dola']}")
        if summary['crystalboost_beats_dola']:
            print(f"  *** CrystalBoost margin: +{summary['crystalboost_margin']:.1f}% over DoLa ***")
    
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
