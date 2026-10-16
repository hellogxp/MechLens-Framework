#!/usr/bin/env python3
"""E1: CAA vs DoLa per-sample evaluation + bootstrap p-value.

Runs CAA and DoLa on Qwen2.5-7B TruthfulQA, saves per-question results,
then bootstraps (n=1000) to compute the p-value for the DoLa > CAA reversal.

Usage: python e1_caa_dola_bootstrap.py
Output: results/rebuttal_2026may/e1_caa_dola_bootstrap.json
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "rebuttal_2026may"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MECHLENS_MODEL_PATH", "Qwen/Qwen2.5-7B")
MODEL_NAME = "Qwen/Qwen2.5-7B"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e1")


# ==================== MODEL LOADING ====================

def load_model():
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {MODEL_NAME} from {MODEL_PATH}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
    )
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, hf_model=hf_model, tokenizer=tokenizer,
        torch_dtype=torch.float16, device="cuda",
    )
    logger.info(f"Loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    return model


def load_truthfulqa():
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")
    logger.info(f"Loaded {len(dataset)} TruthfulQA samples")
    return dataset


# ==================== MC1 EVALUATION (per-sample) ====================

def compute_answer_log_prob(model, question, answer):
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
    total = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total += log_probs[i - 1, token_id].item()
    return total


def evaluate_mc1_per_sample(model, dataset, score_fn, method_name):
    per_sample = []
    correct = 0
    total = 0
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  [{method_name}] {i}/{len(dataset)}")
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        if not best_answer or not incorrect_answers:
            continue
        best_score = score_fn(model, question, best_answer)
        incorrect_scores = [score_fn(model, question, a) for a in incorrect_answers]
        all_scores = [best_score] + incorrect_scores
        is_correct = best_score == max(all_scores)
        if is_correct:
            correct += 1
        total += 1
        per_sample.append({
            "sample_idx": i,
            "question": question,
            "is_correct": is_correct,
            "best_score": best_score,
            "max_incorrect_score": max(incorrect_scores) if incorrect_scores else None,
        })
    mc1 = correct / total if total > 0 else 0
    logger.info(f"  [{method_name}] MC1: {mc1:.4f} ({correct}/{total})")
    return {"mc1_score": mc1, "n_correct": correct, "n_total": total, "per_sample": per_sample}


# ==================== DOLA ====================

def unembed_at_layer(model, resid):
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def compute_dola_log_prob(model, question, answer, premature_layers=None):
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
    all_layers = set(premature_layers) | {mature_layer}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]
    with torch.no_grad():
        _, cache = model.run_with_cache(full_tokens, names_filter=hook_names)
    total = 0.0
    for pos in range(q_len, full_tokens.shape[1]):
        target_token = full_tokens[0, pos].item()
        mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, pos - 1, :]
        mature_logits = unembed_at_layer(model, mature_resid)
        mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)
        mature_probs = mature_log_probs.exp()
        best_premature = premature_layers[0]
        best_jsd = -1.0
        for p_layer in premature_layers:
            p_resid = cache[f"blocks.{p_layer}.hook_resid_post"][0, pos - 1, :]
            p_logits = unembed_at_layer(model, p_resid)
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
        p_resid = cache[f"blocks.{best_premature}.hook_resid_post"][0, pos - 1, :]
        p_logits = unembed_at_layer(model, p_resid)
        p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
        dola_logits = mature_log_probs - p_log_probs
        dola_log_probs = F.log_softmax(dola_logits, dim=-1)
        total += dola_log_probs[target_token].item()
    return total


# ==================== CAA ====================

def learn_caa_directions(model, dataset, top_k_layers=10):
    n_layers = model.cfg.n_layers
    target_layers = list(range(n_layers - top_k_layers, n_layers))
    correct_acts = {l: [] for l in target_layers}
    incorrect_acts = {l: [] for l in target_layers}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in target_layers]
    for sample in dataset[:100]:
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        if not best_answer or not incorrect_answers:
            continue
        correct_text = f"Q: {question}\nA: {best_answer}"
        correct_tokens = model.to_tokens(correct_text, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(correct_tokens, names_filter=hook_names)
        for l in target_layers:
            correct_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu())
        incorrect_text = f"Q: {question}\nA: {incorrect_answers[0]}"
        incorrect_tokens = model.to_tokens(incorrect_text, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(incorrect_tokens, names_filter=hook_names)
        for l in target_layers:
            incorrect_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu())
    directions = {}
    for l in target_layers:
        if correct_acts[l] and incorrect_acts[l]:
            correct_mean = torch.stack(correct_acts[l]).mean(dim=0)
            incorrect_mean = torch.stack(incorrect_acts[l]).mean(dim=0)
            direction = correct_mean - incorrect_mean
            direction = direction / direction.norm()
            directions[l] = direction
    return {"target_layers": target_layers, "directions": directions}


def compute_caa_log_prob(model, question, answer, caa_info, coeff=5.0):
    directions = caa_info["directions"]
    target_layers = caa_info["target_layers"]
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    q_len = prompt_tokens.shape[1]
    if full_tokens.shape[1] <= q_len:
        return float("-inf")

    def make_caa_hook(layer_idx):
        direction = directions[layer_idx].to(model.cfg.device, model.cfg.dtype)
        def hook_fn(activation, hook):
            activation[:, :, :] = activation + (coeff * direction).unsqueeze(0).unsqueeze(0)
            return activation
        return hook_fn

    hooks = [(f"blocks.{l}.hook_resid_post", make_caa_hook(l)) for l in target_layers if l in directions]
    with torch.no_grad():
        logits = model.run_with_hooks(full_tokens, fwd_hooks=hooks)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total += log_probs[i - 1, token_id].item()
    return total


# ==================== BOOTSTRAP ====================

def bootstrap_p_value(caa_results, dola_results, n_bootstrap=1000):
    caa_correct = np.array([s["is_correct"] for s in caa_results["per_sample"]])
    dola_correct = np.array([s["is_correct"] for s in dola_results["per_sample"]])
    n = len(caa_correct)
    assert n == len(dola_correct), f"Mismatch: {n} vs {len(dola_correct)}"
    caa_rate = caa_correct.mean()
    dola_rate = dola_correct.mean()
    observed_diff = dola_rate - caa_rate
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        diff = dola_correct[idx].mean() - caa_correct[idx].mean()
        boot_diffs.append(diff)
    boot_diffs = np.array(boot_diffs)
    p_value = (boot_diffs <= 0).mean() * 2
    p_value = min(p_value, 1.0)
    ci = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        "caa_mc1": float(caa_rate),
        "dola_mc1": float(dola_rate),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95": [float(ci[0]), float(ci[1])],
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


# ==================== MAIN ====================

def main():
    logger.info("=" * 60)
    logger.info("E1: CAA vs DoLa per-sample + bootstrap")
    logger.info("=" * 60)

    dataset = load_truthfulqa()
    model = load_model()

    # Baseline
    logger.info("Phase 1: Baseline")
    baseline = evaluate_mc1_per_sample(
        model, dataset,
        lambda m, q, a: compute_answer_log_prob(m, q, a),
        "baseline"
    )

    # DoLa
    logger.info("Phase 2: DoLa")
    dola = evaluate_mc1_per_sample(
        model, dataset,
        lambda m, q, a: compute_dola_log_prob(m, q, a),
        "dola"
    )

    # CAA
    logger.info("Phase 3: CAA (learning directions)")
    caa_info = learn_caa_directions(model, dataset, top_k_layers=10)
    logger.info("Phase 3b: CAA evaluation")
    caa = evaluate_mc1_per_sample(
        model, dataset,
        lambda m, q, a: compute_caa_log_prob(m, q, a, caa_info, coeff=5.0),
        "caa"
    )

    # Bootstrap
    logger.info("Phase 4: Bootstrap p-value")
    bootstrap = bootstrap_p_value(caa, dola, n_bootstrap=1000)

    # Summary
    results = {
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_mc1": baseline["mc1_score"],
        "dola_mc1": dola["mc1_score"],
        "caa_mc1": caa["mc1_score"],
        "dola_improvement_pct": (dola["mc1_score"] - baseline["mc1_score"]) / baseline["mc1_score"] * 100,
        "caa_improvement_pct": (caa["mc1_score"] - baseline["mc1_score"]) / baseline["mc1_score"] * 100,
        "bootstrap": bootstrap,
        "baseline_per_sample": baseline["per_sample"],
        "dola_per_sample": dola["per_sample"],
        "caa_per_sample": caa["per_sample"],
    }

    output_path = RESULTS_DIR / "e1_caa_dola_bootstrap.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("E1 SUMMARY: CAA vs DoLa Bootstrap")
    print("=" * 60)
    print(f"Baseline MC1: {baseline['mc1_score']:.4f}")
    print(f"DoLa MC1:     {dola['mc1_score']:.4f} (+{results['dola_improvement_pct']:.1f}%)")
    print(f"CAA MC1:      {caa['mc1_score']:.4f} (+{results['caa_improvement_pct']:.1f}%)")
    print(f"Observed diff (DoLa-CAA): {bootstrap['observed_diff']:.4f}")
    print(f"Bootstrap p-value:        {bootstrap['p_value']:.4f}")
    print(f"95% CI: [{bootstrap['ci_95'][0]:.4f}, {bootstrap['ci_95'][1]:.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
