"""Instruct Model Intervention Comparison.

Tests whether CAA outperforms DoLa on instruction-tuned models,
validating the FEP theory prediction: lower crystallization favors
activation-space methods over logit-space methods.

GPU time: ~3 hours on A100 80GB
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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
logger = logging.getLogger("instruct_intervention")

RESULTS_DIR = PROJECT_ROOT / "results" / "instruct_intervention"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = {
    "name": "Qwen/Qwen2.5-7B-Instruct",
    "local_path": "/root/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct",
    "n_layers": 28,
}

# Base model results for comparison
BASE_RESULTS = {
    "mc1_baseline": 0.2215,
    "mc1_dola": 0.2778,  # +25.4%
    "mc1_caa": 0.2558,   # +15.5%
    "late_crystal_pct": 0.859,
}

# Instruct FEP results (from pilot)
INSTRUCT_FEP = {
    "late_crystal_pct": 0.373,
    "mean_fep": 25.5,
}


def load_model():
    """Load Qwen-Instruct as HookedTransformer."""
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = MODEL_CONFIG["name"]
    local_path = MODEL_CONFIG.get("local_path")

    if local_path and os.path.isdir(local_path):
        logger.info(f"Loading from local: {local_path}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_path, torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    else:
        logger.info(f"Loading via HuggingFace: {model_name}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = HookedTransformer.from_pretrained(
        model_name, hf_model=hf_model, tokenizer=tokenizer,
        torch_dtype=torch.float16, device="cuda",
    )
    logger.info(f"Loaded: {model.cfg.n_layers}L, {model.cfg.n_heads}H")
    return model


def format_prompt_instruct(question: str) -> str:
    """Format prompt for instruct model."""
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ======================== MC1 Evaluation ========================

def evaluate_mc1_baseline(model, dataset: list) -> dict:
    """Evaluate baseline MC1."""
    correct = 0
    total = 0

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Baseline MC1: {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        best_score = float("-inf")
        best_idx = 0
        prompt = format_prompt_instruct(question)

        for j, choice in enumerate(choices):
            full_text = prompt + " " + choice
            tokens = model.to_tokens(full_text, prepend_bos=True)
            if tokens.shape[1] > 256:
                tokens = tokens[:, :256]

            with torch.no_grad():
                logits = model(tokens)

            prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
            answer_start = prompt_tokens.shape[1]
            if answer_start >= tokens.shape[1]:
                continue

            log_probs = F.log_softmax(logits[0, answer_start - 1:-1], dim=-1)
            answer_token_ids = tokens[0, answer_start:]
            token_log_probs = log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)
            score = token_log_probs.sum().item()

            if score > best_score:
                best_score = score
                best_idx = j

        if labels[best_idx] == 1:
            correct += 1
        total += 1

    mc1 = correct / max(total, 1)
    return {"mc1": mc1, "correct": correct, "total": total}


# ======================== DoLa Evaluation ========================

def dola_decode_instruct(model, prompt: str, mature_layer: int = None) -> torch.Tensor:
    """DoLa decoding for instruct model."""
    if mature_layer is None:
        mature_layer = model.cfg.n_layers - 1

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 256:
        tokens = tokens[:, :256]

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    # Get mature layer logits
    mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, -1, :]
    normed = model.ln_final(mature_resid)
    mature_logits = normed @ model.W_U
    if model.b_U is not None:
        mature_logits = mature_logits + model.b_U
    mature_probs = F.softmax(mature_logits.float(), dim=-1)

    # Dynamic premature layer selection
    best_jsd = -1
    best_premature = 0

    for l in range(min(16, model.cfg.n_layers - 1)):
        resid = cache[f"blocks.{l}.hook_resid_post"][0, -1, :]
        normed = model.ln_final(resid)
        logits = normed @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U
        probs = F.softmax(logits.float(), dim=-1)

        # JSD
        m = 0.5 * (mature_probs + probs)
        jsd = 0.5 * (
            F.kl_div(m.log(), mature_probs, reduction='sum') +
            F.kl_div(m.log(), probs, reduction='sum')
        )

        if jsd > best_jsd:
            best_jsd = jsd
            best_premature = l

    # Get premature logits
    premature_resid = cache[f"blocks.{best_premature}.hook_resid_post"][0, -1, :]
    normed = model.ln_final(premature_resid)
    premature_logits = normed @ model.W_U
    if model.b_U is not None:
        premature_logits = premature_logits + model.b_U

    # Contrast
    dola_logits = mature_logits - premature_logits
    return dola_logits


def evaluate_mc1_dola(model, dataset: list) -> dict:
    """Evaluate MC1 with DoLa."""
    correct = 0
    total = 0

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  DoLa MC1: {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        best_score = float("-inf")
        best_idx = 0
        base_prompt = format_prompt_instruct(question)

        for j, choice in enumerate(choices):
            full_prompt = base_prompt + " " + choice
            dola_logits = dola_decode_instruct(model, full_prompt)

            # Score = sum of log-probs for answer tokens
            answer_tokens = model.to_tokens(choice, prepend_bos=False)[0]
            if len(answer_tokens) == 0:
                continue

            log_probs = F.log_softmax(dola_logits, dim=-1)
            score = sum(log_probs[t].item() for t in answer_tokens[:5])

            if score > best_score:
                best_score = score
                best_idx = j

        if labels[best_idx] == 1:
            correct += 1
        total += 1

    mc1 = correct / max(total, 1)
    return {"mc1": mc1, "correct": correct, "total": total}


# ======================== CAA Evaluation ========================

def extract_caa_directions(model, dataset: list, n_pairs: int = 256) -> dict:
    """Extract CAA steering directions."""
    logger.info(f"Extracting CAA directions from {n_pairs} pairs...")

    directions = {l: [] for l in range(model.cfg.n_layers)}
    count = 0

    for sample in dataset:
        if count >= n_pairs:
            break

        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        correct_idx = labels.index(1) if 1 in labels else None
        if correct_idx is None:
            continue

        incorrect_indices = [i for i, l in enumerate(labels) if l == 0]
        if not incorrect_indices:
            continue

        question = sample["question"]
        correct_choice = choices[correct_idx]
        incorrect_choice = choices[incorrect_indices[0]]

        prompt_correct = format_prompt_instruct(question) + " " + correct_choice
        prompt_incorrect = format_prompt_instruct(question) + " " + incorrect_choice

        tokens_c = model.to_tokens(prompt_correct, prepend_bos=True)
        tokens_i = model.to_tokens(prompt_incorrect, prepend_bos=True)

        if tokens_c.shape[1] > 256:
            tokens_c = tokens_c[:, :256]
        if tokens_i.shape[1] > 256:
            tokens_i = tokens_i[:, :256]

        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

        with torch.no_grad():
            _, cache_c = model.run_with_cache(tokens_c, names_filter=hook_names)
            _, cache_i = model.run_with_cache(tokens_i, names_filter=hook_names)

        for l in range(model.cfg.n_layers):
            act_c = cache_c[f"blocks.{l}.hook_resid_post"][0, -1, :].float()
            act_i = cache_i[f"blocks.{l}.hook_resid_post"][0, -1, :].float()
            directions[l].append(act_c - act_i)

        count += 1

    # Average directions
    avg_directions = {}
    for l in range(model.cfg.n_layers):
        if directions[l]:
            stacked = torch.stack(directions[l])
            avg_directions[l] = stacked.mean(dim=0)

    # Compute layer importance by norm
    norms = {l: avg_directions[l].norm().item() for l in avg_directions}
    max_norm = max(norms.values())
    importance = {l: norms[l] / max_norm for l in norms}

    return {"directions": avg_directions, "importance": importance}


def evaluate_mc1_caa(model, dataset: list, caa_data: dict, top_k: int = 10, coeff: float = 5.0) -> dict:
    """Evaluate MC1 with CAA steering."""
    directions = caa_data["directions"]
    importance = caa_data["importance"]

    # Select top-k layers
    sorted_layers = sorted(importance.keys(), key=lambda l: importance[l], reverse=True)
    selected_layers = sorted_layers[:top_k]
    logger.info(f"CAA selected layers: {selected_layers}")

    correct = 0
    total = 0

    # Create hooks
    def make_hook(layer_dir, coefficient):
        def hook_fn(act, hook):
            act[:, -1, :] = act[:, -1, :] + coefficient * layer_dir.to(act.device).half()
            return act
        return hook_fn

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  CAA MC1: {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        best_score = float("-inf")
        best_idx = 0
        base_prompt = format_prompt_instruct(question)

        for j, choice in enumerate(choices):
            full_text = base_prompt + " " + choice
            tokens = model.to_tokens(full_text, prepend_bos=True)
            if tokens.shape[1] > 256:
                tokens = tokens[:, :256]

            # Add hooks
            hooks = []
            for l in selected_layers:
                hook_name = f"blocks.{l}.hook_resid_post"
                hooks.append((hook_name, make_hook(directions[l], coeff)))

            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            prompt_tokens = model.to_tokens(base_prompt, prepend_bos=True)
            answer_start = prompt_tokens.shape[1]
            if answer_start >= tokens.shape[1]:
                continue

            log_probs = F.log_softmax(logits[0, answer_start - 1:-1], dim=-1)
            answer_token_ids = tokens[0, answer_start:]
            token_log_probs = log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)
            score = token_log_probs.sum().item()

            if score > best_score:
                best_score = score
                best_idx = j

        if labels[best_idx] == 1:
            correct += 1
        total += 1

    mc1 = correct / max(total, 1)
    return {"mc1": mc1, "correct": correct, "total": total, "top_k": top_k, "coeff": coeff}


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("INSTRUCT MODEL INTERVENTION COMPARISON")
    logger.info(f"Model: {MODEL_CONFIG['name']}")
    logger.info("Testing: Does CAA > DoLa on lower-crystallization instruct model?")
    logger.info("=" * 70)

    # Load model
    model = load_model()

    # Load TruthfulQA
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    logger.info(f"Loaded {len(dataset)} samples")

    results = {
        "model": MODEL_CONFIG["name"],
        "n_samples": len(dataset),
        "instruct_fep": INSTRUCT_FEP,
        "base_results": BASE_RESULTS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. Baseline
    logger.info("\n[1/3] Evaluating baseline MC1...")
    t0 = time.time()
    baseline_result = evaluate_mc1_baseline(model, dataset)
    results["baseline"] = baseline_result
    results["baseline"]["time_s"] = time.time() - t0
    logger.info(f"Baseline MC1: {baseline_result['mc1']:.4f}")

    # 2. DoLa
    logger.info("\n[2/3] Evaluating DoLa dynamic...")
    t0 = time.time()
    dola_result = evaluate_mc1_dola(model, dataset)
    results["dola"] = dola_result
    results["dola"]["time_s"] = time.time() - t0
    logger.info(f"DoLa MC1: {dola_result['mc1']:.4f}")

    # 3. CAA
    logger.info("\n[3/3] Evaluating CAA (top_k=10, coeff=5.0)...")
    t0 = time.time()
    caa_data = extract_caa_directions(model, dataset)
    caa_result = evaluate_mc1_caa(model, dataset, caa_data)
    results["caa"] = caa_result
    results["caa"]["time_s"] = time.time() - t0
    logger.info(f"CAA MC1: {caa_result['mc1']:.4f}")

    # Analysis
    baseline_mc1 = baseline_result["mc1"]
    dola_mc1 = dola_result["mc1"]
    caa_mc1 = caa_result["mc1"]

    dola_delta = (dola_mc1 - baseline_mc1) / baseline_mc1 * 100
    caa_delta = (caa_mc1 - baseline_mc1) / baseline_mc1 * 100

    results["analysis"] = {
        "baseline_mc1": baseline_mc1,
        "dola_mc1": dola_mc1,
        "caa_mc1": caa_mc1,
        "dola_delta_pct": dola_delta,
        "caa_delta_pct": caa_delta,
        "caa_beats_dola": caa_mc1 > dola_mc1,
        "theory_validated": caa_mc1 > dola_mc1,  # FEP theory predicts CAA > DoLa on low-crystallization
    }

    # Save
    output_path = RESULTS_DIR / "instruct_intervention_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"\nSaved to {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("INSTRUCT MODEL INTERVENTION COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Crystallization: {INSTRUCT_FEP['late_crystal_pct']:.1%} (vs base {BASE_RESULTS['late_crystal_pct']:.1%})")
    print(f"\nResults:")
    print(f"  Baseline MC1: {baseline_mc1:.4f}")
    print(f"  DoLa MC1:     {dola_mc1:.4f} ({dola_delta:+.1f}%)")
    print(f"  CAA MC1:      {caa_mc1:.4f} ({caa_delta:+.1f}%)")
    print(f"\nFEP Theory Prediction: CAA > DoLa on low-crystallization model")
    print(f"Result: {'VALIDATED' if caa_mc1 > dola_mc1 else 'NOT VALIDATED'}")
    print(f"\nComparison with base model (high crystallization 85.9%):")
    print(f"  Base: DoLa +25.4% > CAA +15.5%")
    print(f"  Instruct: DoLa {dola_delta:+.1f}% vs CAA {caa_delta:+.1f}%")

    del model
    torch.cuda.empty_cache()
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
