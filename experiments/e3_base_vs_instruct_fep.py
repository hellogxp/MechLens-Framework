#!/usr/bin/env python3
"""E3: Base vs Instruct FEP comparison.

Runs FEP detection on Qwen2.5-7B base and Qwen2.5-7B-Instruct,
compares crystallization rates to show post-training effect.

Usage: python e3_base_vs_instruct_fep.py
Output: results/rebuttal_2026may/e3_base_vs_instruct_fep.json
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "rebuttal_2026may"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_PATH = os.environ.get("MECHLENS_BASE_PATH", "Qwen/Qwen2.5-7B")
INSTRUCT_PATH = os.environ.get("MECHLENS_INSTRUCT_PATH", "Qwen/Qwen2.5-7B-Instruct")
TOP_K = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e3")


# ==================== MODEL LOADING (HuggingFace) ====================

def load_hf_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {n_layers} layers")
    return model, tokenizer


def load_truthfulqa():
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")
    logger.info(f"Loaded {len(dataset)} TruthfulQA samples")
    return dataset


# ==================== FEP DETECTION ====================

def get_model_components(model):
    norm_module = model.model.norm
    lm_head_module = model.lm_head
    return norm_module, lm_head_module


def unembed_at_layer(norm_module, lm_head_module, hidden_state):
    target_device = next(norm_module.parameters()).device
    h = hidden_state.to(target_device).to(torch.float16)
    normed = norm_module(h)
    head_device = next(lm_head_module.parameters()).device
    normed = normed.to(head_device)
    logits = lm_head_module(normed)
    return logits


def detect_fep_for_sample(model, tokenizer, norm_module, lm_head_module, question, correct_answer, top_k=TOP_K):
    n_layers = model.config.num_hidden_layers
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    answer_tokens = tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(answer_tokens) == 0:
        return None
    target_token = answer_tokens[0]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    layer_ranks = []
    fep_layer = None

    for l in range(n_layers):
        h = hidden_states[l + 1][0, -1, :]
        logits = unembed_at_layer(norm_module, lm_head_module, h)
        sorted_indices = logits.argsort(descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank_val = rank[0].item() if len(rank) > 0 else logits.shape[-1]
        layer_ranks.append(rank_val)
        in_topk = rank_val < top_k
        if in_topk and fep_layer is None:
            fep_layer = l

    if fep_layer is None:
        fep_layer = n_layers

    return {
        "fep_layer": fep_layer,
        "n_layers": n_layers,
        "fep_depth": fep_layer / n_layers,
        "late_crystallization": fep_layer == n_layers,
    }


def run_fep_for_model(model, tokenizer, dataset, model_name):
    norm_module, lm_head_module = get_model_components(model)
    results = []
    for idx, sample in enumerate(dataset):
        if idx % 100 == 0:
            logger.info(f"  [{model_name}] {idx}/{len(dataset)}")
        correct_answers = sample.get("correct_answers", [])
        if not correct_answers:
            correct_answers = [sample.get("best_answer", "")]
        if not correct_answers[0]:
            continue
        torch.cuda.empty_cache()
        try:
            result = detect_fep_for_sample(
                model, tokenizer, norm_module, lm_head_module,
                sample["question"], correct_answers[0]
            )
            if result:
                result["sample_idx"] = idx
                results.append(result)
        except Exception as e:
            logger.warning(f"  Sample {idx} error: {e}")
            continue

    if not results:
        return {"model_name": model_name, "error": "no_valid_results"}

    fep_layers = [r["fep_layer"] for r in results]
    n_layers = results[0]["n_layers"]
    late_count = sum(1 for r in results if r["late_crystallization"])
    fep_depths = [r["fep_depth"] for r in results]

    return {
        "model_name": model_name,
        "n_layers": n_layers,
        "n_samples": len(results),
        "late_crystallization_pct": round(late_count / len(results) * 100, 1),
        "mean_fep_depth": round(np.mean(fep_depths) * 100, 1),
        "fep_distribution": {
            "early_(<50%)": sum(1 for d in fep_depths if d < 0.5),
            "mid_(50-80%)": sum(1 for d in fep_depths if 0.5 <= d < 0.8),
            "late_(>80%)": sum(1 for d in fep_depths if d >= 0.8),
        },
        "per_sample_results": results,
    }


# ==================== MAIN ====================

def main():
    logger.info("=" * 60)
    logger.info("E3: Base vs Instruct FEP comparison")
    logger.info("=" * 60)

    dataset = load_truthfulqa()

    # Base model
    logger.info("--- Base Model ---")
    base_model, base_tokenizer = load_hf_model(BASE_PATH)
    base_results = run_fep_for_model(base_model, base_tokenizer, dataset, "Qwen2.5-7B-base")
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # Instruct model
    logger.info("--- Instruct Model ---")
    inst_model, inst_tokenizer = load_hf_model(INSTRUCT_PATH)
    inst_results = run_fep_for_model(inst_model, inst_tokenizer, dataset, "Qwen2.5-7B-Instruct")
    del inst_model, inst_tokenizer
    torch.cuda.empty_cache()

    # Comparison
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base": {k: v for k, v in base_results.items() if k != "per_sample_results"},
        "instruct": {k: v for k, v in inst_results.items() if k != "per_sample_results"},
        "delta": {
            "late_crystallization_pct": (
                inst_results["late_crystallization_pct"] - base_results["late_crystallization_pct"]
                if "late_crystallization_pct" in inst_results and "late_crystallization_pct" in base_results
                else None
            ),
            "mean_fep_depth": (
                inst_results["mean_fep_depth"] - base_results["mean_fep_depth"]
                if "mean_fep_depth" in inst_results and "mean_fep_depth" in base_results
                else None
            ),
        },
        "base_per_sample": base_results.get("per_sample_results", []),
        "instruct_per_sample": inst_results.get("per_sample_results", []),
    }

    output_path = RESULTS_DIR / "e3_base_vs_instruct_fep.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("E3 SUMMARY: Base vs Instruct FEP")
    print("=" * 60)
    if "error" not in base_results:
        print(f"Base:     Late crystal = {base_results['late_crystallization_pct']}%, FEP depth = {base_results['mean_fep_depth']}%")
    if "error" not in inst_results:
        print(f"Instruct: Late crystal = {inst_results['late_crystallization_pct']}%, FEP depth = {inst_results['mean_fep_depth']}%")
    if comparison["delta"]["late_crystallization_pct"] is not None:
        print(f"Delta:    Late crystal = {comparison['delta']['late_crystallization_pct']:+.1f}%, FEP depth = {comparison['delta']['mean_fep_depth']:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
