#!/usr/bin/env python3
"""E2: Causal Tracing + FEP comparison on TruthfulQA.

For 50 TruthfulQA questions, computes:
1. FEP (when answer becomes top-10 decodable via logit lens)
2. Causal tracing (which layers' MLP outputs have causal effect)

Key output: causal peak layer vs FEP layer — showing they are complementary
(causal involvement early, vocabulary-space decodability late).

Usage: python e2_causal_tracing_fep.py
Output: results/qwen_7b/e2_causal_tracing_fep.json
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

RESULTS_DIR = PROJECT_ROOT / "results" / "qwen_7b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MECHLENS_MODEL_PATH", "Qwen/Qwen2.5-7B")
MODEL_NAME = "Qwen/Qwen2.5-7B"
N_SAMPLES = 50
TOP_K = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2")


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


# ==================== FEP DETECTION (via logit lens) ====================

def detect_fep(model, question, correct_answer, top_k=TOP_K):
    """Detect FEP using TransformerLens logit lens at each layer."""
    n_layers = model.cfg.n_layers
    prompt = f"Q: {question}\nA:"
    tokens = model.to_tokens(prompt, prepend_bos=True)

    answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)
    target_token_id = answer_tokens[0, 0].item() if answer_tokens.shape[1] > 0 else None
    if target_token_id is None:
        return None

    target_idx = tokens.shape[1] - 1

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    layer_ranks = []
    fep_layer = None
    for l in range(n_layers):
        resid = cache[f"blocks.{l}.hook_resid_post"][0, target_idx, :]
        normed = model.ln_final(resid.unsqueeze(0))
        logits = normed @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U
        logits = logits[0].float()
        sorted_indices = logits.argsort(descending=True)
        rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0]
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
        "layer_ranks": layer_ranks,
        "late_crystallization": fep_layer == n_layers,
    }


# ==================== CAUSAL TRACING ====================

def run_causal_tracing(model, question, correct_answer, n_runs=3):
    """Run causal tracing: corrupt question tokens, patch each MLP layer."""
    from mechlens.analysis.causal_tracing_v2 import run_causal_tracing_v2

    input_text = f"Q: {question}\nA: {correct_answer}"
    # Use the question text as the "subject" for corruption
    subject = question

    try:
        result = run_causal_tracing_v2(
            model=model,
            input_text=input_text,
            subject=subject,
            component_type="mlp",
            noise_factor=3.0,
            n_runs=n_runs,
            use_kl=True,
        )
        scores = result.patch_results.tolist()
        peak_layer = int(np.argmax(scores))
        max_recovery = float(max(scores))
        return {
            "scores": scores,
            "peak_layer": peak_layer,
            "max_recovery": max_recovery,
            "base_output": result.base_output,
            "corrupted_output": result.corrupted_output,
        }
    except Exception as e:
        logger.warning(f"  Causal tracing failed: {e}")
        return None


# ==================== MAIN ====================

def main():
    logger.info("=" * 60)
    logger.info(f"E2: Causal Tracing + FEP on {N_SAMPLES} TruthfulQA questions")
    logger.info("=" * 60)

    dataset = load_truthfulqa()
    model = load_model()
    n_layers = model.cfg.n_layers

    # Select samples: use first N_SAMPLES that have best_answer
    selected = []
    for idx, sample in enumerate(dataset):
        best = sample.get("best_answer", "")
        if best and len(selected) < N_SAMPLES:
            selected.append((idx, sample))
    logger.info(f"Selected {len(selected)} samples")

    results = []
    for i, (idx, sample) in enumerate(selected):
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        if i % 10 == 0:
            logger.info(f"  Processing {i}/{len(selected)}: {question[:50]}...")

        torch.cuda.empty_cache()

        # FEP
        fep_result = detect_fep(model, question, best_answer)
        if fep_result is None:
            continue

        # Causal tracing
        ct_result = run_causal_tracing(model, question, best_answer, n_runs=3)
        if ct_result is None:
            continue

        results.append({
            "sample_idx": idx,
            "question": question,
            "best_answer": best_answer,
            "fep_layer": fep_result["fep_layer"],
            "fep_depth": fep_result["fep_depth"],
            "late_crystallization": fep_result["late_crystallization"],
            "causal_peak_layer": ct_result["peak_layer"],
            "causal_max_recovery": ct_result["max_recovery"],
            "causal_scores": ct_result["scores"],
            "base_output": ct_result["base_output"],
            "corrupted_output": ct_result["corrupted_output"],
        })

    # Summary statistics
    fep_layers = [r["fep_layer"] for r in results]
    peak_layers = [r["causal_peak_layer"] for r in results]
    fep_depths = [r["fep_depth"] for r in results]
    peak_depths = [pl / n_layers for pl in peak_layers]

    summary = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_samples": len(results),
        "mean_fep_layer": float(np.mean(fep_layers)),
        "mean_causal_peak_layer": float(np.mean(peak_layers)),
        "mean_fep_depth": float(np.mean(fep_depths)),
        "mean_causal_peak_depth": float(np.mean(peak_depths)),
        "fep_minus_peak": float(np.mean(fep_layers) - np.mean(peak_layers)),
        "late_crystal_count": sum(1 for r in results if r["late_crystallization"]),
        "early_causal_count": sum(1 for r in results if r["causal_peak_layer"] < n_layers * 0.5),
        "complementary_count": sum(
            1 for r in results
            if r["causal_peak_layer"] < n_layers * 0.5
            and r["fep_layer"] >= n_layers * 0.8
        ),
    }

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "per_sample": results,
    }

    output_path = RESULTS_DIR / "e2_causal_tracing_fep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("E2 SUMMARY: Causal Tracing vs FEP")
    print("=" * 60)
    print(f"Samples: {len(results)}")
    print(f"Mean FEP layer:          {summary['mean_fep_layer']:.1f} / {n_layers} (depth {summary['mean_fep_depth']:.1%})")
    print(f"Mean causal peak layer:  {summary['mean_causal_peak_layer']:.1f} / {n_layers} (depth {summary['mean_causal_peak_depth']:.1%})")
    print(f"FEP - peak gap:          {summary['fep_minus_peak']:.1f} layers")
    print(f"Late crystallization:    {summary['late_crystal_count']}/{len(results)}")
    print(f"Early causal peak:       {summary['early_causal_count']}/{len(results)}")
    print(f"Complementary (early+late): {summary['complementary_count']}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
