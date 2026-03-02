"""Instruction-Tuned Model FEP Pilot Experiment.

Tests whether Late Crystallization persists in instruction-tuned (chat) models.
This addresses the reviewer concern: "Results may not generalize to
instruction-tuned or RLHF-aligned models."

Uses Qwen2.5-7B-Instruct as primary target (same architecture as base model,
enabling direct comparison of FEP distributions).

GPU time: ~2 hours on A100 40GB

Usage:
    python experiments/run_instruct_model_pilot.py [--max-samples 817]
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
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
logger = logging.getLogger("instruct_pilot")

RESULTS_DIR = PROJECT_ROOT / "results" / "instruct_pilot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Instruct model configurations
INSTRUCT_MODELS = {
    "qwen_instruct": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "local_path": "/root/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct",
        "n_layers": 28,
        "base_model_key": "qwen",
        "chat_template": True,
    },
}

# TruthfulQA baselines from base models (from paper)
BASE_MODEL_BASELINES = {
    "qwen": {
        "mean_fep": 27.3,
        "std_fep": 1.8,
        "late_crystal_pct": 0.859,
        "mc1_baseline": 0.2215,
        "mc1_dola": 0.2778,
        "mc1_caa": 0.2558,
    },
}


# ======================== Model Loading ========================

def load_model_hooked(model_config: dict):
    """Load instruct model as HookedTransformer."""
    model_name = model_config["name"]
    local_path = model_config.get("local_path")

    from transformer_lens import HookedTransformer

    if local_path and os.path.isdir(local_path):
        logger.info(f"Loading from local: {local_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_path, torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, tokenizer=tokenizer,
            torch_dtype=torch.float16, device="cuda",
        )
    else:
        logger.info(f"Loading via HuggingFace: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
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


# ======================== FEP Detection ========================

def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def detect_fep_instruct(
    model,
    question: str,
    correct_answer: str,
    use_chat_template: bool = True,
    top_k: int = 10,
) -> dict:
    """Detect FEP for an instruct model sample.

    Uses chat template formatting when available.
    """
    n_layers = model.cfg.n_layers

    # Format prompt
    if use_chat_template:
        # Standard chat format for instruct models
        prompt = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt = f"Q: {question}\nA:"

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 256:
        tokens = tokens[:, :256]

    answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)[0]
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0].item()

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    layer_ranks = []
    layer_probs = []
    layer_in_topk = []

    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        logits = unembed_at_layer(model, resid)
        probs = F.softmax(logits.float(), dim=-1)

        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank = rank[0].item() if len(rank) > 0 else probs.shape[0]

        prob = probs[target_token].item()
        layer_ranks.append(rank)
        layer_probs.append(prob)
        layer_in_topk.append(rank < top_k)

    fep_layer = n_layers
    for layer in range(n_layers):
        if layer_in_topk[layer]:
            fep_layer = layer
            break

    return {
        "fep_layer": fep_layer,
        "layer_ranks": layer_ranks,
        "layer_probs": layer_probs,
        "final_rank": layer_ranks[-1],
        "final_prob": layer_probs[-1],
        "target_token_str": model.to_single_str_token(target_token),
    }


# ======================== MC1/MC2 Evaluation ========================

def evaluate_mc1_instruct(
    model,
    dataset: list,
    use_chat_template: bool = True,
    max_samples: int = None,
) -> dict:
    """Evaluate MC1 accuracy for instruct model baseline."""
    if max_samples:
        dataset = dataset[:max_samples]

    correct = 0
    total = 0

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  MC1 eval: {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        # Score each choice
        best_score = float("-inf")
        best_idx = 0

        for j, choice in enumerate(choices):
            if use_chat_template:
                prompt = (
                    f"<|im_start|>user\n{question}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            else:
                prompt = f"Q: {question}\nA:"

            full_text = prompt + " " + choice
            tokens = model.to_tokens(full_text, prepend_bos=True)
            if tokens.shape[1] > 256:
                tokens = tokens[:, :256]

            with torch.no_grad():
                logits = model(tokens)

            # Compute log-prob of answer tokens
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


# ======================== Analysis ========================

def analyze_instruct_fep(results: list, n_layers: int, base_key: str) -> dict:
    """Analyze FEP for instruct model and compare with base."""
    from scipy import stats as scipy_stats

    feps = np.array([r["fep_layer"] for r in results])

    instruct_stats = {
        "n_samples": len(feps),
        "mean_fep": float(np.mean(feps)),
        "std_fep": float(np.std(feps)),
        "late_crystal_pct": float(np.mean(feps == n_layers)),
        "fep_depth": float(np.mean(feps)) / n_layers,
    }

    # Compare with base model
    base = BASE_MODEL_BASELINES.get(base_key, {})
    comparison = {"instruct": instruct_stats, "base": base}

    if base:
        comparison["delta_mean_fep"] = instruct_stats["mean_fep"] - base["mean_fep"]
        comparison["delta_late_crystal"] = (
            instruct_stats["late_crystal_pct"] - base["late_crystal_pct"]
        )
        # Is the difference meaningful?
        comparison["crystallization_preserved"] = (
            instruct_stats["late_crystal_pct"] > 0.5  # Still majority late
        )

    # Per-category if available
    category_stats = {}
    cats = defaultdict(list)
    for r in results:
        if "category" in r:
            cats[r["category"]].append(r["fep_layer"])

    for cat, cat_feps in cats.items():
        if len(cat_feps) >= 5:
            c = np.array(cat_feps)
            category_stats[cat] = {
                "n": len(c),
                "mean_fep": float(np.mean(c)),
                "late_crystal_pct": float(np.mean(c == n_layers)),
            }

    return {
        "instruct_stats": instruct_stats,
        "base_comparison": comparison,
        "category_stats": category_stats,
    }


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-mc1", action="store_true",
                        help="Skip MC1 evaluation (FEP only)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("INSTRUCTION-TUNED MODEL FEP PILOT")
    logger.info("=" * 70)

    # Load TruthfulQA
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")

    for model_key, config in INSTRUCT_MODELS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {config['name']}")
        logger.info(f"{'=' * 60}")

        model = load_model_hooked(config)
        n_layers = config["n_layers"]
        use_chat = config.get("chat_template", False)

        eval_dataset = dataset[:args.max_samples] if args.max_samples else dataset

        # FEP Detection
        results = []
        fep_dist = defaultdict(int)

        for i, sample in enumerate(eval_dataset):
            if i % 50 == 0:
                logger.info(f"  [{model_key}] FEP: {i}/{len(eval_dataset)}")

            question = sample["question"]
            best_answer = sample.get("best_answer", "")
            if not best_answer.strip():
                continue

            fep_result = detect_fep_instruct(
                model, question, best_answer,
                use_chat_template=use_chat,
            )
            if "error" in fep_result:
                continue

            fep_result["id"] = sample["id"]
            fep_result["category"] = sample.get("category", "Unknown")
            results.append(fep_result)
            fep_dist[fep_result["fep_layer"]] += 1

        # MC1 evaluation
        mc1_result = None
        if not args.skip_mc1:
            logger.info("Evaluating MC1...")
            mc1_result = evaluate_mc1_instruct(
                model, eval_dataset, use_chat_template=use_chat
            )
            logger.info(f"MC1: {mc1_result['mc1']:.4f} ({mc1_result['correct']}/{mc1_result['total']})")

        # Analysis
        analysis = analyze_instruct_fep(results, n_layers, config["base_model_key"])

        output = {
            "model": config["name"],
            "model_key": model_key,
            "base_model": config["base_model_key"],
            "n_layers": n_layers,
            "n_samples": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fep_distribution": dict(fep_dist),
            "analysis": analysis,
            "mc1_result": mc1_result,
            "per_sample_results": results,
        }

        output_path = RESULTS_DIR / f"instruct_fep_{model_key}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Saved to {output_path}")

        # Print summary
        stats = analysis["instruct_stats"]
        base_comp = analysis["base_comparison"]
        print(f"\n{'─' * 55}")
        print(f"Model: {config['name']}")
        print(f"  Mean FEP: {stats['mean_fep']:.1f} ± {stats['std_fep']:.1f}")
        print(f"  Late Crystallization: {stats['late_crystal_pct']:.1%}")
        if base_comp.get("delta_late_crystal") is not None:
            print(f"  vs Base: ΔFEP={base_comp['delta_mean_fep']:+.1f}, "
                  f"ΔCrystal={base_comp['delta_late_crystal']:+.1%}")
            status = "PRESERVED" if base_comp.get("crystallization_preserved") else "CHANGED"
            print(f"  Crystallization status: {status}")
        if mc1_result:
            print(f"  MC1 (instruct): {mc1_result['mc1']:.4f}")

        del model
        torch.cuda.empty_cache()

    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
