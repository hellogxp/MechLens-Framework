"""MMLU Cross-Benchmark FEP Validation Experiment.

Validates Late Crystallization on MMLU (Hendrycks et al., 2021) to address
the "single benchmark" concern. Tests whether FEP distributions on MMLU
show the same late crystallization pattern observed on TruthfulQA.

GPU time: ~3 hours on A100 40GB (3 models x ~1h each)

Usage:
    python experiments/run_mmlu_fep_validation.py [--models qwen] [--max-samples 500]
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
logger = logging.getLogger("mmlu_fep")

RESULTS_DIR = PROJECT_ROOT / "results" / "mmlu_fep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reuse model configs from tuned lens script
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B",
        "local_path": None,
        "n_layers": 28,
    },
    "llama": {
        "name": "meta-llama/Llama-3.1-8B",
        "local_path": "/root/.cache/modelscope/LLM-Research/Meta-Llama-3___1-8B",
        "n_layers": 32,
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
        "local_path": "/root/.cache/modelscope/AI-ModelScope/Mistral-7B-v0___1",
        "n_layers": 32,
    },
}

# MMLU subjects grouped by knowledge type for spectrum analysis
MMLU_SUBJECT_GROUPS = {
    "STEM": [
        "abstract_algebra", "college_mathematics", "college_physics",
        "elementary_mathematics", "high_school_mathematics", "high_school_physics",
        "high_school_chemistry", "machine_learning", "computer_security",
    ],
    "Humanities": [
        "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "philosophy", "moral_scenarios",
    ],
    "Social_Sciences": [
        "high_school_psychology", "sociology", "econometrics",
        "high_school_macroeconomics", "high_school_microeconomics",
    ],
    "Other": [
        "clinical_knowledge", "medical_genetics", "anatomy",
        "professional_medicine", "nutrition",
    ],
}


# ======================== Model Loading ========================

def load_model_hooked(model_key: str):
    """Load model as HookedTransformer."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    local_path = config.get("local_path")

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
        from mechlens.models.model_loader import load_model as ml_load
        model = ml_load(model_name, dtype="float16")

    logger.info(f"Loaded: {model.cfg.n_layers}L, {model.cfg.n_heads}H")
    return model


# ======================== MMLU Loading ========================

def load_mmlu_dataset(subjects: list[str] | None = None, max_per_subject: int = 50) -> list[dict]:
    """Load MMLU dataset from HuggingFace.

    Returns list of {question, answer, subject, choices} dicts.
    """
    from datasets import load_dataset

    all_subjects = []
    if subjects is None:
        # Use representative subjects from each group
        for group, subjs in MMLU_SUBJECT_GROUPS.items():
            all_subjects.extend(subjs)
    else:
        all_subjects = subjects

    samples = []
    choice_labels = ["A", "B", "C", "D"]

    for subject in all_subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            for i, item in enumerate(ds):
                if i >= max_per_subject:
                    break

                question = item["question"]
                choices = item["choices"]
                correct_idx = item["answer"]
                correct_answer = choices[correct_idx]

                # Format as multiple-choice prompt
                prompt_parts = [f"Question: {question}"]
                for j, choice in enumerate(choices):
                    prompt_parts.append(f"{choice_labels[j]}. {choice}")
                prompt_parts.append("Answer:")
                prompt = "\n".join(prompt_parts)

                samples.append({
                    "id": f"{subject}_{i}",
                    "question": prompt,
                    "correct_answer": correct_answer,
                    "correct_label": choice_labels[correct_idx],
                    "subject": subject,
                    "group": next(
                        (g for g, subjs in MMLU_SUBJECT_GROUPS.items() if subject in subjs),
                        "Other"
                    ),
                })

            logger.info(f"Loaded {min(len(ds), max_per_subject)} samples from {subject}")
        except Exception as e:
            logger.warning(f"Failed to load {subject}: {e}")
            continue

    logger.info(f"Total MMLU samples loaded: {len(samples)}")
    return samples


# ======================== FEP Detection ========================

def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def detect_fep_mmlu(
    model,
    prompt: str,
    correct_answer: str,
    top_k: int = 10,
) -> dict:
    """Detect FEP for an MMLU sample.

    Uses the first token of the correct answer for tracking,
    same as TruthfulQA FEP detection.
    """
    n_layers = model.cfg.n_layers

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 256:
        tokens = tokens[:, :256]  # Truncate long prompts

    answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)[0]
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0].item()

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    layer_ranks = []
    layer_in_topk = []

    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        logits = unembed_at_layer(model, resid)
        probs = F.softmax(logits.float(), dim=-1)

        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank = rank[0].item() if len(rank) > 0 else probs.shape[0]

        layer_ranks.append(rank)
        layer_in_topk.append(rank < top_k)

    fep_layer = n_layers
    for layer in range(n_layers):
        if layer_in_topk[layer]:
            fep_layer = layer
            break

    return {
        "fep_layer": fep_layer,
        "layer_ranks": layer_ranks,
        "final_rank": layer_ranks[-1],
        "target_token_str": model.to_single_str_token(target_token),
    }


# ======================== Analysis ========================

def analyze_mmlu_fep(results: list, n_layers: int) -> dict:
    """Analyze FEP distribution on MMLU."""
    feps = np.array([r["fep_layer"] for r in results])

    # Overall stats
    late_crystal_pct = float(np.mean(feps == n_layers))
    mean_fep = float(np.mean(feps))
    std_fep = float(np.std(feps))
    fep_depth = mean_fep / n_layers

    # Per-group analysis (STEM vs Humanities vs Social Sciences)
    group_stats = {}
    groups = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r["fep_layer"])

    for group, group_feps in groups.items():
        g = np.array(group_feps)
        group_stats[group] = {
            "n": len(g),
            "mean_fep": float(np.mean(g)),
            "std_fep": float(np.std(g)),
            "late_crystal_pct": float(np.mean(g == n_layers)),
            "fep_depth": float(np.mean(g)) / n_layers,
        }

    # Per-subject analysis
    subject_stats = {}
    subjects = defaultdict(list)
    for r in results:
        subjects[r["subject"]].append(r["fep_layer"])

    for subj, subj_feps in subjects.items():
        if len(subj_feps) >= 5:
            s = np.array(subj_feps)
            subject_stats[subj] = {
                "n": len(s),
                "mean_fep": float(np.mean(s)),
                "std_fep": float(np.std(s)),
                "late_crystal_pct": float(np.mean(s == n_layers)),
            }

    return {
        "overall": {
            "n_samples": len(feps),
            "mean_fep": mean_fep,
            "std_fep": std_fep,
            "late_crystallization_pct": late_crystal_pct,
            "fep_depth": fep_depth,
        },
        "per_group": group_stats,
        "per_subject": subject_stats,
    }


def compare_with_truthfulqa(mmlu_analysis: dict, model_key: str) -> dict:
    """Compare MMLU FEP results with TruthfulQA baseline from paper."""
    # TruthfulQA baselines from paper
    tqa_baselines = {
        "qwen": {"mean_fep": 27.3, "std_fep": 1.8, "late_crystal_pct": 0.859},
        "llama": {"mean_fep": 29.4, "std_fep": 4.9, "late_crystal_pct": 0.710},
        "mistral": {"mean_fep": 26.3, "std_fep": 6.2, "late_crystal_pct": 0.271},
    }

    tqa = tqa_baselines.get(model_key, {})
    mmlu = mmlu_analysis["overall"]

    comparison = {
        "truthfulqa": tqa,
        "mmlu": {
            "mean_fep": mmlu["mean_fep"],
            "std_fep": mmlu["std_fep"],
            "late_crystal_pct": mmlu["late_crystallization_pct"],
        },
    }

    if tqa:
        comparison["delta_mean_fep"] = mmlu["mean_fep"] - tqa["mean_fep"]
        comparison["delta_late_crystal"] = mmlu["late_crystallization_pct"] - tqa["late_crystal_pct"]
        comparison["consistent"] = (
            abs(comparison["delta_late_crystal"]) < 0.15  # Within 15pp
        )

    return comparison


# ======================== Main ========================

def run_mmlu_for_model(model_key: str, max_samples: int = None) -> dict:
    """Run MMLU FEP analysis for one model."""
    config = MODEL_CONFIGS[model_key]
    n_layers = config["n_layers"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"MMLU FEP Validation: {config['name']}")
    logger.info(f"{'=' * 60}")

    model = load_model_hooked(model_key)

    # Load MMLU
    mmlu_data = load_mmlu_dataset(max_per_subject=50)
    if max_samples and len(mmlu_data) > max_samples:
        import random
        random.seed(42)
        mmlu_data = random.sample(mmlu_data, max_samples)

    # Run FEP detection
    results = []
    fep_dist = defaultdict(int)

    for i, sample in enumerate(mmlu_data):
        if i % 100 == 0:
            logger.info(f"  [{model_key}] MMLU FEP: {i}/{len(mmlu_data)}")

        fep_result = detect_fep_mmlu(
            model, sample["question"], sample["correct_answer"]
        )
        if "error" in fep_result:
            continue

        fep_result["id"] = sample["id"]
        fep_result["subject"] = sample["subject"]
        fep_result["group"] = sample["group"]
        results.append(fep_result)
        fep_dist[fep_result["fep_layer"]] += 1

    # Analyze
    analysis = analyze_mmlu_fep(results, n_layers)
    tqa_comparison = compare_with_truthfulqa(analysis, model_key)

    del model
    torch.cuda.empty_cache()

    return {
        "model": config["name"],
        "model_key": model_key,
        "n_layers": n_layers,
        "n_samples": len(results),
        "benchmark": "MMLU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fep_distribution": dict(fep_dist),
        "analysis": analysis,
        "truthfulqa_comparison": tqa_comparison,
        "per_sample_results": results,
    }


def print_summary(all_results: dict):
    """Print summary with cross-benchmark comparison."""
    print("\n" + "=" * 70)
    print("MMLU CROSS-BENCHMARK FEP VALIDATION")
    print("=" * 70)

    for model_key, result in all_results.items():
        a = result["analysis"]["overall"]
        comp = result["truthfulqa_comparison"]
        print(f"\n{'─' * 55}")
        print(f"Model: {result['model']} ({result['n_layers']}L, {a['n_samples']} MMLU samples)")
        print(f"{'─' * 55}")
        print(f"  MMLU:       Mean FEP={a['mean_fep']:.1f}±{a['std_fep']:.1f}, "
              f"Late Crystal={a['late_crystallization_pct']:.1%}")
        if comp.get("truthfulqa"):
            tqa = comp["truthfulqa"]
            print(f"  TruthfulQA: Mean FEP={tqa['mean_fep']:.1f}±{tqa['std_fep']:.1f}, "
                  f"Late Crystal={tqa['late_crystal_pct']:.1%}")
            print(f"  Delta:      ΔFEP={comp['delta_mean_fep']:+.1f}, "
                  f"ΔCrystal={comp['delta_late_crystal']:+.1%}")
            status = "CONSISTENT" if comp.get("consistent") else "DIVERGENT"
            print(f"  Status:     {status}")

        # Per-group breakdown
        print(f"\n  Per-Group FEP (MMLU):")
        for group, stats in result["analysis"]["per_group"].items():
            print(f"    {group:20s}: FEP={stats['mean_fep']:.1f}±{stats['std_fep']:.1f}, "
                  f"Late={stats['late_crystal_pct']:.1%} (n={stats['n']})")

    print("\n=== EXPERIMENT COMPLETE ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    all_results = {}

    for model_key in args.models:
        try:
            result = run_mmlu_for_model(model_key, max_samples=args.max_samples)
            all_results[model_key] = result

            output_path = RESULTS_DIR / f"mmlu_fep_{model_key}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed for {model_key}: {e}", exc_info=True)

    combined_path = RESULTS_DIR / "mmlu_fep_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

    print_summary(all_results)


if __name__ == "__main__":
    main()
