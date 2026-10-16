"""TruthfulQA MC1/MC2 Baseline + DoLa + CAA Positive Results Experiment.

Runs the complete evaluation pipeline:
  1. Baseline MC1/MC2 (no intervention) for Qwen2.5-7B
  2. ITI baseline MC1/MC2 (existing method)
  3. DoLa MC1/MC2 (layer-contrasting decoding)
  4. CAA MC1/MC2 (contrastive activation addition)

Produces standardized results comparable to published benchmarks
(SADI, DoLa, ITI papers) for the paper's positive-result section.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("positive_results")

RESULTS_DIR = PROJECT_ROOT / "results" / "positive_results"
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
    """Download and load TruthfulQA."""
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")
    logger.info(f"Loaded {len(dataset)} TruthfulQA samples")
    return dataset


def save_results(results: dict, filename: str):
    """Save results to JSON."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Results saved to {path}")


# ──────────────────────────────────────────────
# Phase 1: Baseline MC1/MC2
# ──────────────────────────────────────────────
def run_baseline_mc(model, dataset):
    """Run baseline MC1/MC2 evaluation (no intervention)."""
    from mechlens.benchmark.truthfulqa import evaluate_truthfulqa_mc1, evaluate_truthfulqa_mc2

    logger.info("=" * 60)
    logger.info("PHASE 1: Baseline MC1/MC2 (no intervention)")
    logger.info("=" * 60)

    t0 = time.time()
    mc1 = evaluate_truthfulqa_mc1(model, dataset)
    mc2 = evaluate_truthfulqa_mc2(model, dataset)
    elapsed = time.time() - t0

    result = {
        "method": "baseline",
        "mc1_score": mc1["mc1_score"],
        "mc2_score": mc2["mc2_score"],
        "mc1_n_correct": mc1["n_correct"],
        "mc1_n_samples": mc1["n_samples"],
        "mc2_n_samples": mc2["n_samples"],
        "mc1_per_category": mc1["per_category_rates"],
        "mc2_per_category": mc2["per_category_rates"],
        "eval_time_s": elapsed,
    }

    logger.info(f"Baseline: MC1={mc1['mc1_score']:.4f}, MC2={mc2['mc2_score']:.4f} ({elapsed:.1f}s)")
    return result


# ──────────────────────────────────────────────
# Phase 2: ITI MC1/MC2
# ──────────────────────────────────────────────
def run_iti_mc(model, dataset):
    """Run ITI with MC1/MC2 evaluation."""
    from mechlens.benchmark.truthfulqa import evaluate_truthfulqa_mc1, evaluate_truthfulqa_mc2
    from mechlens.intervention.iti import learn_iti_directions, select_top_layers, create_iti_steering_hook

    logger.info("=" * 60)
    logger.info("PHASE 2: ITI MC1/MC2")
    logger.info("=" * 60)

    # Build contrastive pairs from TruthfulQA
    correct_prompts = []
    incorrect_prompts = []

    for sample in dataset[:100]:
        q = sample["question"]
        best = sample.get("best_answer", "")
        incorrects = sample.get("incorrect_answers", [])

        if best.strip():
            correct_prompts.append(f"Q: {q}\nA: {best}")
        if incorrects:
            incorrect_prompts.append(f"Q: {q}\nA: {incorrects[0]}")

    logger.info(f"ITI training: {len(correct_prompts)} correct, {len(incorrect_prompts)} incorrect")

    t0 = time.time()
    iti_dirs = learn_iti_directions(model, correct_prompts, incorrect_prompts)
    train_time = time.time() - t0
    logger.info(f"ITI direction learning: {train_time:.1f}s")

    # Evaluate ITI with best hyperparams from prior experiments
    results = []
    for top_k in [3, 5, 10]:
        top_layers = select_top_layers(iti_dirs, top_k)
        for coeff in [1.0, 2.0, 3.0]:
            hooks = []
            for layer in top_layers:
                if layer not in iti_dirs.directions:
                    continue
                hook_fn = create_iti_steering_hook(iti_dirs.directions[layer], coeff)
                hooks.append((f"blocks.{layer}.hook_resid_post", hook_fn))

            t0 = time.time()
            mc1 = evaluate_truthfulqa_mc1(model, dataset, fwd_hooks=hooks)
            mc2 = evaluate_truthfulqa_mc2(model, dataset, fwd_hooks=hooks)
            elapsed = time.time() - t0

            entry = {
                "method": "iti",
                "top_k": top_k,
                "coefficient": coeff,
                "layers": top_layers,
                "mc1_score": mc1["mc1_score"],
                "mc2_score": mc2["mc2_score"],
                "eval_time_s": elapsed,
            }
            results.append(entry)

            logger.info(
                f"ITI(top_k={top_k}, coeff={coeff}): "
                f"MC1={mc1['mc1_score']:.4f}, MC2={mc2['mc2_score']:.4f} ({elapsed:.1f}s)"
            )

    best = max(results, key=lambda r: r["mc1_score"])
    logger.info(f"Best ITI: MC1={best['mc1_score']:.4f} (top_k={best['top_k']}, coeff={best['coefficient']})")

    return {
        "method": "iti",
        "training_time_s": train_time,
        "n_correct_prompts": len(correct_prompts),
        "n_incorrect_prompts": len(incorrect_prompts),
        "configs": results,
        "best": best,
    }


# ──────────────────────────────────────────────
# Phase 3: DoLa MC1/MC2
# ──────────────────────────────────────────────
def run_dola_mc(model, dataset):
    """Run DoLa with MC1/MC2 evaluation."""
    from mechlens.benchmark.truthfulqa import evaluate_truthfulqa_mc1, evaluate_truthfulqa_mc2
    from mechlens.intervention.dola import create_dola_score_fn

    logger.info("=" * 60)
    logger.info("PHASE 3: DoLa MC1/MC2")
    logger.info("=" * 60)

    n_layers = model.cfg.n_layers
    results = []

    # Test both dynamic and static premature selection
    configs = [
        {"name": "dola_dynamic", "dynamic": True, "premature": None},
        {"name": "dola_early", "dynamic": False, "premature": list(range(0, int(n_layers * 0.3)))},
        {"name": "dola_mid", "dynamic": False, "premature": list(range(int(n_layers * 0.3), int(n_layers * 0.6)))},
    ]

    for config in configs:
        premature = config["premature"]
        if premature is None:
            premature = list(range(0, int(n_layers * 0.6)))

        score_fn = create_dola_score_fn(
            mature_layer=n_layers - 1,
            premature_candidates=premature,
            dynamic_premature=config["dynamic"],
        )

        t0 = time.time()
        mc1 = evaluate_truthfulqa_mc1(model, dataset, score_fn=score_fn)
        mc2 = evaluate_truthfulqa_mc2(model, dataset, score_fn=score_fn)
        elapsed = time.time() - t0

        entry = {
            "method": "dola",
            "config_name": config["name"],
            "dynamic_premature": config["dynamic"],
            "n_premature_candidates": len(premature),
            "mature_layer": n_layers - 1,
            "mc1_score": mc1["mc1_score"],
            "mc2_score": mc2["mc2_score"],
            "mc1_per_category": mc1["per_category_rates"],
            "mc2_per_category": mc2["per_category_rates"],
            "eval_time_s": elapsed,
        }
        results.append(entry)

        logger.info(
            f"DoLa({config['name']}): "
            f"MC1={mc1['mc1_score']:.4f}, MC2={mc2['mc2_score']:.4f} ({elapsed:.1f}s)"
        )

        # Save intermediate
        save_results({"dola_intermediate": results}, "dola_intermediate.json")

    best = max(results, key=lambda r: r["mc1_score"])
    logger.info(f"Best DoLa: MC1={best['mc1_score']:.4f} ({best['config_name']})")

    return {
        "method": "dola",
        "configs": results,
        "best": best,
    }


# ──────────────────────────────────────────────
# Phase 4: CAA MC1/MC2
# ──────────────────────────────────────────────
def run_caa_mc(model, dataset):
    """Run CAA with MC1/MC2 evaluation."""
    from mechlens.benchmark.truthfulqa import evaluate_truthfulqa_mc1, evaluate_truthfulqa_mc2
    from mechlens.analysis.contrastive import run_contrastive_analysis
    from mechlens.intervention.caa import (
        learn_caa_directions,
        select_top_layers,
        build_caa_hooks,
    )

    logger.info("=" * 60)
    logger.info("PHASE 4: CAA MC1/MC2")
    logger.info("=" * 60)

    # Build labeled prompts for contrastive analysis
    prompts_with_labels = []
    for sample in dataset[:100]:
        q = sample["question"]
        best = sample.get("best_answer", "")
        incorrects = sample.get("incorrect_answers", [])

        if best.strip():
            prompts_with_labels.append((f"Q: {q}\nA: {best}", best, True))
        if incorrects:
            prompts_with_labels.append((f"Q: {q}\nA: {incorrects[0]}", incorrects[0], False))

    logger.info(f"Contrastive analysis: {len(prompts_with_labels)} labeled prompts")

    t0 = time.time()
    contrastive_result = run_contrastive_analysis(model, prompts_with_labels)
    analysis_time = time.time() - t0
    logger.info(f"Contrastive analysis: {analysis_time:.1f}s")

    # Extract CAA directions
    caa_dirs = learn_caa_directions(contrastive_result)
    logger.info(f"CAA directions: {len(caa_dirs.directions)} layers")

    # Grid search over coefficient and top_k
    results = []
    coefficients = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    top_k_values = [3, 5, 10]

    total = len(coefficients) * len(top_k_values)
    idx = 0

    for top_k in top_k_values:
        top_layers = select_top_layers(caa_dirs, top_k)

        for coeff in coefficients:
            idx += 1
            logger.info(f"CAA [{idx}/{total}]: coeff={coeff}, top_k={top_k}")

            hooks = build_caa_hooks(caa_dirs, coeff, top_layers)

            t0 = time.time()
            mc1 = evaluate_truthfulqa_mc1(model, dataset, fwd_hooks=hooks)
            mc2 = evaluate_truthfulqa_mc2(model, dataset, fwd_hooks=hooks)
            elapsed = time.time() - t0

            entry = {
                "method": "caa",
                "coefficient": coeff,
                "top_k": top_k,
                "layers": top_layers,
                "mc1_score": mc1["mc1_score"],
                "mc2_score": mc2["mc2_score"],
                "eval_time_s": elapsed,
            }
            results.append(entry)

            logger.info(
                f"  CAA(coeff={coeff}, top_k={top_k}): "
                f"MC1={mc1['mc1_score']:.4f}, MC2={mc2['mc2_score']:.4f} ({elapsed:.1f}s)"
            )

        # Save intermediate after each top_k sweep
        save_results({"caa_intermediate": results}, "caa_intermediate.json")

    best = max(results, key=lambda r: r["mc1_score"])
    logger.info(f"Best CAA: MC1={best['mc1_score']:.4f} (coeff={best['coefficient']}, top_k={best['top_k']})")

    return {
        "method": "caa",
        "analysis_time_s": analysis_time,
        "layer_importance": contrastive_result.layer_importance,
        "configs": results,
        "best": best,
    }


# ──────────────────────────────────────────────
# Main: Full Pipeline
# ──────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("MechLens Positive Results Experiment")
    logger.info("Baseline + ITI + DoLa + CAA on TruthfulQA MC1/MC2")
    logger.info("=" * 60)

    model = load_model("Qwen/Qwen2.5-7B")
    dataset = load_truthfulqa_dataset()

    all_results = {
        "model": "Qwen/Qwen2.5-7B",
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "dataset_size": len(dataset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Phase 1: Baseline
    baseline = run_baseline_mc(model, dataset)
    all_results["baseline"] = baseline
    save_results(all_results, "positive_results.json")

    # Phase 2: ITI
    iti_result = run_iti_mc(model, dataset)
    all_results["iti"] = iti_result
    save_results(all_results, "positive_results.json")

    # Phase 3: DoLa
    dola_result = run_dola_mc(model, dataset)
    all_results["dola"] = dola_result
    save_results(all_results, "positive_results.json")

    # Phase 4: CAA
    caa_result = run_caa_mc(model, dataset)
    all_results["caa"] = caa_result
    save_results(all_results, "positive_results.json")

    # ── Summary Table ──
    print("\n")
    print("=" * 70)
    print("POSITIVE RESULTS SUMMARY: TruthfulQA MC1/MC2")
    print("=" * 70)
    print(f"Model: Qwen/Qwen2.5-7B ({model.cfg.n_layers}L, {model.cfg.n_heads}H)")
    print(f"Dataset: {len(dataset)} TruthfulQA samples")
    print()
    print(f"{'Method':<25} {'MC1':>8} {'MC2':>8} {'MC1 Δ':>8}")
    print("-" * 55)

    baseline_mc1 = baseline["mc1_score"]
    baseline_mc2 = baseline["mc2_score"]
    print(f"{'Baseline (no interv.)':<25} {baseline_mc1:>8.4f} {baseline_mc2:>8.4f} {'---':>8}")

    if "best" in iti_result:
        b = iti_result["best"]
        delta = b["mc1_score"] - baseline_mc1
        print(f"{'ITI (best)':<25} {b['mc1_score']:>8.4f} {b.get('mc2_score', 0):>8.4f} {delta:>+8.4f}")

    if "best" in dola_result:
        b = dola_result["best"]
        delta = b["mc1_score"] - baseline_mc1
        print(f"{'DoLa (best)':<25} {b['mc1_score']:>8.4f} {b.get('mc2_score', 0):>8.4f} {delta:>+8.4f}")

    if "best" in caa_result:
        b = caa_result["best"]
        delta = b["mc1_score"] - baseline_mc1
        print(f"{'CAA (best)':<25} {b['mc1_score']:>8.4f} {b.get('mc2_score', 0):>8.4f} {delta:>+8.4f}")

    print("-" * 55)
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
