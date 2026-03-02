"""ITI Experiment on Qwen2.5-0.5B with Chinese Hallucination Benchmark.

Learns truthfulness directions from Chinese hallucination samples
(50 train / 50 eval split), then evaluates ITI steering across
multiple coefficient values and layer selection strategies.

Complements the Pythia ITI experiment by demonstrating cross-language
and cross-architecture generality of directional intervention.
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
logger = logging.getLogger("iti_qwen")


def load_model():
    """Load Qwen2.5-0.5B."""
    from mechlens.models.model_loader import load_model as ml_load
    from mechlens.config import DEFAULT_DEVICE
    model_name = "Qwen/Qwen2.5-0.5B"
    logger.info(f"Loading model: {model_name} (device={DEFAULT_DEVICE})")
    model = ml_load(model_name, dtype="float32" if DEFAULT_DEVICE == "mps" else "float16")
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    return model


def prepare_chinese_training_data(
    dataset: list,
    model,
    n_train: int = 50,
) -> tuple[list[str], list[str]]:
    """Prepare contrastive prompt pairs from Chinese hallucination benchmark.

    For each sample:
    - Correct prompt: question + ground truth answer
    - Incorrect prompt: question + model's own (likely hallucinated) answer

    We first generate model outputs, then use those that don't match
    ground truth as the incorrect set, and ground truth completions
    as the correct set.

    Args:
        dataset: List of HallucinationSample
        model: HookedTransformer model for generating incorrect completions
        n_train: Number of training samples

    Returns:
        (correct_prompts, incorrect_prompts)
    """
    correct_prompts = []
    incorrect_prompts = []

    train_samples = dataset[:n_train]

    for sample in train_samples:
        question = sample.question
        ground_truth = sample.ground_truth

        # Correct: question + ground truth
        correct_prompts.append(f"{question}{ground_truth}")

        # Incorrect: generate model's own answer
        tokens = model.to_tokens(question)
        with torch.no_grad():
            output_ids = model.generate(tokens, max_new_tokens=50, do_sample=False)
        model_answer = model.to_string(output_ids[0, tokens.shape[1]:]).strip()

        # Use model's answer as incorrect (it may or may not be hallucinated,
        # but the contrastive signal comes from the difference with ground truth)
        incorrect_prompts.append(f"{question}{model_answer}")

    logger.info(f"Prepared {len(correct_prompts)} contrastive pairs for ITI training")
    return correct_prompts, incorrect_prompts


def run_iti_experiment(
    model,
    dataset: list,
    n_train: int = 50,
    coefficients: list[float] | None = None,
    top_k_layers_options: list[int] | None = None,
) -> dict:
    """Run ITI experiment on Chinese hallucination benchmark.

    Args:
        model: HookedTransformer model
        dataset: Full Chinese hallucination dataset (100 samples)
        n_train: Number of samples for direction learning
        coefficients: Steering coefficients to test
        top_k_layers_options: Top-K layer counts to test

    Returns:
        Complete experiment results
    """
    from mechlens.intervention.iti import (
        learn_iti_directions,
        generate_with_iti,
        select_top_layers,
        serialize_directions,
    )
    from mechlens.benchmark.chinese_hallucination import evaluate, _check_correct

    if coefficients is None:
        coefficients = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    if top_k_layers_options is None:
        top_k_layers_options = [3, 5, 10]

    # Split dataset: first n_train for learning, rest for evaluation
    train_data = dataset[:n_train]
    eval_data = dataset[n_train:]

    logger.info(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval")

    # Step 1: Learn ITI directions
    logger.info("=" * 50)
    logger.info("STEP 1: Learning ITI directions from Chinese hallucination data")
    logger.info("=" * 50)

    correct_prompts, incorrect_prompts = prepare_chinese_training_data(
        train_data, model, n_train
    )

    t0 = time.time()
    iti_directions = learn_iti_directions(
        model,
        correct_prompts=correct_prompts,
        incorrect_prompts=incorrect_prompts,
    )
    direction_time = time.time() - t0
    logger.info(f"Direction learning took {direction_time:.1f}s")

    # Get layer ranking
    all_top_layers = select_top_layers(iti_directions, top_k=model.cfg.n_layers)

    results = {
        "model": "Qwen/Qwen2.5-0.5B",
        "benchmark": "chinese_hallucination",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(correct_prompts),
        "n_eval": len(eval_data),
        "direction_learning_time_s": direction_time,
        "layer_ranking": all_top_layers,
        "projection_magnitudes": {
            str(k): v for k, v in iti_directions.projection_magnitudes.items()
        },
        "baseline": {},
        "iti_results": [],
    }

    # Step 2: Baseline evaluation (no intervention)
    logger.info("=" * 50)
    logger.info("STEP 2: Baseline evaluation on eval split")
    logger.info("=" * 50)

    baseline = evaluate(model, eval_data, model_name="Qwen/Qwen2.5-0.5B")
    results["baseline"] = {
        "hallucination_rate": baseline["hallucination_rate"],
        "per_type_rates": baseline["per_type_rates"],
        "per_domain_rates": baseline["per_domain_rates"],
        "n_samples": baseline["n_samples"],
    }
    logger.info(f"Baseline hallucination rate: {baseline['hallucination_rate']:.3f}")

    # Step 3: Sweep over coefficients and layer selections
    logger.info("=" * 50)
    logger.info("STEP 3: ITI steering evaluation sweep")
    logger.info("=" * 50)

    for top_k in top_k_layers_options:
        selected_layers = all_top_layers[:top_k]

        for coeff in coefficients:
            config_name = f"top{top_k}_coeff{coeff}"
            logger.info(f"Testing: {config_name} (layers={selected_layers})")

            def make_iti_intervention_fn(dirs, layers, c):
                def intervention_fn(mdl, question):
                    original, steered = generate_with_iti(
                        mdl, question, dirs,
                        coefficient=c,
                        layers=layers,
                        max_new_tokens=100,
                    )
                    return original, steered
                return intervention_fn

            intervention_fn = make_iti_intervention_fn(iti_directions, selected_layers, coeff)

            t0 = time.time()
            eval_result = evaluate(
                model,
                eval_data,
                intervention_fn=intervention_fn,
                model_name="Qwen/Qwen2.5-0.5B",
            )
            eval_time = time.time() - t0

            result_entry = {
                "config": config_name,
                "top_k_layers": top_k,
                "selected_layers": selected_layers,
                "coefficient": coeff,
                "hallucination_rate_before": eval_result["hallucination_rate"],
                "hallucination_rate_after": eval_result["hallucination_rate_after"],
                "hallucination_reduction": eval_result["hallucination_rate_reduction"],
                "per_type_rates": eval_result["per_type_rates"],
                "per_domain_rates": eval_result["per_domain_rates"],
                "eval_time_s": eval_time,
            }
            results["iti_results"].append(result_entry)

            logger.info(
                f"  {config_name}: hallucination {eval_result['hallucination_rate']:.3f} -> "
                f"{eval_result['hallucination_rate_after']:.3f} "
                f"(reduction: {eval_result['hallucination_rate_reduction']:+.3f}), "
                f"time={eval_time:.1f}s"
            )

    # Find best configuration
    if results["iti_results"]:
        best = max(results["iti_results"], key=lambda r: r["hallucination_reduction"])
        results["best_config"] = {
            "config": best["config"],
            "hallucination_reduction": best["hallucination_reduction"],
            "hallucination_rate_after": best["hallucination_rate_after"],
            "top_k_layers": best["top_k_layers"],
            "coefficient": best["coefficient"],
        }
        logger.info(
            f"\nBest config: {best['config']} "
            f"(reduction: {best['hallucination_reduction']:+.3f})"
        )

    # Save direction info (without full vectors)
    results["direction_info"] = serialize_directions(iti_directions)
    results["direction_info"].pop("directions", None)

    return results


def main():
    results_dir = PROJECT_ROOT / "results" / "qwen_0.5b_iti"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MechLens ITI Experiment - Qwen2.5-0.5B + Chinese Hallucination")
    logger.info("=" * 60)

    # Load model
    model = load_model()

    # Load Chinese hallucination benchmark
    from mechlens.benchmark.chinese_hallucination import load_dataset

    dataset_path = PROJECT_ROOT / "data" / "chinese_hallucination_bench" / "dataset.json"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} Chinese hallucination samples")

    # Run ITI experiment (50 train / 50 eval split)
    results = run_iti_experiment(
        model,
        dataset,
        n_train=50,
        coefficients=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        top_k_layers_options=[3, 5, 10],
    )

    # Save results
    output_path = results_dir / "iti_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n\n=== ITI EXPERIMENT SUMMARY (Qwen2.5-0.5B) ===")
    print(f"Training pairs: {results['n_train']}")
    print(f"Eval samples: {results['n_eval']}")
    print(f"Direction learning time: {results['direction_learning_time_s']:.1f}s")
    print(f"Baseline hallucination rate: {results['baseline']['hallucination_rate']:.3f}")

    if "best_config" in results:
        best = results["best_config"]
        print(f"\nBest configuration: {best['config']}")
        print(f"  Hallucination rate after ITI: {best['hallucination_rate_after']:.3f}")
        print(f"  Reduction: {best['hallucination_reduction']:+.3f}")

    print("\nAll configurations tested:")
    for r in results["iti_results"]:
        print(
            f"  {r['config']:20s}: "
            f"halluc={r['hallucination_rate_after']:.3f} "
            f"(reduction: {r['hallucination_reduction']:+.3f})"
        )


if __name__ == "__main__":
    main()
