"""ITI Experiment on Pythia-1.4B with TruthfulQA.

Learns truthfulness directions from TruthfulQA correct/incorrect answer pairs,
then evaluates ITI steering across multiple coefficient values and layer
selection strategies.

This provides the positive intervention baseline: directional steering
should improve truthfulness where fixed scaling failed.
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
logger = logging.getLogger("iti_pythia")


def load_model():
    """Load Pythia-1.4B."""
    from mechlens.models.model_loader import load_model as ml_load
    from mechlens.config import DEFAULT_DEVICE
    model_name = "EleutherAI/pythia-1.4b"
    logger.info(f"Loading model: {model_name} (device={DEFAULT_DEVICE})")
    model = ml_load(model_name, dtype="float32" if DEFAULT_DEVICE == "mps" else "float16")
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    return model


def prepare_training_data(dataset: list[dict], model, n_train: int = 100) -> tuple[list[str], list[str]]:
    """Prepare contrastive prompt pairs for ITI direction learning.

    Uses TruthfulQA correct/incorrect answer completions as contrastive pairs.
    Prompts are formatted as "Q: {question}\nA: {answer}" to give clear context.

    Args:
        dataset: TruthfulQA samples
        model: Model (unused but kept for API consistency)
        n_train: Number of training samples to use

    Returns:
        (correct_prompts, incorrect_prompts) lists of formatted strings
    """
    correct_prompts = []
    incorrect_prompts = []

    for sample in dataset[:n_train]:
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        correct_answers = sample.get("correct_answers", [])
        incorrect_answers = sample.get("incorrect_answers", [])

        # Build correct prompt using best answer
        if best_answer:
            correct_prompts.append(f"Q: {question}\nA: {best_answer}")
        elif correct_answers:
            correct_prompts.append(f"Q: {question}\nA: {correct_answers[0]}")

        # Build incorrect prompt using first incorrect answer
        if incorrect_answers:
            incorrect_prompts.append(f"Q: {question}\nA: {incorrect_answers[0]}")

    # Balance the sets
    min_len = min(len(correct_prompts), len(incorrect_prompts))
    correct_prompts = correct_prompts[:min_len]
    incorrect_prompts = incorrect_prompts[:min_len]

    logger.info(f"Prepared {min_len} contrastive pairs for ITI training")
    return correct_prompts, incorrect_prompts


def run_iti_experiment(
    model,
    dataset: list[dict],
    n_train: int = 100,
    n_eval: int = 100,
    coefficients: list[float] | None = None,
    top_k_layers_options: list[int] | None = None,
) -> dict:
    """Run full ITI experiment with hyperparameter sweep.

    Args:
        model: HookedTransformer model
        dataset: Full TruthfulQA dataset
        n_train: Number of samples for direction learning
        n_eval: Number of samples for evaluation
        coefficients: List of steering coefficients to test
        top_k_layers_options: List of top-K layer counts to test

    Returns:
        Complete experiment results dict
    """
    from mechlens.intervention.iti import (
        learn_iti_directions,
        generate_with_iti,
        select_top_layers,
        serialize_directions,
    )
    from mechlens.benchmark.truthfulqa import (
        evaluate_truthfulqa,
        _check_truthful,
        _generate_response,
    )

    if coefficients is None:
        coefficients = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    if top_k_layers_options is None:
        top_k_layers_options = [3, 5, 10]

    # Split dataset: first n_train for direction learning, next n_eval for evaluation
    train_data = dataset[:n_train]
    eval_data = dataset[n_train:n_train + n_eval]

    if len(eval_data) < n_eval:
        logger.warning(
            f"Only {len(eval_data)} eval samples available "
            f"(requested {n_eval}). Using all available."
        )

    logger.info(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval")

    # Step 1: Learn ITI directions
    logger.info("=" * 50)
    logger.info("STEP 1: Learning ITI directions")
    logger.info("=" * 50)

    correct_prompts, incorrect_prompts = prepare_training_data(train_data, model, n_train)

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
        "model": "EleutherAI/pythia-1.4b",
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
    logger.info("STEP 2: Baseline evaluation")
    logger.info("=" * 50)

    baseline = evaluate_truthfulqa(model, eval_data, max_samples=len(eval_data), max_new_tokens=80)
    results["baseline"] = {
        "truthful_rate": baseline["truthful_rate"],
        "informative_rate": baseline["informative_rate"],
        "n_samples": baseline["n_samples"],
    }
    logger.info(
        f"Baseline: truthful={baseline['truthful_rate']:.3f}, "
        f"informative={baseline['informative_rate']:.3f}"
    )

    # Step 3: Sweep over coefficients and layer selections
    logger.info("=" * 50)
    logger.info("STEP 3: ITI steering evaluation sweep")
    logger.info("=" * 50)

    for top_k in top_k_layers_options:
        selected_layers = all_top_layers[:top_k]

        for coeff in coefficients:
            config_name = f"top{top_k}_coeff{coeff}"
            logger.info(f"Testing: {config_name} (layers={selected_layers})")

            # Build intervention function compatible with evaluate_truthfulqa
            def make_iti_intervention_fn(dirs, layers, c):
                def intervention_fn(mdl, question):
                    original, steered = generate_with_iti(
                        mdl, question, dirs,
                        coefficient=c,
                        layers=layers,
                        max_new_tokens=80,
                    )
                    return original, steered
                return intervention_fn

            intervention_fn = make_iti_intervention_fn(iti_directions, selected_layers, coeff)

            t0 = time.time()
            eval_result = evaluate_truthfulqa(
                model,
                eval_data,
                intervention_fn=intervention_fn,
                max_samples=len(eval_data),
                max_new_tokens=80,
            )
            eval_time = time.time() - t0

            result_entry = {
                "config": config_name,
                "top_k_layers": top_k,
                "selected_layers": selected_layers,
                "coefficient": coeff,
                "truthful_rate_before": eval_result["truthful_rate"],
                "truthful_rate_after": eval_result["truthful_rate_after"],
                "informative_rate_before": eval_result["informative_rate"],
                "informative_rate_after": eval_result["informative_rate_after"],
                "truthful_improvement": eval_result["truthful_improvement"],
                "eval_time_s": eval_time,
            }
            results["iti_results"].append(result_entry)

            logger.info(
                f"  {config_name}: truthful {eval_result['truthful_rate']:.3f} -> "
                f"{eval_result['truthful_rate_after']:.3f} "
                f"(+{eval_result['truthful_improvement']:+.3f}), "
                f"informative: {eval_result['informative_rate_after']:.3f}, "
                f"time={eval_time:.1f}s"
            )

    # Find best configuration
    if results["iti_results"]:
        best = max(results["iti_results"], key=lambda r: r["truthful_improvement"])
        results["best_config"] = {
            "config": best["config"],
            "truthful_improvement": best["truthful_improvement"],
            "truthful_rate_after": best["truthful_rate_after"],
            "informative_rate_after": best["informative_rate_after"],
            "top_k_layers": best["top_k_layers"],
            "coefficient": best["coefficient"],
        }
        logger.info(
            f"\nBest config: {best['config']} "
            f"(improvement: {best['truthful_improvement']:+.3f})"
        )

    # Save direction magnitudes for figure generation
    results["direction_info"] = serialize_directions(iti_directions)
    # Remove the actual direction vectors to keep JSON size manageable
    results["direction_info"].pop("directions", None)

    return results


def main():
    results_dir = PROJECT_ROOT / "results" / "pythia_iti"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MechLens ITI Experiment - Pythia-1.4B + TruthfulQA")
    logger.info("=" * 60)

    # Load model
    model = load_model()

    # Load TruthfulQA dataset
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tqa_path = download_truthfulqa(data_dir)
    dataset = load_truthfulqa(tqa_path)
    logger.info(f"Loaded {len(dataset)} TruthfulQA samples")

    # Run ITI experiment
    results = run_iti_experiment(
        model,
        dataset,
        n_train=100,
        n_eval=100,
        coefficients=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        top_k_layers_options=[3, 5, 10],
    )

    # Save results
    output_path = results_dir / "iti_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n\n=== ITI EXPERIMENT SUMMARY (Pythia-1.4B) ===")
    print(f"Training pairs: {results['n_train']}")
    print(f"Eval samples: {results['n_eval']}")
    print(f"Direction learning time: {results['direction_learning_time_s']:.1f}s")
    print(f"Baseline truthful rate: {results['baseline']['truthful_rate']:.3f}")

    if "best_config" in results:
        best = results["best_config"]
        print(f"\nBest configuration: {best['config']}")
        print(f"  Truthful rate after ITI: {best['truthful_rate_after']:.3f}")
        print(f"  Improvement: {best['truthful_improvement']:+.3f}")
        print(f"  Informative rate: {best['informative_rate_after']:.3f}")

    print("\nAll configurations tested:")
    for r in results["iti_results"]:
        print(
            f"  {r['config']:20s}: "
            f"truthful={r['truthful_rate_after']:.3f} "
            f"(+{r['truthful_improvement']:+.3f}), "
            f"informative={r['informative_rate_after']:.3f}"
        )


if __name__ == "__main__":
    main()
