#!/usr/bin/env python3
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
"""Cross-Architecture FEP Detection — EMNLP Extended Version (5 models).

Uses HuggingFace models directly with device_map="auto" for reliable multi-GPU support.
Bypasses TransformerLens to avoid n_devices tensor mismatch bugs.

Architecture coverage:
  1. Qwen2.5-7B (GQA, 28 layers)
  2. Llama-3.1-8B (GQA, 32 layers)
  3. Mistral-7B-v0.1 (GQA + SWA, 32 layers)
  4. Pythia-6.9B (MHA, 32 layers)
  5. Gemma-7B (MQA, 28 layers)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cross_arch_emnlp")

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_architecture_emnlp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== EMNLP: 5 Model Families =====
CROSS_ARCH_MODELS = [
    "Qwen/Qwen2.5-7B",            # GQA, 28 layers
    "meta-llama/Llama-3.1-8B",     # GQA, 32 layers
    "mistralai/Mistral-7B-v0.1",   # GQA + SWA, 32 layers
    "EleutherAI/pythia-6.9b",      # MHA (standard multi-head), 32 layers
    "google/gemma-7b",             # MQA (multi-query), 28 layers
]

LOCAL_MODEL_PATHS = {
    "Qwen/Qwen2.5-7B": "/home/admin/workspace/models_cache/Qwen/Qwen2___5-7B",
    "meta-llama/Llama-3.1-8B": "/home/admin/workspace/models_cache/LLM-Research/Meta-Llama-3___1-8B",
    "mistralai/Mistral-7B-v0.1": "/home/admin/workspace/models_cache/AI-ModelScope/Mistral-7B-v0___1",
    "EleutherAI/pythia-6.9b": "/home/admin/workspace/models_cache/EleutherAI/pythia-6___9b",
    "google/gemma-7b": "/home/admin/workspace/models_cache/AI-ModelScope/gemma-7b",
}


def load_model(model_name: str):
    """Load model with HuggingFace device_map='auto' for multi-GPU support."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = LOCAL_MODEL_PATHS.get(model_name)
    if not local_path or not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local path not found: {local_path}")

    logger.info(f"Loading from local: {local_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {model_name} — {n_layers} layers, device_map=auto")
    return model, tokenizer


def load_truthfulqa_dataset():
    """Load TruthfulQA dataset."""
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")
    return dataset


def get_model_components(model, model_name: str):
    """Get architecture-specific final norm and lm_head modules."""
    if "pythia" in model_name.lower():
        norm_module = model.gpt_neox.final_layer_norm
        lm_head_module = model.embed_out
    else:
        # Works for Qwen, Llama, Mistral, Gemma
        norm_module = model.model.norm
        lm_head_module = model.lm_head
    return norm_module, lm_head_module


def unembed_at_layer(norm_module, lm_head_module, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply final layer norm + lm_head to get logits from hidden state.
    Handles device movements automatically."""
    target_device = next(norm_module.parameters()).device
    h = hidden_state.to(target_device).to(torch.float16)
    normed = norm_module(h)
    # lm_head may be on a different device
    head_device = next(lm_head_module.parameters()).device
    normed = normed.to(head_device)
    logits = lm_head_module(normed)
    return logits


def detect_fep_for_sample(
    model,
    tokenizer,
    norm_module,
    lm_head_module,
    question: str,
    correct_answer: str,
    top_k: int = 10,
) -> dict:
    """Detect Factual Emergence Point using HuggingFace output_hidden_states."""
    n_layers = model.config.num_hidden_layers

    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Get answer's first token
    answer_tokens = tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0]

    # Forward pass with all hidden states
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # (n_layers + 1,) tuple

    # Track answer rank at each layer (last position)
    layer_ranks = []
    layer_in_topk = []
    fep_layer = None

    for l in range(n_layers):
        # hidden_states[l+1] is output of layer l (index 0 is embedding output)
        h = hidden_states[l + 1][0, -1, :]  # last token hidden state
        logits = unembed_at_layer(norm_module, lm_head_module, h)

        sorted_indices = logits.argsort(descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank_val = rank[0].item() if len(rank) > 0 else logits.shape[-1]

        layer_ranks.append(rank_val)
        in_topk = rank_val < top_k
        layer_in_topk.append(in_topk)

        if in_topk and fep_layer is None:
            fep_layer = l

    # FEP = first layer in top-k; if never, set to n_layers (final)
    if fep_layer is None:
        fep_layer = n_layers  # never entered top-k

    return {
        "fep_layer": fep_layer,
        "n_layers": n_layers,
        "fep_depth": fep_layer / n_layers,
        "final_rank": layer_ranks[-1],
        "layer_ranks": layer_ranks,
        "target_token": target_token,
        "late_crystallization": fep_layer == n_layers,
    }


def run_fep_detection_for_model(model, tokenizer, model_name: str, dataset: list, max_samples: int = None) -> dict:
    """Run FEP detection for all samples on a model."""
    norm_module, lm_head_module = get_model_components(model, model_name)
    samples_to_process = dataset[:max_samples] if max_samples else dataset
    logger.info(f"Running FEP on {model_name}: {len(samples_to_process)} samples")

    results = []
    for idx, sample in enumerate(samples_to_process):
        if idx % 50 == 0:
            logger.info(f"  [{model_name}] {idx}/{len(samples_to_process)}")

        question = sample.get("question", "")
        correct_answers = sample.get("correct_answers", [])
        if not correct_answers:
            continue

        try:
            torch.cuda.empty_cache()
            result = detect_fep_for_sample(
                model, tokenizer, norm_module, lm_head_module,
                question, correct_answers[0]
            )
            if "error" not in result:
                result["sample_idx"] = idx
                results.append(result)
        except Exception as e:
            logger.warning(f"  Sample {idx} error: {e}")
            continue

    # Compute summary statistics
    if not results:
        return {"model_name": model_name, "error": "no_valid_results"}

    fep_layers = [r["fep_layer"] for r in results]
    n_layers = results[0]["n_layers"]
    late_crystal_count = sum(1 for r in results if r["late_crystallization"])
    late_crystal_pct = late_crystal_count / len(results) * 100

    # Mean FEP depth (normalized)
    fep_depths = [r["fep_depth"] for r in results]
    mean_fep_depth = np.mean(fep_depths)

    # Correct at final layer
    correct_final = sum(1 for r in results if r["final_rank"] < 10)
    correct_final_pct = correct_final / len(results) * 100

    summary = {
        "model_name": model_name,
        "n_layers": n_layers,
        "n_samples": len(results),
        "late_crystallization_pct": round(late_crystal_pct, 1),
        "mean_fep_depth": round(mean_fep_depth * 100, 1),
        "correct_at_final_layer_pct": round(correct_final_pct, 1),
        "fep_distribution": {
            "early_(<50%)": sum(1 for d in fep_depths if d < 0.5),
            "mid_(50-80%)": sum(1 for d in fep_depths if 0.5 <= d < 0.8),
            "late_(>80%)": sum(1 for d in fep_depths if d >= 0.8),
        },
        "per_sample_results": results,
    }

    return summary


def compare_architectures(all_results: dict) -> dict:
    """Generate cross-architecture comparison."""
    comparison = {}
    for name, res in all_results.items():
        if "error" in res:
            continue
        comparison[name] = {
            "n_layers": res["n_layers"],
            "late_crystallization_pct": res["late_crystallization_pct"],
            "mean_fep_depth": res["mean_fep_depth"],
            "correct_at_final_pct": res["correct_at_final_layer_pct"],
            "fep_distribution": res["fep_distribution"],
        }
    return comparison


def main():
    """Main execution: run FEP detection on all 5 architectures."""
    dataset = load_truthfulqa_dataset()
    logger.info(f"Loaded {len(dataset)} TruthfulQA samples")

    all_results = {}

    for model_name in CROSS_ARCH_MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model_name}")
        logger.info(f"{'='*60}")

        try:
            model, tokenizer = load_model(model_name)
            result = run_fep_detection_for_model(model, tokenizer, model_name, dataset, max_samples=None)
            all_results[model_name] = result

            # Save per-model
            safe_name = model_name.replace("/", "_")
            output_path = RESULTS_DIR / f"fep_{safe_name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved: {output_path}")

            # Free memory
            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed: {model_name} — {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-architecture comparison
    comparison = compare_architectures(all_results)
    comp_path = RESULTS_DIR / "cross_architecture_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison saved: {comp_path}")

    # Print summary table
    logger.info("\n" + "="*70)
    logger.info(f"{'Model':<30} {'Layers':<8} {'Late Crystal%':<15} {'FEP Depth%':<12}")
    logger.info("-"*70)
    for name, data in comparison.items():
        short = name.split("/")[-1]
        logger.info(f"{short:<30} {data['n_layers']:<8} {data['late_crystallization_pct']:<15} {data['mean_fep_depth']:<12}")
    logger.info("="*70)
    logger.info("\n=== CROSS-ARCHITECTURE FEP COMPLETE ===")


if __name__ == "__main__":
    main()
