#!/usr/bin/env python3
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
"""Head-Level Attribution Experiment for Late Crystallization.

Uses HuggingFace models directly with device_map="auto" for multi-GPU support.
Identifies which attention heads at the FEP boundary layer drive the
crystallization transition via zero-ablation.

For each model:
1. Find FEP layer via output_hidden_states (logit lens)
2. At the FEP layer, ablate each head via o_proj input hook
3. Measure the drop in correct-answer rank
4. Rank heads by attribution score
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
logger = logging.getLogger("head_attribution")

RESULTS_DIR = PROJECT_ROOT / "results" / "head_attribution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All 5 models
ALL_MODELS = [
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.1",
    "EleutherAI/pythia-6.9b",
    "google/gemma-7b",
]

LOCAL_MODEL_PATHS = {
    "Qwen/Qwen2.5-7B": "/home/admin/workspace/models_cache/Qwen/Qwen2___5-7B",
    "meta-llama/Llama-3.1-8B": "/home/admin/workspace/models_cache/LLM-Research/Meta-Llama-3___1-8B",
    "mistralai/Mistral-7B-v0.1": "/home/admin/workspace/models_cache/AI-ModelScope/Mistral-7B-v0___1",
    "EleutherAI/pythia-6.9b": "/home/admin/workspace/models_cache/EleutherAI/pythia-6___9b",
    "google/gemma-7b": "/home/admin/workspace/models_cache/AI-ModelScope/gemma-7b",
}


def load_model(model_name: str):
    """Load model with device_map='auto'."""
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
    n_heads = model.config.num_attention_heads
    logger.info(f"Loaded: {model_name} — {n_layers}L, {n_heads}H")
    return model, tokenizer


def load_truthfulqa_dataset():
    """Load TruthfulQA dataset."""
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")
    return dataset


def get_model_components(model, model_name: str):
    """Get architecture-specific components."""
    if "pythia" in model_name.lower():
        layers = model.gpt_neox.layers
        norm_module = model.gpt_neox.final_layer_norm
        lm_head_module = model.embed_out
    else:
        layers = model.model.layers
        norm_module = model.model.norm
        lm_head_module = model.lm_head
    return layers, norm_module, lm_head_module


def get_attn_module(layers, layer_idx, model_name: str):
    """Get the attention module for a specific layer."""
    layer = layers[layer_idx]
    if "pythia" in model_name.lower():
        return layer.attention
    else:
        return layer.self_attn


def get_o_proj(attn_module, model_name: str):
    """Get the output projection module of attention."""
    if "pythia" in model_name.lower():
        return attn_module.dense
    else:
        return attn_module.o_proj


def unembed(norm_module, lm_head_module, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply final norm + lm_head."""
    target_device = next(norm_module.parameters()).device
    h = hidden_state.to(target_device).to(torch.float16)
    normed = norm_module(h)
    head_device = next(lm_head_module.parameters()).device
    normed = normed.to(head_device)
    return lm_head_module(normed)


def get_answer_rank(logits: torch.Tensor, target_token: int) -> int:
    """Get rank of target token (0-indexed)."""
    sorted_indices = logits.argsort(descending=True)
    rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
    if len(rank) == 0:
        return logits.shape[-1]
    return rank[0].item()


def find_fep(model, tokenizer, model_name, question, correct_answer, top_k=10):
    """Find FEP layer and baseline rank using logit lens."""
    _, norm_module, lm_head_module = get_model_components(model, model_name)

    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    answer_tokens = tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(answer_tokens) == 0:
        return None, None, None
    target_token = answer_tokens[0]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    n_layers = model.config.num_hidden_layers
    fep_layer = n_layers

    for l in range(n_layers):
        h = hidden_states[l + 1][0, -1, :]
        logits = unembed(norm_module, lm_head_module, h)
        rank = get_answer_rank(logits, target_token)
        if rank < top_k and fep_layer == n_layers:
            fep_layer = l

    # Baseline: final layer rank
    final_h = hidden_states[n_layers][0, -1, :]
    final_logits = unembed(norm_module, lm_head_module, final_h)
    baseline_rank = get_answer_rank(final_logits, target_token)

    return fep_layer, baseline_rank, target_token


def run_head_ablation(model, tokenizer, model_name, question, attribution_layer, target_token):
    """Ablate each head at attribution_layer and measure rank change."""
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    layers, norm_module, lm_head_module = get_model_components(model, model_name)
    attn_module = get_attn_module(layers, attribution_layer, model_name)
    o_proj = get_o_proj(attn_module, model_name)

    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    head_ranks = []

    for head_idx in range(n_heads):
        # Register pre-hook on o_proj to zero out head_idx's contribution
        def make_hook(h_idx):
            def hook_fn(module, args):
                x = args[0]  # input to o_proj: [batch, seq, n_heads * d_head]
                x_mod = x.clone()
                x_mod[:, :, h_idx * d_head: (h_idx + 1) * d_head] = 0.0
                return (x_mod,) + args[1:]
            return hook_fn

        handle = o_proj.register_forward_pre_hook(make_hook(head_idx))

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        handle.remove()

        # Get rank after ablation
        final_logits = outputs.logits[0, -1, :]
        ablated_rank = get_answer_rank(final_logits, target_token)
        head_ranks.append(ablated_rank)

    return head_ranks


def run_head_attribution_single_sample(
    model, tokenizer, model_name, question, correct_answer, top_k=10
) -> dict:
    """Full head attribution for one sample."""
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    # Step 1: Find FEP
    fep_layer, baseline_rank, target_token = find_fep(
        model, tokenizer, model_name, question, correct_answer, top_k
    )
    if fep_layer is None:
        return {"error": "empty_answer"}

    # Attribution layer = FEP layer (or last layer if never crystallized)
    attribution_layer = fep_layer if fep_layer < n_layers else n_layers - 1

    # Step 2: Ablate each head
    head_ranks = run_head_ablation(
        model, tokenizer, model_name, question, attribution_layer, target_token
    )

    # Compute attribution scores
    head_attributions = []
    for head_idx, ablated_rank in enumerate(head_ranks):
        rank_change = ablated_rank - baseline_rank
        head_attributions.append({
            "head": head_idx,
            "ablated_rank": ablated_rank,
            "rank_change": rank_change,
        })

    head_attributions.sort(key=lambda x: x["rank_change"], reverse=True)

    return {
        "fep_layer": fep_layer,
        "attribution_layer": attribution_layer,
        "baseline_final_rank": baseline_rank,
        "target_token": target_token,
        "head_attributions": head_attributions,
        "top_heads": [h["head"] for h in head_attributions[:5]],
    }


def run_head_attribution_for_model(model, tokenizer, model_name, dataset, max_samples=200):
    """Run head attribution across samples."""
    n_heads = model.config.num_attention_heads
    logger.info(f"Running head attribution: {model_name} ({max_samples} samples, {n_heads} heads)")

    results = []
    samples_to_process = dataset[:max_samples]

    for idx, sample in enumerate(samples_to_process):
        if idx % 20 == 0:
            logger.info(f"  [{model_name}] Sample {idx}/{len(samples_to_process)}")

        question = sample.get("question", "")
        correct_answers = sample.get("correct_answers", [])
        if not correct_answers:
            continue

        try:
            torch.cuda.empty_cache()
            result = run_head_attribution_single_sample(
                model, tokenizer, model_name, question, correct_answers[0]
            )
            if "error" not in result:
                results.append(result)
        except Exception as e:
            logger.warning(f"  Error sample {idx}: {e}")
            continue

    # Aggregate over all samples (not just "correct" ones)
    head_mean_rank_change = np.zeros(n_heads)
    head_freq_top5 = np.zeros(n_heads)
    valid_count = len(results)  # Use all samples

    for r in results:
        for h_info in r["head_attributions"]:
            head_mean_rank_change[h_info["head"]] += h_info["rank_change"]
        for h in r["top_heads"]:
            head_freq_top5[h] += 1

    if valid_count > 0:
        head_mean_rank_change /= valid_count
        head_freq_top5 /= valid_count

    n_critical = max(1, n_heads // 10)
    critical_heads = np.argsort(head_mean_rank_change)[-n_critical:][::-1].tolist()

    return {
        "model_name": model_name,
        "n_layers": model.config.num_hidden_layers,
        "n_heads": n_heads,
        "n_samples": len(results),
        "n_valid": valid_count,
        "critical_heads": critical_heads,
        "head_mean_rank_change": head_mean_rank_change.tolist(),
        "head_frequency_in_top5": head_freq_top5.tolist(),
        "per_sample_results": results,
    }


def main():
    """Main execution."""
    dataset = load_truthfulqa_dataset()
    logger.info(f"Loaded {len(dataset)} TruthfulQA samples")

    all_results = {}
    max_samples = 200

    for model_name in ALL_MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model_name}")
        logger.info(f"{'='*60}")

        try:
            model, tokenizer = load_model(model_name)
            result = run_head_attribution_for_model(
                model, tokenizer, model_name, dataset, max_samples=max_samples
            )
            all_results[model_name] = result

            safe_name = model_name.replace("/", "_")
            output_path = RESULTS_DIR / f"head_attribution_{safe_name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved: {output_path}")

            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed: {model_name} — {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary_path = RESULTS_DIR / "head_attribution_summary.json"
    summary = {}
    for name, res in all_results.items():
        if "error" not in res:
            summary[name] = {
                "n_layers": res["n_layers"],
                "n_heads": res["n_heads"],
                "n_valid_samples": res["n_valid"],
                "critical_heads": res["critical_heads"],
                "top5_head_mean_rank_change": sorted(
                    enumerate(res["head_mean_rank_change"]),
                    key=lambda x: x[1], reverse=True
                )[:5],
            }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved: {summary_path}")
    logger.info("=== HEAD ATTRIBUTION COMPLETE ===")


if __name__ == "__main__":
    main()
