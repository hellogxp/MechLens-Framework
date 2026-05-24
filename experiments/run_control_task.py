#!/usr/bin/env python3
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""Control Task FEP — EMNLP Extended (5 models, SST-2 + NLI).

Uses HuggingFace models directly with device_map="auto" for multi-GPU support.
Confirms Late Crystallization is specific to factual knowledge retrieval.

Expected results:
- Sentiment (SST-2): very LOW late crystallization (< 10%)
- NLI (entailment/contradiction): LOW late crystallization (< 15%)
- Factual (TruthfulQA): HIGH late crystallization (27-86%)
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
logger = logging.getLogger("control_task_emnlp")

RESULTS_DIR = PROJECT_ROOT / "results" / "control_task_emnlp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Loaded: {model_name} — {model.config.num_hidden_layers} layers")
    return model, tokenizer


def get_model_components(model, model_name: str):
    """Get architecture-specific final norm and lm_head."""
    if "pythia" in model_name.lower():
        return model.gpt_neox.final_layer_norm, model.embed_out
    else:
        return model.model.norm, model.lm_head


def unembed(norm_module, lm_head_module, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply final norm + lm_head."""
    target_device = next(norm_module.parameters()).device
    h = hidden_state.to(target_device).to(torch.float16)
    normed = norm_module(h)
    head_device = next(lm_head_module.parameters()).device
    normed = normed.to(head_device)
    return lm_head_module(normed)


# ===== SST-2 SENTIMENT DATA =====
def get_sst2_samples(max_samples: int = 200):
    """Get SST-2 samples. Try datasets library, fallback to built-in."""
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        samples = []
        for item in ds:
            if len(samples) >= max_samples:
                break
            label = "positive" if item["label"] == 1 else "negative"
            samples.append({"sentence": item["sentence"], "label": label})
        return samples
    except Exception:
        logger.warning("Cannot load SST-2 from HF, using built-in samples")
        return _builtin_sst2()


def _builtin_sst2():
    """Minimal built-in SST-2 samples for fallback."""
    pos = [
        "a stirring , funny and finally transporting re-imagining of beauty and the beast",
        "the film provides a great deal of insight into the psychology",
        "a masterpiece four years in the making",
        "a beautiful , moving story that draws its power from the quiet",
        "one of the best films of the year",
        "an intelligent , multi-layered film that builds tension",
        "surprisingly poignant and emotionally resonant",
        "delivers a powerful message with incredible artistry",
        "unforgettable performances by the entire cast",
        "a triumph of storytelling and visual beauty",
    ]
    neg = [
        "unflinchingly bleak and desperate",
        "a complete waste of time",
        "the movie is so bad it made me want to leave the theater",
        "lacks substance , depth , and any reason to watch",
        "painfully slow and uninteresting",
        "one of the worst films i have ever seen",
        "tedious and predictable from beginning to end",
        "fails on every conceivable level",
        "an insult to the audience intelligence",
        "boring , pointless , and utterly forgettable",
    ]
    samples = []
    for s in pos:
        samples.append({"sentence": s, "label": "positive"})
    for s in neg:
        samples.append({"sentence": s, "label": "negative"})
    return samples


# ===== NLI DATA =====
def get_nli_samples(max_samples: int = 200):
    """Get NLI samples from MNLI dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "mnli", split="validation_matched")
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        samples = []
        for item in ds:
            if len(samples) >= max_samples:
                break
            samples.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": label_map[item["label"]],
            })
        return samples
    except Exception:
        logger.warning("Cannot load MNLI from HF, using built-in samples")
        return _builtin_nli()


def _builtin_nli():
    """Minimal built-in NLI samples."""
    return [
        {"premise": "A man is playing guitar.", "hypothesis": "A person is making music.", "label": "entailment"},
        {"premise": "A dog is running in a park.", "hypothesis": "A cat is sleeping.", "label": "contradiction"},
        {"premise": "The woman is cooking dinner.", "hypothesis": "Someone is in the kitchen.", "label": "entailment"},
        {"premise": "Children are playing outside.", "hypothesis": "It is a sunny day.", "label": "neutral"},
        {"premise": "The store is closed.", "hypothesis": "People are shopping inside.", "label": "contradiction"},
        {"premise": "She is reading a book on the bench.", "hypothesis": "A woman is sitting.", "label": "entailment"},
        {"premise": "The car is parked in the driveway.", "hypothesis": "Someone drove home.", "label": "neutral"},
        {"premise": "It is raining heavily.", "hypothesis": "The sun is shining brightly.", "label": "contradiction"},
        {"premise": "The teacher is writing on the board.", "hypothesis": "A class is in session.", "label": "entailment"},
        {"premise": "He finished his lunch early.", "hypothesis": "He is still eating.", "label": "contradiction"},
    ]


def detect_fep_control(model, tokenizer, model_name, prompt: str, target_answer: str, top_k: int = 10) -> dict:
    """Detect FEP for a control task sample using HF output_hidden_states."""
    norm_module, lm_head_module = get_model_components(model, model_name)
    n_layers = model.config.num_hidden_layers

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    answer_tokens = tokenizer.encode(target_answer, add_special_tokens=False)
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    fep_layer = None
    layer_ranks = []

    for l in range(n_layers):
        h = hidden_states[l + 1][0, -1, :]
        logits = unembed(norm_module, lm_head_module, h)
        sorted_indices = logits.argsort(descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0]
        rank_val = rank[0].item() if len(rank) > 0 else logits.shape[-1]
        layer_ranks.append(rank_val)

        if rank_val < top_k and fep_layer is None:
            fep_layer = l

    if fep_layer is None:
        fep_layer = n_layers

    return {
        "fep_layer": fep_layer,
        "n_layers": n_layers,
        "fep_depth": fep_layer / n_layers,
        "late_crystallization": fep_layer == n_layers,
        "final_rank": layer_ranks[-1],
    }


def run_sst2_control(model, tokenizer, model_name: str, max_samples: int = 200) -> dict:
    """Run SST-2 control task."""
    samples = get_sst2_samples(max_samples)
    logger.info(f"  SST-2: {len(samples)} samples")

    results = []
    for idx, sample in enumerate(samples):
        prompt = f"The sentiment of \"{sample['sentence']}\" is:"
        target = " " + sample["label"]

        try:
            torch.cuda.empty_cache()
            r = detect_fep_control(model, tokenizer, model_name, prompt, target)
            if "error" not in r:
                results.append(r)
        except Exception:
            continue

    if not results:
        return {"task": "sst2", "error": "no_valid_results"}

    late_crystal = sum(1 for r in results if r["late_crystallization"])
    return {
        "task": "sst2",
        "model": model_name,
        "n_samples": len(results),
        "late_crystallization_pct": round(late_crystal / len(results) * 100, 1),
        "mean_fep_depth": round(np.mean([r["fep_depth"] for r in results]) * 100, 1),
    }


def run_nli_control(model, tokenizer, model_name: str, max_samples: int = 200) -> dict:
    """Run NLI control task."""
    samples = get_nli_samples(max_samples)
    logger.info(f"  NLI: {len(samples)} samples")

    results = []
    for idx, sample in enumerate(samples):
        prompt = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}\nRelationship:"
        target = " " + sample["label"]

        try:
            torch.cuda.empty_cache()
            r = detect_fep_control(model, tokenizer, model_name, prompt, target)
            if "error" not in r:
                results.append(r)
        except Exception:
            continue

    if not results:
        return {"task": "nli", "error": "no_valid_results"}

    late_crystal = sum(1 for r in results if r["late_crystallization"])
    return {
        "task": "nli",
        "model": model_name,
        "n_samples": len(results),
        "late_crystallization_pct": round(late_crystal / len(results) * 100, 1),
        "mean_fep_depth": round(np.mean([r["fep_depth"] for r in results]) * 100, 1),
    }


def main():
    """Run control tasks on all models."""
    all_results = {}

    for model_name in ALL_MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Control tasks: {model_name}")
        logger.info(f"{'='*60}")

        try:
            model, tokenizer = load_model(model_name)

            sst2_result = run_sst2_control(model, tokenizer, model_name)
            nli_result = run_nli_control(model, tokenizer, model_name)

            all_results[model_name] = {
                "sst2": sst2_result,
                "nli": nli_result,
            }

            # Save per-model
            safe_name = model_name.replace("/", "_")
            with open(RESULTS_DIR / f"control_{safe_name}.json", "w") as f:
                json.dump(all_results[model_name], f, indent=2)

            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed: {model_name} — {e}")
            import traceback
            traceback.print_exc()

    # Summary
    summary_path = RESULTS_DIR / "control_task_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    logger.info("\n" + "="*70)
    logger.info(f"{'Model':<25} {'SST-2 Late%':<15} {'NLI Late%':<15}")
    logger.info("-"*70)
    for name, data in all_results.items():
        short = name.split("/")[-1]
        sst2_pct = data.get("sst2", {}).get("late_crystallization_pct", "N/A")
        nli_pct = data.get("nli", {}).get("late_crystallization_pct", "N/A")
        logger.info(f"{short:<25} {sst2_pct:<15} {nli_pct:<15}")
    logger.info("="*70)
    logger.info("\n=== CONTROL TASK EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
