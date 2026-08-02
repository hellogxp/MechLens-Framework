"""Non-factual control-task FEP detection.

Tests whether the observed FEP profile is specific to factual recall or also
appears in a broader class of token-prediction tasks.

Design:
  - Task: SST-2 sentiment analysis (positive/negative classification)
  - Model: Qwen2.5-7B (same as main experiments)
  - Method: Detect FEP for sentiment tokens ("positive"/"negative") 
    at each layer's logit space
  - Prediction: Sentiment tokens should emerge EARLIER than factual tokens,
    because sentiment is a "surface" property, not deep factual recall.

GPU: 2x V100-16GB (device_map="auto")
Estimated time: ~1-2 hours
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("control_fep")

RESULTS_DIR = PROJECT_ROOT / "results" / "control_task_fep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LOCAL = "/mnt/workspace/models/Qwen/Qwen2.5-7B-Instruct"

# NOTE: Using instruct model for direct comparison with instruct FEP pilot data.
# The instruct model has 37.3% late crystallization on factual tasks.
# If sentiment shows LOWER late crystallization → supports task-specificity.

# TruthfulQA FEP reference (from instruct pilot experiment)
TRUTHFULQA_FEP_REFERENCE = {
    "late_crystal_pct": 0.373,   # 37.3% on instruct model (vs 85.9% on base)
    "mean_fep": 25.5,            # Mean FEP layer (out of 28)
    "fep_depth": 0.910,          # Mean FEP / n_layers
    "n_layers": 28,
}


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer(gpu_ids: str = "0,1"):
    """Load model with device_map for multi-GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    model_path = MODEL_LOCAL if os.path.isdir(MODEL_LOCAL) else MODEL_NAME
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {n_layers} layers")
    return model, tokenizer, n_layers


# ============================================================
# SST-2 Dataset Loading
# ============================================================

def load_sst2_dataset(max_samples: int = 500):
    """Load SST-2 sentiment analysis dataset.
    
    Priority: local JSON > HuggingFace > built-in fallback.
    """
    # Try local JSON first
    local_path = PROJECT_ROOT / "data" / "sst2_validation.json"
    if local_path.exists():
        import json
        with open(local_path) as f:
            samples = json.load(f)
        if max_samples and len(samples) > max_samples:
            import random
            random.seed(42)
            samples = random.sample(samples, max_samples)
        logger.info(f"Loaded SST-2: {len(samples)} samples from local JSON")
        return samples
    
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        samples = []
        for item in ds:
            label = "positive" if item["label"] == 1 else "negative"
            samples.append({
                "sentence": item["sentence"],
                "label": label,
                "label_id": item["label"],
            })
        if max_samples and len(samples) > max_samples:
            import random
            random.seed(42)
            samples = random.sample(samples, max_samples)
        logger.info(f"Loaded SST-2: {len(samples)} samples from HuggingFace")
        return samples
    except Exception as e:
        logger.warning(f"HuggingFace datasets not available ({e}), using built-in samples")
        return _get_builtin_sst2_samples()


def _get_builtin_sst2_samples():
    """Small built-in SST-2 samples for fallback."""
    pos_sentences = [
        "a stirring , funny and finally transporting re-imagining of beauty and the beast",
        "unflinchingly bleak and desperate",
        "the film provides some great insight",
        "offers a breath of the fresh air of freedom",
        "a thoughtful , provocative , insistently humanizing film",
    ]
    neg_sentences = [
        "just plain boring",
        "entirely predictable and lacks energy",
        "no surprises , no ## tension , no humor",
        "a waste of time",
        "suffers from the worst case of hand-holding",
    ]
    samples = []
    for s in pos_sentences:
        samples.append({"sentence": s, "label": "positive", "label_id": 1})
    for s in neg_sentences:
        samples.append({"sentence": s, "label": "negative", "label_id": 0})
    return samples


# ============================================================
# FEP Detection for Sentiment Task
# ============================================================

def detect_fep_sentiment(
    model, tokenizer, sentence: str, target_label: str, n_layers: int, top_k: int = 10
) -> dict:
    """Detect FEP for a sentiment classification sample.
    
    Prompt format: "The sentiment of '{sentence}' is: "
    Target token: first token of "positive" or "negative"
    
    FEP = first layer where target token enters top-k in logit space.
    """
    prompt = f"The sentiment of \"{sentence}\" is:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(model.device)
    
    # Get target token (first token of "positive" or "negative")
    target_tokens = tokenizer(f" {target_label}", add_special_tokens=False)["input_ids"]
    if not target_tokens:
        return {"error": "empty_target"}
    target_token_id = target_tokens[0]
    target_token_str = tokenizer.decode([target_token_id])
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # (n_layers+1,) tensors
    
    # Track target token rank at each layer
    layer_ranks = []
    layer_probs = []
    layer_in_topk = []
    
    for layer in range(n_layers):
        # Get hidden state at last position for this layer
        hs = hidden_states[layer + 1][:, -1, :]  # [1, d_model], +1 for embedding
        
        # Apply final layer norm + lm_head to get logits
        normed = model.model.norm(hs)
        logits = model.lm_head(normed)  # [1, vocab_size]
        probs = F.softmax(logits[0].float(), dim=-1)
        
        # Find rank of target token
        sorted_indices = torch.argsort(probs, descending=True)
        rank_mask = (sorted_indices == target_token_id)
        rank = rank_mask.nonzero(as_tuple=True)[0]
        rank = rank[0].item() if len(rank) > 0 else probs.shape[0]
        
        prob = probs[target_token_id].item()
        in_topk = rank < top_k
        
        layer_ranks.append(rank)
        layer_probs.append(prob)
        layer_in_topk.append(in_topk)
    
    # Find FEP: first layer where target enters top-k
    fep_layer = n_layers  # default: never enters
    for layer in range(n_layers):
        if layer_in_topk[layer]:
            fep_layer = layer
            break
    
    return {
        "fep_layer": fep_layer,
        "layer_ranks": layer_ranks,
        "layer_probs": layer_probs,
        "layer_in_topk": layer_in_topk,
        "target_token_id": target_token_id,
        "target_token_str": target_token_str,
        "final_rank": layer_ranks[-1],
        "final_prob": layer_probs[-1],
    }


# ============================================================
# Main Experiment
# ============================================================

def run_control_fep(model, tokenizer, dataset, n_layers, top_k=10):
    """Run FEP detection on all SST-2 samples."""
    results = []
    fep_distribution = defaultdict(int)
    
    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            logger.info(f"  Control FEP: {i}/{len(dataset)}")
        
        fep_result = detect_fep_sentiment(
            model, tokenizer,
            sentence=sample["sentence"],
            target_label=sample["label"],
            n_layers=n_layers,
            top_k=top_k,
        )
        
        if "error" in fep_result:
            continue
        
        entry = {
            "idx": i,
            "sentence": sample["sentence"][:100],
            "label": sample["label"],
            "fep_layer": fep_result["fep_layer"],
            "final_rank": fep_result["final_rank"],
            "final_prob": fep_result["final_prob"],
            "target_token_str": fep_result["target_token_str"],
        }
        results.append(entry)
        fep_distribution[fep_result["fep_layer"]] += 1
    
    return results, dict(fep_distribution)


def analyze_results(results, fep_distribution, n_layers):
    """Analyze and compare with TruthfulQA FEP."""
    fep_values = [r["fep_layer"] for r in results]
    
    if not fep_values:
        return {"error": "no_results"}
    
    mean_fep = np.mean(fep_values)
    std_fep = np.std(fep_values)
    fep_depth = mean_fep / n_layers
    
    # Late crystallization: % that never enter top-k until final layer
    late_crystal_count = sum(1 for f in fep_values if f == n_layers)
    late_crystal_pct = late_crystal_count / len(fep_values)
    
    # Early emergence: % that enter top-k in first half
    early_count = sum(1 for f in fep_values if f < n_layers // 2)
    early_pct = early_count / len(fep_values)
    
    # Mid emergence: % in second half but not final
    mid_count = sum(1 for f in fep_values if n_layers // 2 <= f < n_layers)
    mid_pct = mid_count / len(fep_values)
    
    # Per-label analysis
    pos_feps = [r["fep_layer"] for r in results if r["label"] == "positive"]
    neg_feps = [r["fep_layer"] for r in results if r["label"] == "negative"]
    
    analysis = {
        "n_samples": len(results),
        "mean_fep": float(mean_fep),
        "std_fep": float(std_fep),
        "fep_depth": float(fep_depth),
        "late_crystal_pct": float(late_crystal_pct),
        "early_emergence_pct": float(early_pct),
        "mid_emergence_pct": float(mid_pct),
        "per_label": {
            "positive": {
                "n": len(pos_feps),
                "mean_fep": float(np.mean(pos_feps)) if pos_feps else 0,
                "late_crystal_pct": sum(1 for f in pos_feps if f == n_layers) / max(len(pos_feps), 1),
            },
            "negative": {
                "n": len(neg_feps),
                "mean_fep": float(np.mean(neg_feps)) if neg_feps else 0,
                "late_crystal_pct": sum(1 for f in neg_feps if f == n_layers) / max(len(neg_feps), 1),
            },
        },
        # Comparison with TruthfulQA
        "comparison_with_truthfulqa": {
            "truthfulqa_late_crystal_pct": TRUTHFULQA_FEP_REFERENCE["late_crystal_pct"],
            "truthfulqa_mean_fep": TRUTHFULQA_FEP_REFERENCE["mean_fep"],
            "truthfulqa_fep_depth": TRUTHFULQA_FEP_REFERENCE["fep_depth"],
            "sst2_late_crystal_pct": float(late_crystal_pct),
            "sst2_mean_fep": float(mean_fep),
            "sst2_fep_depth": float(fep_depth),
            "delta_late_crystal": float(late_crystal_pct - TRUTHFULQA_FEP_REFERENCE["late_crystal_pct"]),
            "delta_mean_fep": float(mean_fep - TRUTHFULQA_FEP_REFERENCE["mean_fep"]),
        },
    }
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Control Task FEP - SST-2 Sentiment")
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("CONTROL TASK FEP DETECTION: SST-2 Sentiment Analysis")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Prediction: Sentiment tokens emerge EARLIER than factual tokens")
    logger.info("=" * 70)
    
    # Load model
    model, tokenizer, n_layers = load_model_and_tokenizer(args.gpu_ids)
    
    # Load SST-2
    dataset = load_sst2_dataset(max_samples=args.max_samples)
    
    # Run FEP detection
    logger.info(f"\nRunning FEP detection on {len(dataset)} SST-2 samples...")
    t0 = time.time()
    results, fep_distribution = run_control_fep(
        model, tokenizer, dataset, n_layers, top_k=args.top_k
    )
    elapsed = time.time() - t0
    logger.info(f"FEP detection complete in {elapsed:.1f}s")
    
    # Analyze
    analysis = analyze_results(results, fep_distribution, n_layers)
    
    # Save
    output = {
        "experiment": "control_task_fep_sst2",
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "top_k": args.top_k,
        "n_samples": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": elapsed,
        "fep_distribution": fep_distribution,
        "analysis": analysis,
        "per_sample_results": results,
    }
    
    output_path = RESULTS_DIR / "sst2_fep_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to {output_path}")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("CONTROL TASK FEP - RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model: {MODEL_NAME} ({n_layers} layers)")
    print(f"Task: SST-2 Sentiment Analysis ({len(results)} samples)")
    print(f"\nSST-2 FEP Results:")
    print(f"  Mean FEP:              {analysis['mean_fep']:.1f} / {n_layers}")
    print(f"  FEP Depth:             {analysis['fep_depth']:.3f}")
    print(f"  Late Crystallization:  {analysis['late_crystal_pct']:.1%}")
    print(f"  Early Emergence (<{n_layers//2}): {analysis['early_emergence_pct']:.1%}")
    
    print(f"\nComparison with TruthfulQA (factual knowledge):")
    comp = analysis["comparison_with_truthfulqa"]
    print(f"  {'Metric':<25} {'TruthfulQA':>12} {'SST-2':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Late Crystal %':<25} {comp['truthfulqa_late_crystal_pct']:>11.1%} {comp['sst2_late_crystal_pct']:>11.1%} {comp['delta_late_crystal']:>+11.1%}")
    print(f"  {'Mean FEP':<25} {comp['truthfulqa_mean_fep']:>12.1f} {comp['sst2_mean_fep']:>12.1f} {comp['delta_mean_fep']:>+12.1f}")
    print(f"  {'FEP Depth':<25} {comp['truthfulqa_fep_depth']:>12.3f} {comp['sst2_fep_depth']:>12.3f}")
    
    print(f"\nFEP Distribution (top 5 layers):")
    sorted_dist = sorted(fep_distribution.items(), key=lambda x: x[1], reverse=True)
    for layer, count in sorted_dist[:5]:
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  Layer {layer:2d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Theory validation
    print(f"\n{'=' * 70}")
    print("THEORY VALIDATION")
    print(f"{'=' * 70}")
    if analysis['mean_fep'] < TRUTHFULQA_FEP_REFERENCE['mean_fep']:
        print("  ✓ SST-2 sentiment emerges EARLIER than factual knowledge")
        print("  → Supports: Late Crystallization is FACTUAL-KNOWLEDGE-SPECIFIC")
    else:
        print("  ✗ SST-2 sentiment does NOT emerge earlier")
        print("  → Late Crystallization may be a GENERAL model property")
    
    del model
    torch.cuda.empty_cache()
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
