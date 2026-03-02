"""14B Scale Validation Experiment.

Tests whether Late Crystallization persists at larger model scale (14B parameters).
Uses Qwen2.5-14B to directly compare with Qwen2.5-7B FEP distributions.

GPU time: ~4 hours on A100 40GB (may require int8 quantization for 14B)

Usage:
    python experiments/run_14b_scale_validation.py [--max-samples 817] [--quantize]
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
logger = logging.getLogger("scale_14b")

RESULTS_DIR = PROJECT_ROOT / "results" / "scale_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 7B baseline from paper
BASELINE_7B = {
    "model": "Qwen/Qwen2.5-7B",
    "n_layers": 28,
    "mean_fep": 27.3,
    "std_fep": 1.8,
    "late_crystal_pct": 0.859,
    "fep_depth": 27.3 / 28,
}

# 14B model config
MODEL_14B = {
    "name": "Qwen/Qwen2.5-14B",
    "local_path": None,
    "n_layers": 40,
    "d_model": 5120,
    "n_heads": 40,
}


# ======================== Model Loading ========================

def load_14b_model(quantize: bool = False):
    """Load Qwen2.5-14B model.

    On A100 40GB:
    - fp16: ~28GB VRAM (tight but feasible with cache management)
    - int8: ~14GB VRAM (safe, with minor quality impact)
    """
    model_name = MODEL_14B["name"]

    from transformer_lens import HookedTransformer

    if quantize:
        logger.info(f"Loading {model_name} with int8 quantization...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, tokenizer=tokenizer,
            device="cuda",
        )
    else:
        logger.info(f"Loading {model_name} in fp16...")
        try:
            from mechlens.models.model_loader import load_model as ml_load
            model = ml_load(model_name, dtype="float16")
        except Exception as e:
            logger.warning(f"MechLens loader failed: {e}, trying direct load...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = HookedTransformer.from_pretrained(
                model_name, hf_model=hf_model, tokenizer=tokenizer,
                torch_dtype=torch.float16, device="cuda",
            )

    n_layers = model.cfg.n_layers
    logger.info(f"Loaded: {n_layers}L, d_model={model.cfg.d_model}")
    return model


# ======================== FEP Detection ========================

def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def detect_fep(model, question: str, correct_answer: str, top_k: int = 10) -> dict:
    """Standard FEP detection (same as other scripts)."""
    n_layers = model.cfg.n_layers

    prompt = f"Q: {question}\nA:"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 128:
        tokens = tokens[:, :128]

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

def analyze_scale_comparison(results: list, n_layers: int) -> dict:
    """Analyze 14B FEP and compare with 7B baseline."""
    from scipy import stats

    feps = np.array([r["fep_layer"] for r in results])

    stats_14b = {
        "n_samples": len(feps),
        "mean_fep": float(np.mean(feps)),
        "std_fep": float(np.std(feps)),
        "late_crystal_pct": float(np.mean(feps == n_layers)),
        "fep_depth": float(np.mean(feps)) / n_layers,
    }

    # Scale comparison (normalized by depth)
    comparison = {
        "7B": {
            "n_layers": BASELINE_7B["n_layers"],
            "mean_fep": BASELINE_7B["mean_fep"],
            "fep_depth": BASELINE_7B["fep_depth"],
            "late_crystal_pct": BASELINE_7B["late_crystal_pct"],
        },
        "14B": {
            "n_layers": n_layers,
            "mean_fep": stats_14b["mean_fep"],
            "fep_depth": stats_14b["fep_depth"],
            "late_crystal_pct": stats_14b["late_crystal_pct"],
        },
    }

    # Key question: does crystallization persist at larger scale?
    comparison["crystallization_persists"] = stats_14b["late_crystal_pct"] > 0.5
    comparison["depth_ratio_preserved"] = abs(
        stats_14b["fep_depth"] - BASELINE_7B["fep_depth"]
    ) < 0.1  # Within 10% relative depth

    # Per-category for spectrum analysis
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
                "std_fep": float(np.std(c)),
                "late_crystal_pct": float(np.mean(c == n_layers)),
                "fep_depth": float(np.mean(c)) / n_layers,
            }

    return {
        "stats_14b": stats_14b,
        "scale_comparison": comparison,
        "category_stats": category_stats,
    }


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--quantize", action="store_true",
                        help="Use int8 quantization (saves VRAM)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("14B SCALE VALIDATION: Late Crystallization at Scale")
    logger.info(f"Model: {MODEL_14B['name']} ({MODEL_14B['n_layers']} layers)")
    logger.info(f"Quantize: {args.quantize}")
    logger.info("=" * 70)

    # Load model
    model = load_14b_model(quantize=args.quantize)
    n_layers = model.cfg.n_layers

    # Load TruthfulQA
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    # Run FEP detection
    logger.info(f"Running FEP detection on {len(dataset)} samples...")
    results = []
    fep_dist = defaultdict(int)

    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            logger.info(f"  FEP detection: {i}/{len(dataset)}")

        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        if not best_answer.strip():
            continue

        fep_result = detect_fep(model, question, best_answer)
        if "error" in fep_result:
            continue

        fep_result["id"] = sample["id"]
        fep_result["category"] = sample.get("category", "Unknown")
        results.append(fep_result)
        fep_dist[fep_result["fep_layer"]] += 1

    # Analyze
    analysis = analyze_scale_comparison(results, n_layers)

    output = {
        "model": MODEL_14B["name"],
        "n_layers": n_layers,
        "n_samples": len(results),
        "quantized": args.quantize,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fep_distribution": dict(fep_dist),
        "analysis": analysis,
        "per_sample_results": results,
    }

    output_path = RESULTS_DIR / "scale_14b_fep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Saved to {output_path}")

    # Print summary
    s = analysis["stats_14b"]
    comp = analysis["scale_comparison"]
    print(f"\n{'=' * 60}")
    print("14B SCALE VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Qwen2.5-14B ({n_layers} layers, {len(results)} samples):")
    print(f"  Mean FEP: {s['mean_fep']:.1f} ± {s['std_fep']:.1f}")
    print(f"  FEP Depth: {s['fep_depth']:.1%}")
    print(f"  Late Crystallization: {s['late_crystal_pct']:.1%}")
    print(f"\nComparison with 7B:")
    b7 = comp["7B"]
    print(f"  7B:  FEP Depth={b7['fep_depth']:.1%}, Late Crystal={b7['late_crystal_pct']:.1%}")
    print(f"  14B: FEP Depth={comp['14B']['fep_depth']:.1%}, "
          f"Late Crystal={comp['14B']['late_crystal_pct']:.1%}")
    persist = "YES" if comp["crystallization_persists"] else "NO"
    depth = "YES" if comp["depth_ratio_preserved"] else "NO"
    print(f"\n  Crystallization persists: {persist}")
    print(f"  Depth ratio preserved: {depth}")

    del model
    torch.cuda.empty_cache()
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    main()
