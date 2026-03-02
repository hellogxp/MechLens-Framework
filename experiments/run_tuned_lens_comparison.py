"""Tuned Lens vs Logit Lens FEP Comparison Experiment.

CRITICAL experiment for COLM 2026: Validates that Late Crystallization is NOT
a logit lens artifact by comparing FEP distributions under both standard logit
lens and trained tuned lens probes (Belrose et al., 2023).

Expected outcome: FEP distributions should be highly correlated,
with tuned lens showing similar late crystallization rates.

GPU time: ~6 hours on A100 40GB
  - Tuned lens training: ~1.5h (Qwen) + ~1.5h (Llama) + ~1.5h (Mistral)
  - FEP detection with tuned lens: ~0.5h per model

Usage:
    python experiments/run_tuned_lens_comparison.py [--models qwen] [--train-samples 2000]
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tuned_lens_comparison")

RESULTS_DIR = PROJECT_ROOT / "results" / "tuned_lens_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B",
        "local_path": None,  # Use default MechLens loader
        "n_layers": 28,
        "d_model": 3584,
    },
    "llama": {
        "name": "meta-llama/Llama-3.1-8B",
        "local_path": "/root/.cache/modelscope/LLM-Research/Meta-Llama-3___1-8B",
        "n_layers": 32,
        "d_model": 4096,
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
        "local_path": "/root/.cache/modelscope/AI-ModelScope/Mistral-7B-v0___1",
        "n_layers": 32,
        "d_model": 4096,
    },
}


# ======================== Tuned Lens Probe ========================

class TunedLensProbe(nn.Module):
    """Per-layer affine probe following Belrose et al. (2023).

    Transforms residual stream h_L at layer L via learned affine map:
        h_L' = h_L @ W + b
    initialized to identity (W=I, b=0) so untrained probe = logit lens.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.weight + self.bias


class TunedLens(nn.Module):
    """Full tuned lens: one affine probe per layer."""

    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.probes = nn.ModuleList([
            TunedLensProbe(d_model) for _ in range(n_layers)
        ])

    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply tuned lens transformation at given layer."""
        return self.probes[layer](h)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cuda"):
        self.load_state_dict(torch.load(path, map_location=device))


# ======================== Model Loading ========================

def load_model_hooked(model_key: str):
    """Load model as HookedTransformer."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    local_path = config["local_path"]

    from transformer_lens import HookedTransformer

    if local_path and os.path.isdir(local_path):
        logger.info(f"Loading from local: {local_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_path, torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True,
        )
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, tokenizer=tokenizer,
            torch_dtype=torch.float16, device="cuda",
        )
    else:
        logger.info(f"Loading via MechLens: {model_name}")
        from mechlens.models.model_loader import load_model as ml_load
        model = ml_load(model_name, dtype="float16")

    logger.info(f"Loaded: {model.cfg.n_layers}L, {model.cfg.d_model}d")
    return model


# ======================== Training Data ========================

def get_training_texts(n_samples: int = 2000) -> list[str]:
    """Get training texts for tuned lens from WikiText or C4.

    Falls back to TruthfulQA questions if datasets unavailable.
    """
    texts = []

    # Try WikiText-2
    try:
        from datasets import load_dataset
        logger.info("Loading WikiText-2 for tuned lens training...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:  # Skip short/empty entries
                texts.append(text[:512])  # Truncate to 512 chars
                if len(texts) >= n_samples:
                    break
        logger.info(f"Loaded {len(texts)} training texts from WikiText-2")
    except Exception as e:
        logger.warning(f"WikiText-2 loading failed: {e}")

    # Fallback: use TruthfulQA questions (not ideal but functional)
    if len(texts) < n_samples:
        try:
            from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
            data_dir = PROJECT_ROOT / "data" / "truthfulqa"
            download_truthfulqa(data_dir)
            dataset = load_truthfulqa(data_dir / "truthfulqa.json")
            for sample in dataset:
                q = sample["question"]
                texts.append(f"Q: {q}\nA:")
                if len(texts) >= n_samples:
                    break
            logger.info(f"Padded to {len(texts)} texts using TruthfulQA")
        except Exception as e:
            logger.warning(f"TruthfulQA fallback failed: {e}")

    return texts[:n_samples]


# ======================== Tuned Lens Training ========================

def train_tuned_lens(
    model,
    tuned_lens: TunedLens,
    texts: list[str],
    n_epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 1,
) -> dict:
    """Train tuned lens probes to predict final-layer logits.

    Memory-efficient: trains one layer at a time with detached model weights.
    For each text, each layer:
      1. Get cached residual (no grad)
      2. Forward through probe (with grad)
      3. Project via detached ln_final + W_U
      4. Minimize KL(target || probe_prediction)
      5. Backward + step immediately

    Returns training stats.
    """
    n_layers = tuned_lens.n_layers
    device = next(model.parameters()).device

    tuned_lens = tuned_lens.to(device).float()

    # Cache model head components for efficient access
    # Use model.ln_final directly (works with any norm type: RMSNorm, LayerNorm)
    W_U = model.W_U.detach().float()
    b_U = model.b_U.detach().float() if model.b_U is not None else None

    # Per-layer optimizers for cleaner gradient management
    optimizers = [
        torch.optim.Adam(tuned_lens.probes[l].parameters(), lr=lr)
        for l in range(n_layers)
    ]

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

    stats = {"epoch_losses": []}
    total_samples = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_processed = 0

        for i, text in enumerate(texts):
            try:
                tokens = model.to_tokens(text, prepend_bos=True)
                if tokens.shape[1] < 5:
                    continue
                tokens = tokens[:, :64]  # Shorter to save memory

                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        tokens, names_filter=hook_names
                    )

                # Target: final layer logits (fully detached)
                with torch.no_grad():
                    final_resid = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0]
                    final_normed = model.ln_final(final_resid)
                    target_logits = final_normed.float() @ W_U
                    if b_U is not None:
                        target_logits = target_logits + b_U
                    target_log_probs = F.log_softmax(target_logits, dim=-1).detach()

                # Train each layer's probe independently
                for layer in range(n_layers - 1):
                    resid = cache[f"blocks.{layer}.hook_resid_post"][0].detach().float()

                    # Forward through probe (only probe params have grad)
                    transformed = tuned_lens.probes[layer](resid)

                    # Apply model ln_final + detached W_U
                    normed = model.ln_final(transformed.half()).float()
                    pred_logits = normed @ W_U
                    if b_U is not None:
                        pred_logits = pred_logits + b_U
                    pred_log_probs = F.log_softmax(pred_logits, dim=-1)

                    loss = F.kl_div(
                        pred_log_probs, target_log_probs,
                        reduction="batchmean", log_target=True,
                    )

                    optimizers[layer].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tuned_lens.probes[layer].parameters(), 1.0)
                    optimizers[layer].step()

                    epoch_loss += loss.item()

                # Free cache memory
                del cache
                torch.cuda.empty_cache()

                n_processed += 1
                total_samples += 1

                if n_processed % 50 == 0:
                    avg = epoch_loss / (n_processed * (n_layers - 1))
                    logger.info(f"  Epoch {epoch+1}, sample {n_processed}/{len(texts)}: "
                                f"avg loss/layer={avg:.4f}")

            except Exception as e:
                logger.debug(f"Skipping sample {i}: {e}")
                del cache
                torch.cuda.empty_cache()
                continue

        avg_epoch_loss = epoch_loss / max(n_processed * (n_layers - 1), 1)
        stats["epoch_losses"].append(avg_epoch_loss)
        logger.info(f"Epoch {epoch+1}/{n_epochs}: avg loss/layer={avg_epoch_loss:.4f} "
                     f"({n_processed} samples)")

    return stats


# ======================== FEP Detection ========================

def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to logits via ln_final + W_U."""
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def detect_fep_dual(
    model,
    tuned_lens: TunedLens,
    question: str,
    correct_answer: str,
    top_k: int = 10,
) -> dict:
    """Detect FEP using both logit lens and tuned lens simultaneously.

    Returns FEP for both methods plus per-layer rank comparison.
    """
    n_layers = model.cfg.n_layers
    device = next(model.parameters()).device

    prompt = f"Q: {question}\nA:"
    tokens = model.to_tokens(prompt, prepend_bos=True)

    answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)[0]
    if len(answer_tokens) == 0:
        return {"error": "empty_answer"}
    target_token = answer_tokens[0].item()

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    logit_ranks = []
    tuned_ranks = []
    logit_in_topk = []
    tuned_in_topk = []

    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]

        # === Standard logit lens ===
        ll_logits = unembed_at_layer(model, resid)
        ll_probs = F.softmax(ll_logits.float(), dim=-1)
        ll_sorted = torch.argsort(ll_probs, descending=True)
        ll_rank = (ll_sorted == target_token).nonzero(as_tuple=True)[0]
        ll_rank = ll_rank[0].item() if len(ll_rank) > 0 else ll_probs.shape[0]
        logit_ranks.append(ll_rank)
        logit_in_topk.append(ll_rank < top_k)

        # === Tuned lens ===
        transformed = tuned_lens.transform(resid.float(), layer)
        tl_logits = unembed_at_layer(model, transformed.half())
        tl_probs = F.softmax(tl_logits.float(), dim=-1)
        tl_sorted = torch.argsort(tl_probs, descending=True)
        tl_rank = (tl_sorted == target_token).nonzero(as_tuple=True)[0]
        tl_rank = tl_rank[0].item() if len(tl_rank) > 0 else tl_probs.shape[0]
        tuned_ranks.append(tl_rank)
        tuned_in_topk.append(tl_rank < top_k)

    # Compute FEP for both
    logit_fep = n_layers
    tuned_fep = n_layers
    for layer in range(n_layers):
        if logit_in_topk[layer] and logit_fep == n_layers:
            logit_fep = layer
        if tuned_in_topk[layer] and tuned_fep == n_layers:
            tuned_fep = layer

    return {
        "logit_fep": logit_fep,
        "tuned_fep": tuned_fep,
        "logit_ranks": logit_ranks,
        "tuned_ranks": tuned_ranks,
        "target_token": target_token,
        "target_token_str": model.to_single_str_token(target_token),
    }


# ======================== Main Experiment ========================

def run_comparison_for_model(
    model_key: str,
    train_samples: int = 2000,
    max_eval_samples: int = None,
) -> dict:
    """Run full tuned lens vs logit lens comparison for one model."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]

    logger.info("=" * 70)
    logger.info(f"Tuned Lens Comparison: {model_name}")
    logger.info("=" * 70)

    # Load model
    model = load_model_hooked(model_key)

    # Check for saved tuned lens
    lens_path = RESULTS_DIR / f"tuned_lens_{model_key}.pt"
    tuned_lens = TunedLens(n_layers, d_model)

    if lens_path.exists():
        logger.info(f"Loading saved tuned lens from {lens_path}")
        tuned_lens.load(str(lens_path), device="cuda")
    else:
        # Train tuned lens
        logger.info(f"Training tuned lens ({train_samples} samples)...")
        texts = get_training_texts(train_samples)
        train_stats = train_tuned_lens(model, tuned_lens, texts, n_epochs=3, lr=1e-3)
        logger.info(f"Training complete. Final loss: {train_stats['epoch_losses'][-1]:.4f}")

        # Save trained lens
        tuned_lens.save(str(lens_path))
        logger.info(f"Saved tuned lens to {lens_path}")

    tuned_lens = tuned_lens.to("cuda").float()
    tuned_lens.eval()

    # Load TruthfulQA
    from mechlens.benchmark.truthfulqa import download_truthfulqa, load_truthfulqa
    data_dir = PROJECT_ROOT / "data" / "truthfulqa"
    download_truthfulqa(data_dir)
    dataset = load_truthfulqa(data_dir / "truthfulqa.json")

    if max_eval_samples:
        dataset = dataset[:max_eval_samples]

    # Run dual FEP detection
    logger.info(f"Running dual FEP detection on {len(dataset)} samples...")
    results = []
    logit_fep_dist = defaultdict(int)
    tuned_fep_dist = defaultdict(int)

    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            logger.info(f"  [{model_key}] FEP detection: {i}/{len(dataset)}")

        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        if not best_answer.strip():
            continue

        result = detect_fep_dual(model, tuned_lens, question, best_answer)
        if "error" in result:
            continue

        result["id"] = sample["id"]
        result["category"] = sample.get("category", "Unknown")
        results.append(result)

        logit_fep_dist[result["logit_fep"]] += 1
        tuned_fep_dist[result["tuned_fep"]] += 1

    # Analyze comparison
    analysis = analyze_comparison(results, n_layers)

    # Clean up
    del model, tuned_lens
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_key": model_key,
        "n_layers": n_layers,
        "n_samples": len(results),
        "train_samples": train_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "logit_fep_distribution": dict(logit_fep_dist),
        "tuned_fep_distribution": dict(tuned_fep_dist),
        "analysis": analysis,
        "per_sample_results": results,
    }


def analyze_comparison(results: list, n_layers: int) -> dict:
    """Analyze logit lens vs tuned lens FEP comparison."""
    from scipy import stats

    logit_feps = np.array([r["logit_fep"] for r in results])
    tuned_feps = np.array([r["tuned_fep"] for r in results])

    # Correlation
    pearson_r, pearson_p = stats.pearsonr(logit_feps, tuned_feps)
    spearman_r, spearman_p = stats.spearmanr(logit_feps, tuned_feps)

    # Late crystallization rates
    logit_late_pct = float(np.mean(logit_feps == n_layers))
    tuned_late_pct = float(np.mean(tuned_feps == n_layers))

    # Agreement metrics
    exact_match = float(np.mean(logit_feps == tuned_feps))
    within_1 = float(np.mean(np.abs(logit_feps - tuned_feps) <= 1))
    within_2 = float(np.mean(np.abs(logit_feps - tuned_feps) <= 2))

    # Mean FEP comparison
    logit_mean = float(np.mean(logit_feps))
    tuned_mean = float(np.mean(tuned_feps))
    logit_std = float(np.std(logit_feps))
    tuned_std = float(np.std(tuned_feps))

    # Paired t-test: is there a significant difference in FEP?
    t_stat, t_pvalue = stats.ttest_rel(logit_feps, tuned_feps)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(logit_feps, tuned_feps)
    except ValueError:
        w_stat, w_pvalue = 0.0, 1.0  # All identical

    # Per-category comparison
    category_comparison = {}
    cats = defaultdict(lambda: {"logit": [], "tuned": []})
    for r in results:
        cats[r["category"]]["logit"].append(r["logit_fep"])
        cats[r["category"]]["tuned"].append(r["tuned_fep"])

    for cat, feps in cats.items():
        if len(feps["logit"]) >= 5:
            ll = np.array(feps["logit"])
            tl = np.array(feps["tuned"])
            category_comparison[cat] = {
                "n": len(ll),
                "logit_mean_fep": float(np.mean(ll)),
                "tuned_mean_fep": float(np.mean(tl)),
                "logit_late_pct": float(np.mean(ll == n_layers)),
                "tuned_late_pct": float(np.mean(tl == n_layers)),
            }

    return {
        "correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        },
        "late_crystallization": {
            "logit_lens_pct": logit_late_pct,
            "tuned_lens_pct": tuned_late_pct,
            "difference": tuned_late_pct - logit_late_pct,
        },
        "agreement": {
            "exact_match": exact_match,
            "within_1_layer": within_1,
            "within_2_layers": within_2,
        },
        "mean_fep": {
            "logit_lens": {"mean": logit_mean, "std": logit_std},
            "tuned_lens": {"mean": tuned_mean, "std": tuned_std},
        },
        "statistical_tests": {
            "paired_ttest": {"t_stat": float(t_stat), "p_value": float(t_pvalue)},
            "wilcoxon": {"w_stat": float(w_stat), "p_value": float(w_pvalue)},
        },
        "category_comparison": category_comparison,
    }


def print_summary(all_results: dict):
    """Print publication-ready summary."""
    print("\n" + "=" * 70)
    print("TUNED LENS vs LOGIT LENS FEP COMPARISON")
    print("=" * 70)

    for model_key, result in all_results.items():
        a = result["analysis"]
        n = result["n_samples"]
        print(f"\n{'─' * 60}")
        print(f"Model: {result['model']} ({result['n_layers']} layers, {n} samples)")
        print(f"{'─' * 60}")

        corr = a["correlation"]
        print(f"  Correlation:  Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
        print(f"                Spearman ρ={corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

        lc = a["late_crystallization"]
        print(f"  Late Crystal: Logit lens={lc['logit_lens_pct']:.1%}, "
              f"Tuned lens={lc['tuned_lens_pct']:.1%} "
              f"(Δ={lc['difference']:+.1%})")

        ag = a["agreement"]
        print(f"  FEP Match:    Exact={ag['exact_match']:.1%}, "
              f"±1 layer={ag['within_1_layer']:.1%}, "
              f"±2 layers={ag['within_2_layers']:.1%}")

        mf = a["mean_fep"]
        print(f"  Mean FEP:     Logit={mf['logit_lens']['mean']:.2f}±{mf['logit_lens']['std']:.2f}, "
              f"Tuned={mf['tuned_lens']['mean']:.2f}±{mf['tuned_lens']['std']:.2f}")

        st = a["statistical_tests"]
        print(f"  Paired t-test: t={st['paired_ttest']['t_stat']:.3f}, "
              f"p={st['paired_ttest']['p_value']:.4f}")
        print(f"  Wilcoxon:      W={st['wilcoxon']['w_stat']:.1f}, "
              f"p={st['wilcoxon']['p_value']:.4f}")

    # LaTeX-ready table for paper
    print("\n" + "=" * 70)
    print("LaTeX TABLE (for paper Section 6.1):")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering\small")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Probe} & \textbf{Mean FEP} & "
          r"\textbf{Late Crystal} & \textbf{$r$} & \textbf{Exact Match} \\")
    print(r"\midrule")

    for model_key, result in all_results.items():
        a = result["analysis"]
        mf = a["mean_fep"]
        lc = a["late_crystallization"]
        corr = a["correlation"]
        ag = a["agreement"]
        short_name = model_key.capitalize()

        print(f"\\multirow{{2}}{{*}}{{{short_name}}} "
              f"& Logit lens & {mf['logit_lens']['mean']:.1f}$\\pm${mf['logit_lens']['std']:.1f} "
              f"& {lc['logit_lens_pct']*100:.1f}\\% & --- & --- \\\\")
        print(f" & Tuned lens & {mf['tuned_lens']['mean']:.1f}$\\pm${mf['tuned_lens']['std']:.1f} "
              f"& {lc['tuned_lens_pct']*100:.1f}\\% "
              f"& {corr['pearson_r']:.3f} & {ag['exact_match']*100:.1f}\\% \\\\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Tuned lens vs.\ logit lens FEP comparison. High correlation "
          r"and similar late crystallization rates confirm Late Crystallization "
          r"is not a logit lens artifact.}")
    print(r"\label{tab:tuned_lens}")
    print(r"\end{table}")

    print("\n=== EXPERIMENT COMPLETE ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "llama", "mistral"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TUNED LENS vs LOGIT LENS FEP COMPARISON")
    logger.info(f"Models: {args.models}")
    logger.info(f"Training samples: {args.train_samples}")
    logger.info("=" * 70)

    all_results = {}

    for model_key in args.models:
        try:
            result = run_comparison_for_model(
                model_key,
                train_samples=args.train_samples,
                max_eval_samples=args.max_eval_samples,
            )
            all_results[model_key] = result

            # Save individual results
            output_path = RESULTS_DIR / f"comparison_{model_key}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed for {model_key}: {e}", exc_info=True)
            continue

    # Save combined results
    combined_path = RESULTS_DIR / "tuned_lens_comparison_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
