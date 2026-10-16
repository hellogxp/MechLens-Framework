"""Instruct Model Intervention Comparison - Extended Hyperparameter Grid.

Extended hyperparameter sweep for instruction-tuned model interventions,
investigating whether sub-baseline performance of DoLa and CAA on
instruction-tuned models reflects a hyperparameter transfer issue or a
structural consequence of reshaped representations.

Key design:
  - Uses HuggingFace model with device_map="auto" (splits across 2x V100-16GB)
  - Extended CAA grid: coeff=[0.1,0.3,0.5,1.0,2.0,3.0,5.0,8.0] × top_k=[5,10,15,20]
  - Extended DoLa: dynamic + static (early=0, mid=14, late=24)
  - Saves per-sample results for statistical analysis

GPU: 2x V100-16GB per run (4 parallel runs on 8 GPUs)
Estimated time: ~2-3 hours total
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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
logger = logging.getLogger("instruct_intervention")

RESULTS_DIR = PROJECT_ROOT / "results" / "instruct_intervention"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LOCAL = "/mnt/workspace/models/Qwen/Qwen2.5-7B-Instruct"

# ============================================================
# Extended hyperparameter grids
# ============================================================
CAA_COEFFICIENTS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
CAA_TOP_K = [5, 10, 15, 20]

DOLA_CONFIGS = [
    {"name": "dynamic", "premature": "dynamic"},
    {"name": "static_early", "premature": 0},
    {"name": "static_mid", "premature": 14},
    {"name": "static_late", "premature": 24},
]

# Base model results for comparison
BASE_RESULTS = {
    "mc1_baseline": 0.2215,
    "mc1_dola": 0.2778,
    "mc1_caa": 0.2558,
    "late_crystal_pct": 0.859,
}

INSTRUCT_FEP = {
    "late_crystal_pct": 0.373,
    "mean_fep": 25.5,
}


# ============================================================
# Model Loading (HuggingFace with device_map for multi-GPU)
# ============================================================

def load_model_and_tokenizer(gpu_ids: str = "0,1"):
    """Load model with device_map='auto' for multi-GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    model_path = MODEL_LOCAL if os.path.isdir(MODEL_LOCAL) else MODEL_NAME
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Using GPUs: {gpu_ids}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {n_layers} layers, device_map=auto")
    return model, tokenizer, n_layers


def format_prompt_instruct(question: str) -> str:
    """Format prompt for Qwen instruct model."""
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ============================================================
# Dataset Loading
# ============================================================

def load_truthfulqa():
    """Load TruthfulQA dataset and convert to MC1 format."""
    data_path = PROJECT_ROOT / "data" / "truthfulqa" / "truthfulqa.json"
    if not data_path.exists():
        # Try downloading
        from mechlens.benchmark.truthfulqa import download_truthfulqa
        download_truthfulqa(data_path.parent)

    with open(data_path) as f:
        data = json.load(f)

    samples = data.get("samples", data) if isinstance(data, dict) else data

    # Convert to MC1 format if needed (correct_answers/incorrect_answers -> mc1_targets)
    for sample in samples:
        if "mc1_targets" not in sample:
            correct = sample.get("correct_answers", [])
            incorrect = sample.get("incorrect_answers", [])
            best = sample.get("best_answer", correct[0] if correct else "")
            # MC1: best_answer as correct choice + all incorrect answers
            choices = [best] + incorrect
            labels = [1] + [0] * len(incorrect)
            sample["mc1_targets"] = {"choices": choices, "labels": labels}

    logger.info(f"Loaded TruthfulQA: {len(samples)} samples")
    return samples


# ============================================================
# MC1 Evaluation Core
# ============================================================

def score_choice(model, tokenizer, prompt: str, choice: str) -> float:
    """Score a single choice given a prompt using log-probability."""
    full_text = prompt + " " + choice
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(model.device)

    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    answer_start = prompt_inputs["input_ids"].shape[1]

    if answer_start >= input_ids.shape[1]:
        return float("-inf")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits[0, answer_start - 1:-1].float(), dim=-1)
    answer_ids = input_ids[0, answer_start:]
    token_log_probs = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum().item()


def evaluate_mc1(model, tokenizer, dataset, score_fn=None, desc="baseline"):
    """Evaluate MC1 with optional custom scoring function."""
    correct = 0
    total = 0
    per_sample = []

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  [{desc}] {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        prompt = format_prompt_instruct(question)
        best_score = float("-inf")
        best_idx = 0

        for j, choice in enumerate(choices):
            if score_fn:
                s = score_fn(prompt, choice)
            else:
                s = score_choice(model, tokenizer, prompt, choice)
            if s > best_score:
                best_score = s
                best_idx = j

        is_correct = labels[best_idx] == 1
        if is_correct:
            correct += 1
        total += 1
        per_sample.append({"idx": i, "correct": is_correct, "question": question[:80]})

    mc1 = correct / max(total, 1)
    return {"mc1": mc1, "correct": correct, "total": total, "per_sample": per_sample}


# ============================================================
# DoLa Implementation (using output_hidden_states)
# ============================================================

def evaluate_mc1_dola(model, tokenizer, dataset, n_layers, premature="dynamic"):
    """Evaluate MC1 with DoLa decoding."""
    mature_layer = n_layers - 1
    correct = 0
    total = 0
    per_sample = []

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  [DoLa-{premature}] {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        prompt = format_prompt_instruct(question)
        best_score = float("-inf")
        best_idx = 0

        for j, choice in enumerate(choices):
            full_text = prompt + " " + choice
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(model.device)

            prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            answer_start = prompt_inputs["input_ids"].shape[1]
            if answer_start >= input_ids.shape[1]:
                continue

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

            # Mature layer logits (last hidden state → lm_head)
            mature_hidden = hidden_states[mature_layer + 1][:, -1, :]  # +1 for embedding layer
            mature_logits = model.lm_head(model.model.norm(mature_hidden))
            mature_probs = F.softmax(mature_logits.float(), dim=-1)

            # Select premature layer
            if premature == "dynamic":
                # Dynamic: find layer with max JSD from mature
                best_jsd = -1
                best_prem_layer = 0
                for l in range(min(16, n_layers - 1)):
                    prem_hidden = hidden_states[l + 1][:, -1, :]
                    prem_logits = model.lm_head(model.model.norm(prem_hidden))
                    prem_probs = F.softmax(prem_logits.float(), dim=-1)

                    m = 0.5 * (mature_probs + prem_probs)
                    jsd = 0.5 * (
                        F.kl_div(m.log(), mature_probs, reduction='sum') +
                        F.kl_div(m.log(), prem_probs, reduction='sum')
                    )
                    if jsd > best_jsd:
                        best_jsd = jsd
                        best_prem_layer = l
                prem_layer = best_prem_layer
            else:
                prem_layer = premature

            # Premature logits
            prem_hidden = hidden_states[prem_layer + 1][:, -1, :]
            prem_logits = model.lm_head(model.model.norm(prem_hidden))

            # DoLa contrast
            dola_logits = mature_logits - prem_logits

            # Score answer tokens
            answer_ids = input_ids[0, answer_start:]
            log_probs = F.log_softmax(dola_logits[0].float(), dim=-1)
            score = sum(log_probs[t].item() for t in answer_ids[:5])

            if score > best_score:
                best_score = score
                best_idx = j

        is_correct = labels[best_idx] == 1
        if is_correct:
            correct += 1
        total += 1
        per_sample.append({"idx": i, "correct": is_correct})

    mc1 = correct / max(total, 1)
    return {"mc1": mc1, "correct": correct, "total": total,
            "premature": str(premature), "per_sample": per_sample}


# ============================================================
# CAA Implementation (using register_forward_hook)
# ============================================================

def extract_caa_directions(model, tokenizer, dataset, n_layers, n_pairs=256):
    """Extract CAA steering directions from correct-incorrect pairs."""
    logger.info(f"Extracting CAA directions from {n_pairs} pairs...")

    directions = {l: [] for l in range(n_layers)}
    count = 0

    for sample in dataset:
        if count >= n_pairs:
            break

        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue
        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        correct_idx = labels.index(1) if 1 in labels else None
        if correct_idx is None:
            continue
        incorrect_indices = [i for i, l in enumerate(labels) if l == 0]
        if not incorrect_indices:
            continue

        question = sample["question"]
        prompt = format_prompt_instruct(question)

        text_c = prompt + " " + choices[correct_idx]
        text_i = prompt + " " + choices[incorrect_indices[0]]

        inputs_c = tokenizer(text_c, return_tensors="pt", truncation=True, max_length=256)
        inputs_i = tokenizer(text_i, return_tensors="pt", truncation=True, max_length=256)

        with torch.no_grad():
            out_c = model(inputs_c["input_ids"].to(model.device), output_hidden_states=True)
            out_i = model(inputs_i["input_ids"].to(model.device), output_hidden_states=True)

        for l in range(n_layers):
            act_c = out_c.hidden_states[l + 1][0, -1, :].float().cpu()
            act_i = out_i.hidden_states[l + 1][0, -1, :].float().cpu()
            directions[l].append(act_c - act_i)

        count += 1
        if count % 50 == 0:
            logger.info(f"  CAA directions: {count}/{n_pairs}")

    # Average
    avg_directions = {}
    for l in range(n_layers):
        if directions[l]:
            avg_directions[l] = torch.stack(directions[l]).mean(dim=0)

    # Layer importance by norm
    norms = {l: avg_directions[l].norm().item() for l in avg_directions}
    max_norm = max(norms.values()) if norms else 1.0
    importance = {l: norms[l] / max_norm for l in norms}

    logger.info(f"CAA directions extracted. Top-3 layers: "
                f"{sorted(importance, key=importance.get, reverse=True)[:3]}")
    return {"directions": avg_directions, "importance": importance}


def evaluate_mc1_caa(model, tokenizer, dataset, caa_data, n_layers,
                     top_k=10, coeff=5.0):
    """Evaluate MC1 with CAA steering via forward hooks."""
    importance = caa_data["importance"]
    directions = caa_data["directions"]

    # Select top-k layers
    sorted_layers = sorted(importance.keys(), key=lambda l: importance[l], reverse=True)
    selected_layers = sorted_layers[:top_k]

    # Register hooks
    hooks_handles = []

    def make_hook(layer_idx, direction, coefficient):
        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hs = output[0]
                hs[:, -1, :] = hs[:, -1, :] + coefficient * direction.to(hs.device).half()
                return (hs,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + coefficient * direction.to(output.device).half()
                return output
        return hook_fn

    # Attach hooks to selected layers
    for l in selected_layers:
        layer_module = model.model.layers[l]
        h = layer_module.register_forward_hook(
            make_hook(l, directions[l], coeff)
        )
        hooks_handles.append(h)

    # Evaluate
    correct = 0
    total = 0
    per_sample = []

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  [CAA k={top_k} c={coeff}] {i}/{len(dataset)}")

        question = sample["question"]
        mc1_targets = sample.get("mc1_targets", {})
        if not mc1_targets:
            continue
        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue

        prompt = format_prompt_instruct(question)
        best_score = float("-inf")
        best_idx = 0

        for j, choice in enumerate(choices):
            s = score_choice(model, tokenizer, prompt, choice)
            if s > best_score:
                best_score = s
                best_idx = j

        is_correct = labels[best_idx] == 1
        if is_correct:
            correct += 1
        total += 1
        per_sample.append({"idx": i, "correct": is_correct})

    # Remove hooks
    for h in hooks_handles:
        h.remove()

    mc1 = correct / max(total, 1)
    return {"mc1": mc1, "correct": correct, "total": total,
            "top_k": top_k, "coeff": coeff, "layers": selected_layers,
            "per_sample": per_sample}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Instruct Model Intervention Experiment")
    parser.add_argument("--gpu-ids", type=str, default="0,1",
                        help="GPU IDs to use (default: 0,1 for 2-card parallel)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples for debugging")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline evaluation if already computed")
    parser.add_argument("--caa-only", action="store_true",
                        help="Only run CAA grid search")
    parser.add_argument("--dola-only", action="store_true",
                        help="Only run DoLa configs")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("INSTRUCT MODEL INTERVENTION - EXTENDED HYPERPARAMETER GRID")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"CAA grid: {len(CAA_COEFFICIENTS)} coefficients × {len(CAA_TOP_K)} top_k = {len(CAA_COEFFICIENTS)*len(CAA_TOP_K)} configs")
    logger.info(f"DoLa configs: {len(DOLA_CONFIGS)}")
    logger.info("=" * 70)

    # Load model
    model, tokenizer, n_layers = load_model_and_tokenizer(args.gpu_ids)

    # Load dataset
    dataset = load_truthfulqa()
    if args.max_samples:
        dataset = dataset[:args.max_samples]

    results = {
        "model": MODEL_NAME,
        "n_samples": len(dataset),
        "n_layers": n_layers,
        "instruct_fep": INSTRUCT_FEP,
        "base_results": BASE_RESULTS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_ids": args.gpu_ids,
    }

    # ========== Baseline ==========
    if not args.skip_baseline and not args.caa_only and not args.dola_only:
        logger.info("\n[BASELINE] Evaluating instruct baseline MC1...")
        t0 = time.time()
        baseline = evaluate_mc1(model, tokenizer, dataset, desc="baseline")
        baseline["time_s"] = time.time() - t0
        results["baseline"] = baseline
        logger.info(f"  Baseline MC1: {baseline['mc1']:.4f} ({baseline['correct']}/{baseline['total']})")
        # Save intermediate
        _save_results(results, "instruct_intervention_partial.json")

    # ========== DoLa Grid ==========
    if not args.caa_only:
        results["dola_results"] = []
        for cfg in DOLA_CONFIGS:
            logger.info(f"\n[DoLa-{cfg['name']}] Evaluating...")
            t0 = time.time()
            dola_res = evaluate_mc1_dola(
                model, tokenizer, dataset, n_layers, premature=cfg["premature"]
            )
            dola_res["config_name"] = cfg["name"]
            dola_res["time_s"] = time.time() - t0
            results["dola_results"].append(dola_res)
            logger.info(f"  DoLa-{cfg['name']} MC1: {dola_res['mc1']:.4f}")
            _save_results(results, "instruct_intervention_partial.json")

    # ========== CAA Grid ==========
    if not args.dola_only:
        logger.info("\n[CAA] Extracting steering directions...")
        caa_data = extract_caa_directions(model, tokenizer, dataset, n_layers)

        results["caa_results"] = []
        total_caa_configs = len(CAA_COEFFICIENTS) * len(CAA_TOP_K)
        config_idx = 0

        for top_k in CAA_TOP_K:
            for coeff in CAA_COEFFICIENTS:
                config_idx += 1
                logger.info(f"\n[CAA {config_idx}/{total_caa_configs}] top_k={top_k}, coeff={coeff}")
                t0 = time.time()
                caa_res = evaluate_mc1_caa(
                    model, tokenizer, dataset, caa_data, n_layers,
                    top_k=top_k, coeff=coeff
                )
                caa_res["time_s"] = time.time() - t0
                results["caa_results"].append(caa_res)
                logger.info(f"  CAA(k={top_k},c={coeff}) MC1: {caa_res['mc1']:.4f}")
                _save_results(results, "instruct_intervention_partial.json")

    # ========== Final Save & Summary ==========
    _save_results(results, "instruct_intervention_full.json")
    _print_summary(results)

    del model
    torch.cuda.empty_cache()


def _save_results(results, filename):
    """Save results (strip per_sample for partial saves)."""
    output_path = RESULTS_DIR / filename
    # For partial saves, don't include per_sample to save space
    save_data = json.loads(json.dumps(results, default=str))
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)


def _print_summary(results):
    """Print experiment summary."""
    print(f"\n{'=' * 70}")
    print("INSTRUCT MODEL INTERVENTION - SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model: {results['model']}")
    print(f"Instruct crystallization: {INSTRUCT_FEP['late_crystal_pct']:.1%}")
    print(f"Base crystallization: {BASE_RESULTS['late_crystal_pct']:.1%}")

    if "baseline" in results:
        print(f"\nBaseline MC1: {results['baseline']['mc1']:.4f}")

    if "dola_results" in results:
        print(f"\nDoLa Results:")
        best_dola = max(results["dola_results"], key=lambda x: x["mc1"])
        for d in results["dola_results"]:
            marker = " ★" if d["mc1"] == best_dola["mc1"] else ""
            print(f"  {d['config_name']:15s}: MC1 = {d['mc1']:.4f}{marker}")
        print(f"  Best DoLa: {best_dola['config_name']} ({best_dola['mc1']:.4f})")

    if "caa_results" in results:
        print(f"\nCAA Results (top 5):")
        sorted_caa = sorted(results["caa_results"], key=lambda x: x["mc1"], reverse=True)
        for c in sorted_caa[:5]:
            print(f"  k={c['top_k']:2d} c={c['coeff']:4.1f}: MC1 = {c['mc1']:.4f}")
        best_caa = sorted_caa[0]
        print(f"  Best CAA: k={best_caa['top_k']}, c={best_caa['coeff']} ({best_caa['mc1']:.4f})")

    # Theory validation
    if "dola_results" in results and "caa_results" in results and "baseline" in results:
        best_dola_mc1 = max(d["mc1"] for d in results["dola_results"])
        best_caa_mc1 = max(c["mc1"] for c in results["caa_results"])
        baseline_mc1 = results["baseline"]["mc1"]

        print(f"\n{'=' * 70}")
        print("FEP THEORY VALIDATION")
        print(f"{'=' * 70}")
        print(f"Prediction: Low crystallization (37.3%) → CAA should ≥ DoLa")
        dola_delta = f"{(best_dola_mc1-baseline_mc1)/baseline_mc1*100:+.1f}%" if baseline_mc1 > 0 else "N/A"
        caa_delta = f"{(best_caa_mc1-baseline_mc1)/baseline_mc1*100:+.1f}%" if baseline_mc1 > 0 else "N/A"
        print(f"  Best DoLa: {best_dola_mc1:.4f} (Δ from baseline: {dola_delta})")
        print(f"  Best CAA:  {best_caa_mc1:.4f} (Δ from baseline: {caa_delta})")
        print(f"  Baseline:  {baseline_mc1:.4f}")
        print(f"\n  CAA > DoLa: {'YES ✓' if best_caa_mc1 > best_dola_mc1 else 'NO ✗'}")
        print(f"  Any method > Baseline: {'YES ✓' if max(best_dola_mc1, best_caa_mc1) > baseline_mc1 else 'NO ✗'}")


if __name__ == "__main__":
    main()
