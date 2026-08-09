"""Run censoring-aware FEP measurements with strict token-alignment checks.

This is the canonical runner for the BlackboxNLP revision.  It intentionally
does not overwrite legacy FEP outputs.  Every sample must pass two gates:

1. the target is the first continuation token from the tokenized full prompt;
2. the last hidden-state projection preserves top-k membership and stays within
   a bounded numerical tolerance of the model logits.

Examples:
    python blackboxnlp-2026/experiments/run_corrected_fep.py --model qwen7 --max-samples 50
    python blackboxnlp-2026/experiments/run_corrected_fep.py --model qwen7 --dataset mmlu --max-samples 50
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLACKBOX_ROOT = PROJECT_ROOT / "blackboxnlp-2026"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mechlens.fep import (  # noqa: E402
    TokenAlignmentError,
    aggregate_fep_results,
    extract_aligned_continuation_ids,
    normalize_continuation,
    render_truthfulqa_prompt,
    summarize_topk_trajectory,
)

logger = logging.getLogger("corrected_fep")

MODEL_ALIASES = {
    "pythia": "EleutherAI/pythia-6.9b",
    "qwen7": "Qwen/Qwen2.5-7B",
    "qwen14": "Qwen/Qwen2.5-14B",
    "gemma": "google/gemma-7b",
    "llama": "meta-llama/Llama-3.1-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
}

MMLU_SUBJECT_GROUPS = {
    "STEM": [
        "abstract_algebra",
        "college_mathematics",
        "college_physics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "machine_learning",
        "computer_security",
    ],
    "Humanities": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "philosophy",
        "moral_scenarios",
    ],
    "Social_Sciences": [
        "high_school_psychology",
        "sociology",
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ],
    "Other": [
        "clinical_knowledge",
        "medical_genetics",
        "anatomy",
        "professional_medicine",
        "nutrition",
    ],
}
MMLU_SUBJECTS = [
    subject for subjects in MMLU_SUBJECT_GROUPS.values() for subject in subjects
]
MMLU_SUBJECT_TO_GROUP = {
    subject: group
    for group, subjects in MMLU_SUBJECT_GROUPS.items()
    for subject in subjects
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="qwen7",
        help="Model alias or HuggingFace/local model path",
    )
    parser.add_argument(
        "--dataset", choices=["truthfulqa", "mmlu", "sst2"], default="truthfulqa"
    )
    parser.add_argument(
        "--prompt-template",
        choices=["qa", "question_answer", "instruction"],
        default="qa",
        help="TruthfulQA prompt template; other datasets use their fixed task format",
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Immutable Hugging Face model/tokenizer revision when loading a remote ID",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Immutable datasets revision for remote MMLU loading",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--mmlu-data-dir",
        type=Path,
        default=None,
        help="Optional local cais/mmlu snapshot with <subject>/test-*.parquet files",
    )
    parser.add_argument(
        "--sst2-data-file",
        type=Path,
        default=None,
        help="SST-2 validation TSV with sentence and label columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BLACKBOX_ROOT / "results" / "corrected_fep",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    return parser.parse_args()


def resolve_model_reference(model_arg: str) -> tuple[str, str]:
    """Return a stable display key and a loadable model reference."""

    if model_arg in MODEL_ALIASES:
        return model_arg, MODEL_ALIASES[model_arg]
    path = Path(model_arg).expanduser()
    if path.exists():
        return path.name, str(path.resolve())
    return model_arg.replace("/", "__"), model_arg


def load_model_and_tokenizer(
    model_ref: str, dtype_name: str, cache_dir: Path | None, revision: str | None
):
    """Load one causal LM on the single experiment GPU."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    cache = str(cache_dir) if cache_dir else None
    logger.info("Loading tokenizer from %s", model_ref)
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        cache_dir=cache,
        trust_remote_code=True,
        revision=revision,
    )
    logger.info("Loading model from %s (%s)", model_ref, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        cache_dir=cache,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        revision=revision,
    )
    model.eval()
    return model, tokenizer


def get_final_projection(model):
    """Resolve the model-family-specific final norm and output projection."""

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        final_norm = model.gpt_neox.final_layer_norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_norm = model.transformer.ln_f
    else:
        raise TypeError(f"Unsupported final-normalization layout: {type(model).__name__}")

    output_projection = model.get_output_embeddings()
    if output_projection is None:
        raise TypeError(f"No output embedding found for {type(model).__name__}")
    return final_norm, output_projection


def flatten_input_ids(encoded) -> list[int]:
    """Convert tokenizer outputs or tensors to a single sequence of ids."""

    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def prepare_aligned_target(tokenizer, prompt: str, answer: str) -> dict:
    """Tokenize the complete prompt/answer pair and extract its first new token."""

    continuation = normalize_continuation(answer)
    prompt_ids = flatten_input_ids(tokenizer(prompt, add_special_tokens=True))
    full_ids = flatten_input_ids(tokenizer(prompt + continuation, add_special_tokens=True))
    continuation_ids = extract_aligned_continuation_ids(prompt_ids, full_ids)
    target_token_id = continuation_ids[0]
    return {
        "prompt_ids": prompt_ids,
        "continuation": continuation,
        "continuation_ids": continuation_ids,
        "target_token_id": target_token_id,
        "target_token_str": tokenizer.decode(
            [target_token_id], clean_up_tokenization_spaces=False
        ),
    }


def token_rank(logits: torch.Tensor, target_token_id: int) -> int:
    """Return a zero-based descending rank without sorting the vocabulary."""

    target_logit = logits[target_token_id]
    return int(torch.count_nonzero(logits > target_logit).item())


def target_probability(logits: torch.Tensor, target_token_id: int) -> float:
    """Compute the target probability in float32 for stable serialization."""

    return float(torch.softmax(logits.float(), dim=-1)[target_token_id].item())


def measure_sample(
    model,
    tokenizer,
    final_norm,
    output_projection,
    sample: dict,
    top_k: int,
    max_prompt_tokens: int,
) -> dict:
    """Measure one sample and enforce alignment/logit-parity invariants."""

    prompt = sample["prompt"]
    alignment = prepare_aligned_target(tokenizer, prompt, sample["answer"])
    prompt_ids = alignment["prompt_ids"]
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError(
            f"Prompt has {len(prompt_ids)} tokens, exceeding the limit {max_prompt_tokens}"
        )

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("Model did not return layer hidden states")
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if len(hidden_states) != n_layers + 1:
        raise RuntimeError(
            f"Expected {n_layers + 1} hidden states, received {len(hidden_states)}"
        )

    target_token_id = alignment["target_token_id"]
    layer_ranks: list[int] = []
    layer_probs: list[float] = []
    for layer_index, hidden_state in enumerate(hidden_states[1:]):
        last_hidden = hidden_state[0, -1, :]
        # HuggingFace decoder models return the final tuple element after the
        # model's final norm.  Earlier elements are residual-stream values and
        # need that norm for a logit-lens projection.
        projected_hidden = last_hidden if layer_index == n_layers - 1 else final_norm(last_hidden)
        layer_logits = output_projection(projected_hidden).float()
        layer_ranks.append(token_rank(layer_logits, target_token_id))
        layer_probs.append(target_probability(layer_logits, target_token_id))

    model_final_logits = outputs.logits[0, -1, :].float()
    projected_final_logits = output_projection(hidden_states[-1][0, -1, :]).float()
    model_final_rank = token_rank(model_final_logits, target_token_id)
    projected_final_rank = token_rank(projected_final_logits, target_token_id)
    max_abs_logit_delta = float(
        torch.max(torch.abs(model_final_logits - projected_final_logits)).item()
    )
    final_topk_match = (model_final_rank < top_k) == (projected_final_rank < top_k)
    if not final_topk_match or max_abs_logit_delta > 0.25:
        raise RuntimeError(
            "Final-logit parity failed: "
            f"model rank={model_final_rank}, projected rank={projected_final_rank}, "
            f"max absolute logit delta={max_abs_logit_delta:.6f}"
        )

    # Use the model's native logits for the final layer.  In BF16, recomputing
    # the same projection can move near-tied, low-ranked tokens by one or two
    # positions without changing the scientifically relevant top-k membership.
    layer_ranks[-1] = model_final_rank
    layer_probs[-1] = target_probability(model_final_logits, target_token_id)

    layer_in_topk = [rank < top_k for rank in layer_ranks]
    trajectory = summarize_topk_trajectory(layer_in_topk)
    prediction_id = int(torch.argmax(model_final_logits).item())
    candidate_answers = sample.get("candidate_answers")
    candidate_metrics = {}
    if candidate_answers:
        candidate_alignments = [
            prepare_aligned_target(tokenizer, prompt, answer)
            for answer in candidate_answers
        ]
        candidate_token_ids = [
            alignment["target_token_id"] for alignment in candidate_alignments
        ]
        if len(set(candidate_token_ids)) != len(candidate_token_ids):
            raise TokenAlignmentError(
                f"Candidate answers do not have unique first tokens: {candidate_answers}"
            )
        candidate_logits = model_final_logits[candidate_token_ids]
        candidate_prediction_index = int(torch.argmax(candidate_logits).item())
        candidate_metrics = {
            "candidate_answers": candidate_answers,
            "candidate_token_ids": candidate_token_ids,
            "candidate_token_strs": [
                alignment["target_token_str"] for alignment in candidate_alignments
            ],
            "candidate_prediction_index": candidate_prediction_index,
            "candidate_prediction": candidate_answers[candidate_prediction_index],
            "candidate_correct": (
                candidate_token_ids[candidate_prediction_index] == target_token_id
            ),
        }
    return {
        "id": sample["id"],
        "benchmark": sample["benchmark"],
        "category": sample.get("category"),
        "group": sample.get("group"),
        "question": sample.get("question"),
        "prompt": prompt,
        "answer": sample["answer"],
        "continuation": alignment["continuation"],
        "target_token_id": target_token_id,
        "target_token_str": alignment["target_token_str"],
        "continuation_token_ids": alignment["continuation_ids"],
        "prompt_token_count": len(prompt_ids),
        "n_layers": len(layer_ranks),
        "top_k": top_k,
        "layer_ranks": layer_ranks,
        "layer_probs": layer_probs,
        "layer_in_topk": layer_in_topk,
        "final_rank": model_final_rank,
        "final_prob": target_probability(model_final_logits, target_token_id),
        "prediction_token_id": prediction_id,
        "prediction_token_str": tokenizer.decode(
            [prediction_id], clean_up_tokenization_spaces=False
        ),
        "first_token_correct": prediction_id == target_token_id,
        "final_logit_rank_match": model_final_rank == projected_final_rank,
        "final_logit_rank_delta": projected_final_rank - model_final_rank,
        "final_logit_topk_match": final_topk_match,
        "final_max_abs_logit_delta": max_abs_logit_delta,
        **candidate_metrics,
        **trajectory,
    }


def load_truthfulqa_samples(prompt_template: str = "qa") -> list[dict]:
    path = PROJECT_ROOT / "data" / "truthfulqa.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["samples"] if isinstance(payload, dict) else payload
    return [
        {
            "id": row["id"],
            "benchmark": "truthfulqa",
            "category": row.get("category", "Unknown"),
            "question": row["question"],
            "prompt": render_truthfulqa_prompt(row["question"], prompt_template),
            "answer": row["best_answer"],
        }
        for row in rows
        if row.get("best_answer", "").strip()
    ]


def load_mmlu_samples(
    max_per_subject: int,
    data_dir: Path | None = None,
    revision: str | None = None,
) -> list[dict]:
    labels = ["A", "B", "C", "D"]
    rows = []
    for subject in MMLU_SUBJECTS:
        if data_dir is not None:
            import pyarrow.parquet as parquet

            parquet_files = sorted((data_dir / subject).glob("test-*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No test parquet found for MMLU subject {subject}")
            dataset = parquet.read_table(parquet_files[0]).to_pylist()
        else:
            from datasets import load_dataset

            dataset = load_dataset("cais/mmlu", subject, split="test", revision=revision)
        for index, item in enumerate(dataset):
            if index >= max_per_subject:
                break
            prompt_lines = [f"Question: {item['question']}"]
            prompt_lines.extend(
                f"{label}. {choice}" for label, choice in zip(labels, item["choices"])
            )
            prompt_lines.append("Answer:")
            correct_label = labels[int(item["answer"])]
            rows.append(
                {
                    "id": f"{subject}_{index}",
                    "benchmark": "mmlu",
                    "category": subject,
                    "group": MMLU_SUBJECT_TO_GROUP[subject],
                    "question": item["question"],
                    "prompt": "\n".join(prompt_lines),
                    "answer": correct_label,
                    "candidate_answers": labels,
                }
            )
    return rows


def load_sst2_samples(data_file: Path) -> list[dict]:
    """Load labeled SST-2 rows using the same continuation contract as other tasks."""

    if not data_file.is_file():
        raise FileNotFoundError(f"SST-2 data file not found: {data_file}")
    rows = []
    with data_file.open(encoding="utf-8", newline="") as handle:
        for index, item in enumerate(csv.DictReader(handle, delimiter="\t")):
            label_id = int(item["label"])
            label = "positive" if label_id == 1 else "negative"
            sentence = item["sentence"]
            rows.append(
                {
                    "id": f"sst2_validation_{index}",
                    "benchmark": "sst2",
                    "category": label,
                    "question": sentence,
                    "prompt": f'The sentiment of "{sentence}" is:',
                    "answer": label,
                    "candidate_answers": ["negative", "positive"],
                }
            )
    return rows


def choose_samples(samples: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.sample_strategy == "random":
        rng = random.Random(args.seed)
        samples = samples.copy()
        rng.shuffle(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    return samples


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_provenance(args: argparse.Namespace) -> dict:
    if args.dataset == "truthfulqa":
        path = PROJECT_ROOT / "data" / "truthfulqa.json"
        return {"source": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
    if args.dataset == "mmlu":
        return {
            "source": str(args.mmlu_data_dir) if args.mmlu_data_dir else "cais/mmlu",
            "revision": args.dataset_revision,
            "subjects": MMLU_SUBJECTS,
        }
    assert args.sst2_data_file is not None
    return {
        "source": str(args.sst2_data_file),
        "sha256": file_sha256(args.sst2_data_file),
        "split": "validation",
    }


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)


def build_payload(
    args: argparse.Namespace,
    model_key: str,
    model_ref: str,
    model,
    tokenizer,
    results: list[dict],
    failures: list[dict],
    started_at: str,
) -> dict:
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    return {
        "schema_version": "corrected-fep-v1",
        "experiment": {
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": git_commit(),
            "model_key": model_key,
            "model_reference": model_ref,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "resolved_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get(
                "_commit_hash"
            ),
            "model_class": type(model).__name__,
            "dataset": args.dataset,
            "prompt_template": (
                args.prompt_template if args.dataset == "truthfulqa" else "dataset_default"
            ),
            "top_k": args.top_k,
            "dtype": args.dtype,
            "sample_strategy": args.sample_strategy,
            "seed": args.seed,
            "requested_samples": args.max_samples,
            "target_definition": "first token of full prompt-conditioned continuation",
            "censoring_policy": "never-observed FEP is null",
            "layer_numbering": "fep_layer is zero-based; fep_layer_number is one-based",
            "dataset_provenance": dataset_provenance(args),
        },
        "summary": aggregate_fep_results(results, n_layers=n_layers),
        "failures": failures,
        "per_sample_results": results,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    model_key, model_ref = resolve_model_reference(args.model)
    if args.dataset == "truthfulqa":
        samples = load_truthfulqa_samples(args.prompt_template)
    elif args.dataset == "mmlu":
        per_subject = max(
            1, math.ceil((args.max_samples or 200) / len(MMLU_SUBJECTS))
        )
        samples = load_mmlu_samples(
            per_subject, args.mmlu_data_dir, revision=args.dataset_revision
        )
    else:
        if args.sst2_data_file is None:
            raise ValueError("--sst2-data-file is required for the SST-2 experiment")
        samples = load_sst2_samples(args.sst2_data_file)
    samples = choose_samples(samples, args)
    if not samples:
        raise RuntimeError("No experiment samples were loaded")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_key = model_key.replace("/", "__")
    template_suffix = (
        f"_{args.prompt_template}" if args.dataset == "truthfulqa" else ""
    )
    output_path = args.output_dir / (
        f"{args.dataset}_{safe_model_key}{template_suffix}_{timestamp}.json"
    )
    logger.info("Output path: %s", output_path)

    model, tokenizer = load_model_and_tokenizer(
        model_ref, args.dtype, args.cache_dir, args.model_revision
    )
    final_norm, output_projection = get_final_projection(model)
    results: list[dict] = []
    failures: list[dict] = []

    for index, sample in enumerate(samples, start=1):
        try:
            result = measure_sample(
                model,
                tokenizer,
                final_norm,
                output_projection,
                sample,
                args.top_k,
                args.max_prompt_tokens,
            )
            results.append(result)
        except (TokenAlignmentError, ValueError, RuntimeError) as error:
            logger.error("Sample %s failed: %s", sample["id"], error)
            failures.append({"id": sample["id"], "error": str(error)})

        if index % args.checkpoint_every == 0 or index == len(samples):
            payload = build_payload(
                args,
                model_key,
                model_ref,
                model,
                tokenizer,
                results,
                failures,
                started_at,
            )
            save_payload(output_path, payload)
            summary = payload["summary"]
            logger.info(
                "Checkpoint %d/%d: observed=%s, never=%s, final-top%d=%s, failures=%d",
                index,
                len(samples),
                summary.get("observed_count"),
                summary.get("never_observed_count"),
                args.top_k,
                summary.get("final_topk_count"),
                len(failures),
            )
        if index % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Experiment complete: %s", output_path)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
