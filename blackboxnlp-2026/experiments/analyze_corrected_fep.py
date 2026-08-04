#!/usr/bin/env python3
"""Create paper tables and figures from corrected, canonical FEP artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLACKBOX_ROOT = PROJECT_ROOT / "blackboxnlp-2026"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mechlens.fep_analysis import (  # noqa: E402, I001
    benjamini_hochberg_adjusted_pvalues,
    exact_sign_test_pvalue,
    holm_adjusted_pvalues,
    mcnemar_exact_pvalue,
    summarize_rank_trajectories,
)


DISPLAY_NAMES = {
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen2.5-14B": "Qwen2.5-14B",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "Mistral-7B-v0.1": "Mistral-7B",
    "pythia-6.9b": "Pythia-6.9B",
    "gemma-7b": "Gemma-7B",
}


def load_artifacts(input_dir: Path) -> list[dict]:
    artifacts = []
    for path in sorted(input_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("schema_version") != "corrected-fep-v1":
            raise ValueError(f"Unexpected schema in {path}")
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    if not artifacts:
        raise FileNotFoundError(f"No corrected .json.gz artifacts in {input_dir}")
    return artifacts


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_tables(artifacts: list[dict], output_dir: Path) -> None:
    rows = []
    for artifact in artifacts:
        experiment = artifact["experiment"]
        summary = summarize_rank_trajectories(artifact["per_sample_results"], top_k=10)
        source_summary = artifact["summary"]
        rows.append(
            {
                "dataset": experiment["dataset"],
                "model": DISPLAY_NAMES.get(experiment["model_key"], experiment["model_key"]),
                "n": summary["n"],
                "layers": source_summary["n_layers"],
                "candidate_accuracy": source_summary.get("candidate_accuracy"),
                **summary,
            }
        )
    write_csv(output_dir / "trajectory_summary.csv", rows)

    qwen_truth = next(
        item
        for item in artifacts
        if item["experiment"]["dataset"] == "truthfulqa"
        and item["experiment"]["model_key"] == "Qwen2.5-7B"
    )
    sensitivity = [
        summarize_rank_trajectories(qwen_truth["per_sample_results"], top_k=top_k)
        for top_k in (1, 3, 5, 10, 20, 50, 100)
    ]
    write_csv(output_dir / "topk_sensitivity_qwen7b.csv", sensitivity)

    group_rows = []
    mmlu = next(item for item in artifacts if item["experiment"]["dataset"] == "mmlu")
    groups = sorted({sample["group"] for sample in mmlu["per_sample_results"]})
    for group in groups:
        samples = [sample for sample in mmlu["per_sample_results"] if sample["group"] == group]
        summary = summarize_rank_trajectories(samples, top_k=10)
        group_rows.append(
            {
                "group": group,
                "candidate_accuracy": sum(sample["candidate_correct"] for sample in samples)
                / len(samples),
                **summary,
            }
        )
    write_csv(output_dir / "mmlu_groups.csv", group_rows)

    build_single_layer_sensitivity(artifacts, output_dir)
    build_pairwise_final_visibility(artifacts, output_dir)
    build_pairwise_first_entry_depth(artifacts, output_dir)
    build_truthfulqa_target_audit(artifacts, output_dir)
    build_dataset_sample_manifest(artifacts, output_dir)


def build_single_layer_sensitivity(artifacts: list[dict], output_dir: Path) -> None:
    """Measure how much each TruthfulQA readout affects trajectory summaries."""

    layer_rows = []
    summary_rows = []
    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    for artifact in truth:
        samples = artifact["per_sample_results"]
        model = DISPLAY_NAMES.get(
            artifact["experiment"]["model_key"], artifact["experiment"]["model_key"]
        )
        n_layers = len(samples[0]["layer_ranks"])
        raw = summarize_rank_trajectories(samples, top_k=10)
        prevalence = np.asarray(
            [[rank < 10 for rank in sample["layer_ranks"]] for sample in samples]
        ).mean(axis=0)
        medians = np.median(
            np.asarray([sample["layer_ranks"] for sample in samples]), axis=0
        )
        model_rows = []
        for layer_index in range(n_layers):
            masked = summarize_rank_trajectories(
                samples, top_k=10, ignored_layer_index=layer_index
            )
            row = {
                "model": model,
                "layer_number": layer_index + 1,
                "normalized_depth": (layer_index + 1) / n_layers,
                "top10_prevalence": prevalence[layer_index],
                "median_target_rank": medians[layer_index],
                "raw_dropout_pct": raw["dropout_pct"],
                "masked_dropout_pct": masked["dropout_pct"],
                "dropout_reduction_pct": raw["dropout_pct"] - masked["dropout_pct"],
                "raw_persistent_depth": raw["mean_persistent_depth_final"],
                "masked_persistent_depth": masked["mean_persistent_depth_final"],
            }
            layer_rows.append(row)
            model_rows.append(row)
        most_influential = max(model_rows, key=lambda row: row["dropout_reduction_pct"])
        ordered_influence = sorted(
            model_rows, key=lambda row: row["dropout_reduction_pct"], reverse=True
        )
        second_largest_reduction = ordered_influence[1]["dropout_reduction_pct"]
        layer_index = int(most_influential["layer_number"]) - 1
        summary_rows.append(
            {
                "model": model,
                "layers": n_layers,
                "most_influential_layer_number": layer_index + 1,
                "top10_prevalence_at_layer": prevalence[layer_index],
                "previous_layer_prevalence": (
                    prevalence[layer_index - 1] if layer_index > 0 else ""
                ),
                "final_layer_prevalence": prevalence[-1],
                "previous_layer_median_rank": (
                    medians[layer_index - 1] if layer_index > 0 else ""
                ),
                "median_rank_at_layer": medians[layer_index],
                "next_layer_median_rank": (
                    medians[layer_index + 1] if layer_index + 1 < n_layers else ""
                ),
                "raw_dropout_pct": raw["dropout_pct"],
                "masked_dropout_pct": most_influential["masked_dropout_pct"],
                "dropout_reduction_pct": most_influential["dropout_reduction_pct"],
                "second_largest_dropout_reduction_pct": second_largest_reduction,
                "largest_to_second_ratio": (
                    most_influential["dropout_reduction_pct"]
                    / second_largest_reduction
                    if second_largest_reduction > 0
                    else ""
                ),
                "raw_persistent_depth": raw["mean_persistent_depth_final"],
                "masked_persistent_depth": most_influential[
                    "masked_persistent_depth"
                ],
            }
        )
    write_csv(output_dir / "layer_influence.csv", layer_rows)
    write_csv(output_dir / "single_layer_sensitivity.csv", summary_rows)


def build_pairwise_final_visibility(artifacts: list[dict], output_dir: Path) -> None:
    """Run paired model comparisons with family- and FDR-adjusted p-values."""

    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    rows = []
    for first, second in combinations(truth, 2):
        first_by_id = {
            str(sample["id"]): bool(sample["final_in_topk"])
            for sample in first["per_sample_results"]
        }
        second_by_id = {
            str(sample["id"]): bool(sample["final_in_topk"])
            for sample in second["per_sample_results"]
        }
        if first_by_id.keys() != second_by_id.keys():
            raise ValueError("TruthfulQA artifacts do not contain identical sample IDs")
        first_only = sum(
            first_by_id[sample_id] and not second_by_id[sample_id]
            for sample_id in first_by_id
        )
        second_only = sum(
            second_by_id[sample_id] and not first_by_id[sample_id]
            for sample_id in first_by_id
        )
        rows.append(
            {
                "model_a": DISPLAY_NAMES.get(
                    first["experiment"]["model_key"], first["experiment"]["model_key"]
                ),
                "model_b": DISPLAY_NAMES.get(
                    second["experiment"]["model_key"], second["experiment"]["model_key"]
                ),
                "a_only_final_top10": first_only,
                "b_only_final_top10": second_only,
                "mcnemar_exact_p": mcnemar_exact_pvalue(first_only, second_only),
            }
        )
    raw_pvalues = [row["mcnemar_exact_p"] for row in rows]
    holm = holm_adjusted_pvalues(raw_pvalues)
    bh = benjamini_hochberg_adjusted_pvalues(raw_pvalues)
    for row, holm_pvalue, bh_pvalue in zip(rows, holm, bh, strict=True):
        row["holm_adjusted_p"] = holm_pvalue
        row["holm_reject_0.05"] = holm_pvalue <= 0.05
        row["bh_adjusted_p"] = bh_pvalue
        row["bh_reject_0.05"] = bh_pvalue <= 0.05
    write_csv(output_dir / "pairwise_final_visibility.csv", rows)


def build_pairwise_first_entry_depth(artifacts: list[dict], output_dir: Path) -> None:
    """Compare first-entry depths on items observed by both models."""

    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    rows = []
    for first, second in combinations(truth, 2):
        first_by_id = {str(sample["id"]): sample for sample in first["per_sample_results"]}
        second_by_id = {
            str(sample["id"]): sample for sample in second["per_sample_results"]
        }
        if first_by_id.keys() != second_by_id.keys():
            raise ValueError("TruthfulQA artifacts do not contain identical sample IDs")
        paired = [
            (first_by_id[sample_id]["fep_depth"], second_by_id[sample_id]["fep_depth"])
            for sample_id in first_by_id
            if first_by_id[sample_id]["fep_observed"]
            and second_by_id[sample_id]["fep_observed"]
        ]
        a_earlier = sum(a < b for a, b in paired)
        b_earlier = sum(a > b for a, b in paired)
        ties = sum(a == b for a, b in paired)
        rows.append(
            {
                "model_a": DISPLAY_NAMES.get(
                    first["experiment"]["model_key"], first["experiment"]["model_key"]
                ),
                "model_b": DISPLAY_NAMES.get(
                    second["experiment"]["model_key"], second["experiment"]["model_key"]
                ),
                "jointly_observed_n": len(paired),
                "model_a_mean_depth": np.mean([a for a, _ in paired]),
                "model_b_mean_depth": np.mean([b for _, b in paired]),
                "model_a_earlier_n": a_earlier,
                "model_b_earlier_n": b_earlier,
                "ties_n": ties,
                "exact_sign_p": exact_sign_test_pvalue(a_earlier, b_earlier),
            }
        )
    raw_pvalues = [row["exact_sign_p"] for row in rows]
    holm = holm_adjusted_pvalues(raw_pvalues)
    for row, adjusted in zip(rows, holm, strict=True):
        row["holm_adjusted_p"] = adjusted
        row["holm_reject_0.05"] = adjusted <= 0.05
    write_csv(output_dir / "pairwise_first_entry_depth.csv", rows)


def build_truthfulqa_target_audit(artifacts: list[dict], output_dir: Path) -> None:
    """Describe how much the first-token target compresses multi-token answers."""

    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    summary_rows = []
    frequency_rows = []
    for artifact in truth:
        model = DISPLAY_NAMES.get(
            artifact["experiment"]["model_key"], artifact["experiment"]["model_key"]
        )
        samples = artifact["per_sample_results"]
        counts = Counter(sample["target_token_str"].strip() for sample in samples)
        summary_rows.append(
            {
                "model": model,
                "n": len(samples),
                "multi_token_answer_pct": sum(
                    len(sample["continuation_token_ids"]) > 1 for sample in samples
                )
                / len(samples),
                "distinct_lexical_first_tokens": len(counts),
                "top10_lexical_first_token_share": sum(
                    count for _, count in counts.most_common(10)
                )
                / len(samples),
            }
        )
        for rank, (token, count) in enumerate(counts.most_common(), start=1):
            frequency_rows.append(
                {
                    "model": model,
                    "frequency_rank": rank,
                    "lexical_first_token": token,
                    "count": count,
                    "share": count / len(samples),
                }
            )
    write_csv(output_dir / "truthfulqa_target_audit.csv", summary_rows)
    write_csv(output_dir / "truthfulqa_first_token_frequencies.csv", frequency_rows)


def build_dataset_sample_manifest(artifacts: list[dict], output_dir: Path) -> None:
    """Hash the logical sample inputs embedded in canonical artifacts."""

    rows = []
    for dataset in sorted({item["experiment"]["dataset"] for item in artifacts}):
        candidates = [item for item in artifacts if item["experiment"]["dataset"] == dataset]
        canonical = candidates[0]
        fields = ("id", "category", "group", "question", "prompt", "answer")
        records = [
            {field: sample.get(field) for field in fields}
            for sample in canonical["per_sample_results"]
        ]
        encoded = json.dumps(
            records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        for other in candidates[1:]:
            other_records = [
                {field: sample.get(field) for field in fields}
                for sample in other["per_sample_results"]
            ]
            other_encoded = json.dumps(
                other_records,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if hashlib.sha256(other_encoded).hexdigest() != digest:
                raise ValueError(f"Canonical {dataset} artifacts contain different samples")
        rows.append(
            {
                "dataset": dataset,
                "n": len(records),
                "logical_input_sha256": digest,
                "fields": "+".join(fields),
            }
        )
    write_csv(output_dir / "dataset_sample_manifest.csv", rows)


def build_prompt_table(
    artifacts: list[dict], prompt_dir: Path, output_dir: Path
) -> None:
    baseline = next(
        item
        for item in artifacts
        if item["experiment"]["dataset"] == "truthfulqa"
        and item["experiment"]["model_key"] == "Qwen2.5-7B"
    )
    prompt_artifacts = load_artifacts(prompt_dir)
    all_runs = [("qa", baseline)] + [
        (item["experiment"]["prompt_template"], item) for item in prompt_artifacts
    ]
    baseline_by_id = {
        str(sample["id"]): sample for sample in baseline["per_sample_results"]
    }
    rows = []
    for template, artifact in all_runs:
        samples = artifact["per_sample_results"]
        summary = summarize_rank_trajectories(samples, top_k=10)
        gains = losses = 0
        if template != "qa":
            for sample in samples:
                original = baseline_by_id[str(sample["id"])]["final_in_topk"]
                revised = sample["final_in_topk"]
                gains += bool(not original and revised)
                losses += bool(original and not revised)
        rows.append(
            {
                "template": template,
                **summary,
                "final_visibility_gains_vs_qa": gains,
                "final_visibility_losses_vs_qa": losses,
                "mcnemar_exact_p_vs_qa": (
                    mcnemar_exact_pvalue(gains, losses) if template != "qa" else ""
                ),
            }
        )
    write_csv(output_dir / "prompt_sensitivity.csv", rows)


def prevalence_curve(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([[rank < 10 for rank in sample["layer_ranks"]] for sample in samples])
    depths = np.arange(1, matrix.shape[1] + 1) / matrix.shape[1]
    return depths, matrix.mean(axis=0)


def plot_truthfulqa(artifacts: list[dict], figures_dir: Path) -> None:
    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    order = ["Qwen2.5-7B", "Qwen2.5-14B", "Llama-3.1-8B", "Mistral-7B-v0.1", "pythia-6.9b", "gemma-7b"]
    truth.sort(key=lambda item: order.index(item["experiment"]["model_key"]))

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    for artifact in truth:
        key = artifact["experiment"]["model_key"]
        depths, prevalence = prevalence_curve(artifact["per_sample_results"])
        ax_curve.plot(depths, prevalence, linewidth=2, label=DISPLAY_NAMES[key])
    ax_curve.set(xlabel="Normalized layer depth", ylabel="Fraction in vocabulary top-10", xlim=(0, 1), ylim=(0, 0.8))
    ax_curve.grid(alpha=0.2)
    ax_curve.legend(fontsize=7, ncol=2, loc="upper left")

    names = [DISPLAY_NAMES[item["experiment"]["model_key"]] for item in truth]
    never = [item["summary"]["never_observed_pct"] for item in truth]
    dropout = [item["summary"]["dropout_after_entry_pct"] for item in truth]
    y = np.arange(len(names))
    ax_bar.barh(y - 0.18, never, height=0.34, label="Never observed")
    ax_bar.barh(y + 0.18, dropout, height=0.34, label="Any later absence")
    ax_bar.set(yticks=y, yticklabels=names, xlabel="Fraction of examples", xlim=(0, 0.55))
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.legend(fontsize=8, loc="lower right")
    for suffix in ("pdf", "png"):
        fig.savefig(figures_dir / f"corrected_truthfulqa_trajectories.{suffix}", dpi=240)
    plt.close(fig)


def plot_task_format(artifacts: list[dict], figures_dir: Path) -> None:
    selected = [
        item
        for item in artifacts
        if item["experiment"]["model_key"] == "Qwen2.5-7B"
    ]
    order = ["truthfulqa", "mmlu", "sst2"]
    selected.sort(key=lambda item: order.index(item["experiment"]["dataset"]))
    labels = {"truthfulqa": "TruthfulQA answer", "mmlu": "MMLU label", "sst2": "SST-2 label"}

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    for artifact in selected:
        dataset = artifact["experiment"]["dataset"]
        depths, prevalence = prevalence_curve(artifact["per_sample_results"])
        ax.plot(depths, prevalence, linewidth=2.4, label=labels[dataset])
    ax.set(xlabel="Normalized layer depth", ylabel="Fraction in vocabulary top-10", xlim=(0, 1), ylim=(0, 1.03))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")
    for suffix in ("pdf", "png"):
        fig.savefig(figures_dir / f"corrected_task_format.{suffix}", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BLACKBOX_ROOT / "results" / "corrected_fep",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BLACKBOX_ROOT / "results" / "analysis"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=BLACKBOX_ROOT / "paper" / "figures"
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=BLACKBOX_ROOT / "results" / "prompt_sensitivity",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts(args.input_dir)
    build_tables(artifacts, args.output_dir)
    build_prompt_table(artifacts, args.prompt_dir, args.output_dir)
    plot_truthfulqa(artifacts, args.figures_dir)
    plot_task_format(artifacts, args.figures_dir)
    print(f"Analyzed {len(artifacts)} canonical artifacts")


if __name__ == "__main__":
    main()
