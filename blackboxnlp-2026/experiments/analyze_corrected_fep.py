#!/usr/bin/env python3
"""Create paper tables and figures from corrected, canonical FEP artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLACKBOX_ROOT = PROJECT_ROOT / "blackboxnlp-2026"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mechlens.fep_analysis import (  # noqa: E402, I001
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
    ax_bar.barh(y + 0.18, dropout, height=0.34, label="Drops after entry")
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
