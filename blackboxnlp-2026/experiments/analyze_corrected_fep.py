#!/usr/bin/env python3
"""Create analysis tables and figures from corrected, canonical FEP artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
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
    frequent_target_types,
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

TOP_K_SWEEP = (1, 3, 5, 10, 20, 50, 100)


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

    # The reporting checklist asks for threshold sensitivity, so the sweep covers
    # every TruthfulQA model rather than the one printed in the paper table.
    sensitivity = [
        {
            "model": DISPLAY_NAMES.get(
                artifact["experiment"]["model_key"], artifact["experiment"]["model_key"]
            ),
            **summarize_rank_trajectories(artifact["per_sample_results"], top_k=top_k),
        }
        for artifact in artifacts
        if artifact["experiment"]["dataset"] == "truthfulqa"
        for top_k in TOP_K_SWEEP
    ]
    write_csv(output_dir / "topk_sensitivity.csv", sensitivity)

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
    build_first_token_strata(artifacts, output_dir)
    build_pairwise_first_entry_depth_by_stratum(artifacts, output_dir)
    build_probability_criterion(artifacts, output_dir)
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


def frequent_first_tokens(samples: list[dict], top_n: int = 10) -> set[str]:
    """Return the most frequent lexical first-token types in one artifact."""

    return frequent_target_types(
        [sample["target_token_str"] for sample in samples], top_n
    )


def build_first_token_strata(artifacts: list[dict], output_dir: Path) -> None:
    """Split TruthfulQA targets into frequent and tail first-token types.

    The answer-prefix target distribution is concentrated, so readout depth may
    track target frequency rather than the model.  Each model is stratified by
    its own ten most frequent lexical first tokens.
    """

    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    rows = []
    for artifact in truth:
        model = DISPLAY_NAMES.get(
            artifact["experiment"]["model_key"], artifact["experiment"]["model_key"]
        )
        samples = artifact["per_sample_results"]
        frequent = frequent_first_tokens(samples)
        in_frequent = [
            sample
            for sample in samples
            if sample["target_token_str"].strip() in frequent
        ]
        in_tail = [
            sample
            for sample in samples
            if sample["target_token_str"].strip() not in frequent
        ]
        frequent_summary = summarize_rank_trajectories(in_frequent, top_k=10)
        tail_summary = summarize_rank_trajectories(in_tail, top_k=10)
        rows.append(
            {
                "model": model,
                "n_frequent": frequent_summary["n"],
                "frequent_first_depth": frequent_summary["mean_first_depth_observed"],
                "frequent_never_pct": frequent_summary["never_observed_pct"],
                "frequent_dropout_pct": frequent_summary["dropout_pct"],
                "n_tail": tail_summary["n"],
                "tail_first_depth": tail_summary["mean_first_depth_observed"],
                "tail_never_pct": tail_summary["never_observed_pct"],
                "tail_dropout_pct": tail_summary["dropout_pct"],
                "first_depth_frequent_minus_tail": (
                    frequent_summary["mean_first_depth_observed"]
                    - tail_summary["mean_first_depth_observed"]
                ),
            }
        )
    write_csv(output_dir / "first_token_strata.csv", rows)


def stratified_depth_comparisons(
    truth: list[dict], top_n: int
) -> list[dict]:
    """Compare first-entry depths inside first-token frequency strata.

    Items whose stratum membership differs between the two tokenizers are
    dropped so that each comparison uses one unambiguous target population.
    Holm correction is applied within each stratum.
    """

    frequent_sets = {
        item["_path"]: frequent_first_tokens(item["per_sample_results"], top_n)
        for item in truth
    }
    rows = []
    for first, second in combinations(truth, 2):
        first_by_id = {str(sample["id"]): sample for sample in first["per_sample_results"]}
        second_by_id = {
            str(sample["id"]): sample for sample in second["per_sample_results"]
        }
        if first_by_id.keys() != second_by_id.keys():
            raise ValueError("TruthfulQA artifacts do not contain identical sample IDs")
        first_frequent = frequent_sets[first["_path"]]
        second_frequent = frequent_sets[second["_path"]]
        for stratum, wants_frequent in (("frequent_types", True), ("tail_types", False)):
            paired = []
            ambiguous = 0
            for sample_id, first_sample in first_by_id.items():
                second_sample = second_by_id[sample_id]
                first_is_frequent = (
                    first_sample["target_token_str"].strip() in first_frequent
                )
                second_is_frequent = (
                    second_sample["target_token_str"].strip() in second_frequent
                )
                if first_is_frequent != second_is_frequent:
                    ambiguous += 1
                    continue
                if first_is_frequent != wants_frequent:
                    continue
                if first_sample["fep_observed"] and second_sample["fep_observed"]:
                    paired.append(
                        (first_sample["fep_depth"], second_sample["fep_depth"])
                    )
            a_earlier = sum(a < b for a, b in paired)
            b_earlier = sum(a > b for a, b in paired)
            rows.append(
                {
                    "frequency_cut": top_n,
                    "model_a": DISPLAY_NAMES.get(
                        first["experiment"]["model_key"], first["experiment"]["model_key"]
                    ),
                    "model_b": DISPLAY_NAMES.get(
                        second["experiment"]["model_key"],
                        second["experiment"]["model_key"],
                    ),
                    "stratum": stratum,
                    "ambiguous_membership_n": ambiguous,
                    "jointly_observed_n": len(paired),
                    "model_a_mean_depth": np.mean([a for a, _ in paired]),
                    "model_b_mean_depth": np.mean([b for _, b in paired]),
                    "model_a_earlier_n": a_earlier,
                    "model_b_earlier_n": b_earlier,
                    "ties_n": sum(a == b for a, b in paired),
                    "exact_sign_p": exact_sign_test_pvalue(a_earlier, b_earlier),
                }
            )
    for stratum in ("frequent_types", "tail_types"):
        selected = [row for row in rows if row["stratum"] == stratum]
        adjusted = holm_adjusted_pvalues([row["exact_sign_p"] for row in selected])
        for row, value in zip(selected, adjusted, strict=True):
            row["holm_adjusted_p"] = value
            row["holm_reject_0.05"] = value <= 0.05
    return rows


def build_pairwise_first_entry_depth_by_stratum(
    artifacts: list[dict], output_dir: Path
) -> None:
    """Release the reported stratified comparison and its cut sensitivity.

    The reported cut is ten types, matching the target audit.  The sweep shows
    whether the stratified conclusions depend on that choice.
    """

    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    write_csv(
        output_dir / "pairwise_first_entry_depth_by_stratum.csv",
        stratified_depth_comparisons(truth, top_n=10),
    )

    sweep_rows = []
    for top_n in (5, 10, 20, 30):
        rows = stratified_depth_comparisons(truth, top_n=top_n)
        by_pair: dict[tuple[str, str], dict[str, dict]] = {}
        for row in rows:
            by_pair.setdefault((row["model_a"], row["model_b"]), {})[row["stratum"]] = row
        reversals = 0
        for strata in by_pair.values():
            frequent, tail = strata["frequent_types"], strata["tail_types"]
            frequent_gap = frequent["model_a_mean_depth"] - frequent["model_b_mean_depth"]
            tail_gap = tail["model_a_mean_depth"] - tail["model_b_mean_depth"]
            if (
                frequent_gap * tail_gap < 0
                and frequent["holm_reject_0.05"]
                and tail["holm_reject_0.05"]
            ):
                reversals += 1
        sweep_rows.append(
            {
                "frequency_cut": top_n,
                "n_pairs": len(by_pair),
                "holm_significant_frequent": sum(
                    row["holm_reject_0.05"]
                    for row in rows
                    if row["stratum"] == "frequent_types"
                ),
                "holm_significant_tail": sum(
                    row["holm_reject_0.05"]
                    for row in rows
                    if row["stratum"] == "tail_types"
                ),
                "sign_reversals_significant_in_both": reversals,
            }
        )
    write_csv(output_dir / "first_token_cut_sensitivity.csv", sweep_rows)


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


PROBABILITY_CRITERIA = (0.05, 0.10)


def build_probability_criterion(artifacts: list[dict], output_dir: Path) -> None:
    """Compare the rank criterion with a probability criterion.

    Top-$k$ membership in a large vocabulary can coexist with almost no
    probability mass, so the stored per-layer target probabilities are used to
    ask what the rank-based event is worth and how the timing changes when the
    criterion is a probability floor instead.
    """

    truth = [item for item in artifacts if item["experiment"]["dataset"] == "truthfulqa"]
    rows = []
    for artifact in truth:
        samples = artifact["per_sample_results"]
        n_layers = len(samples[0]["layer_ranks"])
        observed = [sample for sample in samples if sample["fep_observed"]]
        entry_probability = np.asarray(
            [sample["layer_probs"][sample["fep_layer"]] for sample in observed]
        )
        row = {
            "model": DISPLAY_NAMES.get(
                artifact["experiment"]["model_key"], artifact["experiment"]["model_key"]
            ),
            "n_observed": len(observed),
            "median_prob_at_rank_entry": float(np.median(entry_probability)),
            "median_final_prob_observed": float(
                np.median([sample["final_prob"] for sample in observed])
            ),
            "share_entry_below_1pct": float((entry_probability < 0.01).mean()),
            "rank_first_depth": float(
                np.mean([(sample["fep_layer"] + 1) / n_layers for sample in observed])
            ),
            "rank_coverage": len(observed) / len(samples),
        }
        for floor in PROBABILITY_CRITERIA:
            # Compare the two criteria on the samples that satisfy both, so the
            # shift is a within-sample difference rather than a difference
            # between two conditional populations.
            reached = []
            paired_shift = []
            for sample in samples:
                position = next(
                    (
                        index
                        for index, value in enumerate(sample["layer_probs"])
                        if value >= floor
                    ),
                    None,
                )
                if position is None:
                    continue
                depth = (position + 1) / n_layers
                reached.append(depth)
                if sample["fep_observed"]:
                    paired_shift.append(
                        depth - (sample["fep_layer"] + 1) / n_layers
                    )
            tag = f"p{int(floor * 100)}"
            row[f"{tag}_first_depth"] = float(np.mean(reached)) if reached else ""
            row[f"{tag}_coverage"] = len(reached) / len(samples)
            row[f"{tag}_paired_n"] = len(paired_shift)
            row[f"{tag}_paired_depth_shift"] = (
                float(np.mean(paired_shift)) if paired_shift else ""
            )
        rows.append(row)
    rows.sort(key=lambda item: item["p5_paired_depth_shift"])
    write_csv(output_dir / "probability_criterion.csv", rows)


def format_pvalue(value: float) -> str:
    """Render a p-value for LaTeX without losing very small exponents."""

    if value >= 0.001:
        return f"{value:.3f}".lstrip("0")
    exponent = math.floor(math.log10(value))
    mantissa = value / 10**exponent
    return f"${mantissa:.1f}{{\\times}}10^{{{exponent}}}$"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_latex(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_appendix_tables(analysis_dir: Path, tables_dir: Path) -> None:
    """Emit appendix tabulars from the released CSVs.

    The paper summarises these results in one sentence each; the appendix prints
    them so that a reader of the PDF alone can check every count.  Generating the
    LaTeX from the same CSVs removes any chance of transcription drift.
    """

    model_order = [
        "Llama-3.1-8B",
        "Mistral-7B",
        "Qwen2.5-14B",
        "Qwen2.5-7B",
        "Gemma-7B",
        "Pythia-6.9B",
    ]

    sweep = read_csv_rows(analysis_dir / "topk_sensitivity.csv")
    indexed = {(row["model"], int(row["top_k"])): row for row in sweep}
    header = " & ".join(["Model"] + [f"$k{{=}}{k}$" for k in TOP_K_SWEEP])
    lines = [
        "\\begin{tabular}{l" + "r" * len(TOP_K_SWEEP) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        f"\\multicolumn{{{len(TOP_K_SWEEP) + 1}}}{{l}}{{\\emph{{Non-entry rate (\\%)}}}} \\\\",
    ]
    for model in model_order:
        values = [
            f"{float(indexed[(model, k)]['never_observed_pct']) * 100:.1f}"
            for k in TOP_K_SWEEP
        ]
        lines.append(" & ".join([model] + values) + " \\\\")
    lines.append("\\midrule")
    lines.append(
        f"\\multicolumn{{{len(TOP_K_SWEEP) + 1}}}{{l}}"
        "{\\emph{Conditional first-entry depth}} \\\\"
    )
    for model in model_order:
        values = [
            f"{float(indexed[(model, k)]['mean_first_depth_observed']):.3f}".lstrip("0")
            for k in TOP_K_SWEEP
        ]
        lines.append(" & ".join([model] + values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(tables_dir / "appendix_topk.tex", lines)

    visibility = read_csv_rows(analysis_dir / "pairwise_final_visibility.csv")
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Model A & Model B & A only & B only & $p$ & Holm & BH \\\\",
        "\\midrule",
    ]
    for row in visibility:
        lines.append(
            " & ".join(
                [
                    row["model_a"],
                    row["model_b"],
                    row["a_only_final_top10"],
                    row["b_only_final_top10"],
                    format_pvalue(float(row["mcnemar_exact_p"])),
                    format_pvalue(float(row["holm_adjusted_p"])),
                    format_pvalue(float(row["bh_adjusted_p"])),
                ]
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(tables_dir / "appendix_pairwise_visibility.tex", lines)

    pooled = {
        (row["model_a"], row["model_b"]): row
        for row in read_csv_rows(analysis_dir / "pairwise_first_entry_depth.csv")
    }
    strata: dict[tuple[str, str], dict[str, dict]] = {}
    for row in read_csv_rows(analysis_dir / "pairwise_first_entry_depth_by_stratum.csv"):
        strata.setdefault((row["model_a"], row["model_b"]), {})[row["stratum"]] = row
    lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "& & & \\multicolumn{2}{c}{Pooled} & \\multicolumn{2}{c}{Frequent}"
        " & \\multicolumn{2}{c}{Tail} \\\\",
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
        "Model A & Model B & $n$ & $\\Delta$ & Holm & $\\Delta$ & Holm"
        " & $\\Delta$ & Holm \\\\",
        "\\midrule",
    ]
    for key, pooled_row in pooled.items():
        frequent = strata[key]["frequent_types"]
        tail = strata[key]["tail_types"]

        def gap(row: dict) -> float:
            return float(row["model_a_mean_depth"]) - float(row["model_b_mean_depth"])

        reversed_sign = (
            gap(frequent) * gap(tail) < 0
            and frequent["holm_reject_0.05"] == "True"
            and tail["holm_reject_0.05"] == "True"
        )
        cells = [key[0], key[1] + ("$^{\\dagger}$" if reversed_sign else "")]
        cells.append(pooled_row["jointly_observed_n"])
        for row in (pooled_row, frequent, tail):
            cells.append("$" + f"{gap(row):+.3f}".replace("0.", ".") + "$")
            cells.append(format_pvalue(float(row["holm_adjusted_p"])))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(tables_dir / "appendix_pairwise_depth.tex", lines)

    cuts = read_csv_rows(analysis_dir / "first_token_cut_sensitivity.csv")
    lines = [
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "Cut & Holm sig.\\ (frequent) & Holm sig.\\ (tail) & Sign reversals \\\\",
        "\\midrule",
    ]
    for row in cuts:
        lines.append(
            " & ".join(
                [
                    row["frequency_cut"],
                    f"{row['holm_significant_frequent']}/{row['n_pairs']}",
                    f"{row['holm_significant_tail']}/{row['n_pairs']}",
                    f"{row['sign_reversals_significant_in_both']}/{row['n_pairs']}",
                ]
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(tables_dir / "appendix_cut_sensitivity.tex", lines)

    criterion = read_csv_rows(analysis_dir / "probability_criterion.csv")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Median $p$ & $p{<}.01$ & Cover & Paired $\\Delta$ \\\\",
        "\\midrule",
    ]
    for row in criterion:
        lines.append(
            " & ".join(
                [
                    row["model"],
                    f"{float(row['median_prob_at_rank_entry']):.3f}".lstrip("0"),
                    f"{float(row['share_entry_below_1pct']) * 100:.1f}",
                    f"{float(row['p5_coverage']) * 100:.1f}",
                    "$"
                    + f"{float(row['p5_paired_depth_shift']):+.3f}".replace("0.", ".")
                    + "$",
                ]
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(tables_dir / "probability_criterion.tex", lines)


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


def assert_legend_clear(figure, axes, legend, panel: str) -> None:
    """Fail if a legend box overlaps plotted data.

    A legend that hides a bar or a curve silently misreports the figure, and the
    overlap is easy to miss once the figure is scaled into a column.  The check
    therefore runs at generation time instead of relying on visual inspection.
    """

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = legend.get_window_extent(renderer)
    collisions = []
    for line in axes.get_lines():
        covered = [
            point
            for point in line.get_transform().transform(line.get_xydata())
            if frame.x0 <= point[0] <= frame.x1 and frame.y0 <= point[1] <= frame.y1
        ]
        if covered:
            collisions.append(f"curve {line.get_label()} ({len(covered)} points)")
    for patch in axes.patches:
        if patch.get_window_extent(renderer).overlaps(frame):
            collisions.append(f"bar ending at x={patch.get_width():.3f}")
    if collisions:
        raise RuntimeError(
            f"Legend in {panel} covers plotted data: {'; '.join(collisions)}. "
            "Move the legend or widen the axis limits."
        )


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
    curve_legend = ax_curve.legend(fontsize=7, ncol=2, loc="upper left")

    names = [DISPLAY_NAMES[item["experiment"]["model_key"]] for item in truth]
    never = [item["summary"]["never_observed_pct"] for item in truth]
    dropout = [item["summary"]["dropout_after_entry_pct"] for item in truth]
    y = np.arange(len(names))
    ax_bar.barh(y - 0.18, never, height=0.34, label="Never observed")
    dropout_bars = ax_bar.barh(y + 0.18, dropout, height=0.34, label="Any later absence")
    # Gemma's rate is dominated by one penultimate readout; mark it as fragile.
    fragile_index = names.index(DISPLAY_NAMES["gemma-7b"])
    dropout_bars[fragile_index].set_hatch("///")
    dropout_bars[fragile_index].set_edgecolor("white")
    ax_bar.set(yticks=y, yticklabels=names, xlabel="Fraction of examples", xlim=(0, 0.62))
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.2)
    # Gemma's any-later-absence bar reaches .48, so the legend has to sit at the
    # top where the longest bar is .31.
    bar_legend = ax_bar.legend(fontsize=8, loc="upper right", framealpha=0.95)
    assert_legend_clear(fig, ax_curve, curve_legend, "TruthfulQA prevalence panel")
    assert_legend_clear(fig, ax_bar, bar_legend, "TruthfulQA event-rate panel")
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
    legend = ax.legend(fontsize=8, loc="upper left")
    assert_legend_clear(fig, ax, legend, "task-format panel")
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
        "--figures-dir", type=Path, default=BLACKBOX_ROOT / "generated" / "figures"
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=BLACKBOX_ROOT / "results" / "prompt_sensitivity",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=BLACKBOX_ROOT / "generated" / "tables"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts(args.input_dir)
    build_tables(artifacts, args.output_dir)
    build_prompt_table(artifacts, args.prompt_dir, args.output_dir)
    build_appendix_tables(args.output_dir, args.tables_dir)
    plot_truthfulqa(artifacts, args.figures_dir)
    plot_task_format(artifacts, args.figures_dir)
    print(f"Analyzed {len(artifacts)} canonical artifacts")


if __name__ == "__main__":
    main()
