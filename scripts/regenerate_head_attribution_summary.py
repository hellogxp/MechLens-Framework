"""
Regenerate head_attribution top-level fields from preserved per_sample_results.

Background
----------
The original `experiments/run_head_attribution.py` (filter version, fixed 2026-06)
contained a regression that discarded all samples as "invalid":

    valid_count = 0
    for r in results:
        if r.get("baseline_final_rank", 999) < 10:
            valid_count += 1

All 200 samples per model had `baseline_final_rank >> 10` (e.g., Qwen first
sample = 4699), so `valid_count` collapsed to 0 and every top-level aggregate
field (`head_mean_rank_change`, `head_frequency_in_top5`, `critical_heads`)
froze at zero. The raw per-sample data in `per_sample_results` — including
`baseline_final_rank`, `ablated_rank`, `rank_change` for every head — was
preserved untouched.

This script restores the top-level aggregations from that preserved raw data,
without re-running any GPU experiment. It mirrors the aggregation logic of
the fixed (no-filter) version of `run_head_attribution.py`:

    head_mean_rank_change[h] = sum over samples of rank_change[h]
                              / n_samples
    head_frequency_in_top5[h] = (# times h appears in top_heads)
                                / n_samples
    critical_heads            = top (n_heads // 10) by head_mean_rank_change

Usage
-----
    python scripts/regenerate_head_attribution_summary.py \
        --results-dir results/head_attribution

Reads each `head_attribution_<model>.json` in `--results-dir`, overwrites
the top-level aggregate fields in place, and regenerates
`head_attribution_summary.json`.

Verification
------------
After running, Gini coefficients computed from the raw data (using
|rank_change|, matching the paper's Table 5 methodology) should reproduce:

    Model                       Gini (recomputed)    Paper Table 5
    Qwen/Qwen2.5-7B             0.0081              0.008
    meta-llama/Llama-3.1-8B     0.0064              0.006
    mistralai/Mistral-7B-v0.1   0.0125              0.013
    EleutherAI/pythia-6.9b      0.0083              0.008
    google/gemma-7b             0.0013              0.001
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def recompute_model_aggregates(data: dict) -> dict:
    """Recompute top-level fields from preserved per_sample_results."""
    n_heads = data["n_heads"]
    n_samples_actual = len(data["per_sample_results"])

    head_mean_rank_change = np.zeros(n_heads, dtype=float)
    head_freq_top5 = np.zeros(n_heads, dtype=float)

    for sample in data["per_sample_results"]:
        for h_info in sample.get("head_attributions", []):
            head_idx = h_info["head"]
            if 0 <= head_idx < n_heads:
                head_mean_rank_change[head_idx] += h_info.get("rank_change", 0.0)
        for h in sample.get("top_heads", []):
            if 0 <= h < n_heads:
                head_freq_top5[h] += 1.0

    n_valid = n_samples_actual
    if n_valid > 0:
        head_mean_rank_change /= n_valid
        head_freq_top5 /= n_valid

    n_critical = max(1, n_heads // 10)
    critical_heads = np.argsort(head_mean_rank_change)[-n_critical:][::-1].tolist()

    data["n_samples"] = n_samples_actual
    data["n_valid"] = n_valid
    data["critical_heads"] = critical_heads
    data["head_mean_rank_change"] = head_mean_rank_change.tolist()
    data["head_frequency_in_top5"] = head_freq_top5.tolist()
    return data


def build_summary(all_results: dict) -> dict:
    """Build summary.json mirroring the original script's summary structure."""
    summary = {}
    for name, res in all_results.items():
        if "error" in res:
            continue
        summary[name] = {
            "n_layers": res["n_layers"],
            "n_heads": res["n_heads"],
            "n_valid_samples": res["n_valid"],
            "critical_heads": res["critical_heads"],
            "top5_head_mean_rank_change": sorted(
                enumerate(res["head_mean_rank_change"]),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }
    return summary


def verify_gini(all_results: dict) -> None:
    """Sanity-check: print Gini from |rank_change|, match against paper Table 5.

    Paper Table 5 reports Gini for the cross-architecture comparison
    (section on head-level attribution). This is informational; if numbers
    diverge, the per_sample_results themselves have drifted.
    """
    paper_reference = {
        "Qwen/Qwen2.5-7B": 0.008,
        "meta-llama/Llama-3.1-8B": 0.006,
        "mistralai/Mistral-7B-v0.1": 0.013,
        "EleutherAI/pythia-6.9b": 0.008,
        "google/gemma-7b": 0.001,
    }

    def gini(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.sum() == 0 or len(x) == 0:
            return 0.0
        x = np.sort(x)
        n = len(x)
        cum = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

    logger.info("Gini verification (computed from |rank_change|, matches paper Table 5):")
    for name, res in all_results.items():
        n_heads = res["n_heads"]
        head_abs_impact = np.zeros(n_heads)
        for sample in res["per_sample_results"]:
            for h_info in sample.get("head_attributions", []):
                idx = h_info["head"]
                if 0 <= idx < n_heads:
                    head_abs_impact[idx] += abs(h_info.get("rank_change", 0.0))
        g = gini(head_abs_impact)
        paper = paper_reference.get(name)
        marker = ""
        if paper is not None:
            marker = " (paper: %.3f, Δ=%.4f)" % (paper, abs(g - paper))
        logger.info(f"  {name:<32} Gini = {g:.4f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/head_attribution"),
        help="Directory containing head_attribution_<model>.json files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recompute and print verification, but do not write files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    model_files = sorted(results_dir.glob("head_attribution_*.json"))
    model_files = [p for p in model_files if "summary" not in p.name]
    if not model_files:
        raise FileNotFoundError(
            f"no head_attribution_<model>.json files in {results_dir}"
        )

    logger.info(f"Found {len(model_files)} per-model result files")
    all_results: dict[str, dict] = {}

    for path in model_files:
        with open(path) as f:
            data = json.load(f)
        before_n_valid = data.get("n_valid", "MISSING")
        before_head_mean = data.get("head_mean_rank_change")
        before_zero = (
            before_head_mean is not None
            and all(abs(v) < 1e-12 for v in before_head_mean)
        )
        logger.info(
            f"  {path.name}: n_samples_in_file={len(data['per_sample_results'])}, "
            f"current n_valid={before_n_valid}, "
            f"head_mean_rank_change_all_zero={before_zero}"
        )
        data = recompute_model_aggregates(data)
        all_results[data["model_name"]] = data

    verify_gini(all_results)

    if args.check_only:
        logger.info("--check-only: not writing files")
        return

    for path in model_files:
        with open(path, "w") as f:
            json.dump(
                all_results[next(
                    k for k, v in all_results.items()
                    if k.replace("/", "_") == path.stem.removeprefix("head_attribution_")
                )],
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Updated: {path}")

    summary = build_summary(all_results)
    summary_path = results_dir / "head_attribution_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
