"""Analysis helpers for censored layerwise top-k trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean


def mcnemar_exact_pvalue(gains: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""

    if gains < 0 or losses < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    smaller = min(gains, losses)
    lower_tail = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def exact_sign_test_pvalue(less: int, greater: int) -> float:
    """Two-sided exact sign-test p-value after discarding ties."""

    return mcnemar_exact_pvalue(less, greater)


def holm_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""

    _validate_pvalues(pvalues)
    total = len(pvalues)
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * total
    running_max = 0.0
    for rank, (original_index, pvalue) in enumerate(ordered):
        running_max = max(running_max, (total - rank) * pvalue)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def benjamini_hochberg_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    """Return Benjamini--Hochberg FDR adjusted p-values in input order."""

    _validate_pvalues(pvalues)
    total = len(pvalues)
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1], reverse=True)
    adjusted = [0.0] * total
    running_min = 1.0
    for reverse_rank, (original_index, pvalue) in enumerate(ordered):
        rank = total - reverse_rank
        running_min = min(running_min, total * pvalue / rank)
        adjusted[original_index] = min(1.0, running_min)
    return adjusted


def _validate_pvalues(pvalues: Sequence[float]) -> None:
    if not all(0.0 <= pvalue <= 1.0 for pvalue in pvalues):
        raise ValueError("p-values must lie between zero and one")


def trajectory_at_k(
    layer_ranks: Sequence[int], top_k: int, ignored_layer_index: int | None = None
) -> dict:
    """Summarize zero-based vocabulary ranks at a chosen top-k threshold."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not layer_ranks:
        raise ValueError("layer_ranks must not be empty")

    n_layers = len(layer_ranks)
    if ignored_layer_index is not None and not 0 <= ignored_layer_index < n_layers:
        raise ValueError("ignored_layer_index must identify an existing layer")

    included = [
        (index, int(rank) < top_k)
        for index, rank in enumerate(layer_ranks)
        if index != ignored_layer_index
    ]
    if not included:
        raise ValueError("at least one layer must remain after masking")

    entries = [index for index, present in included if present]
    first = entries[0] if entries else None
    final = included[-1][1]

    persistent = None
    if final:
        persistent_position = len(included) - 1
        while persistent_position > 0 and included[persistent_position - 1][1]:
            persistent_position -= 1
        persistent = included[persistent_position][0]

    dropout = first is not None and any(
        not present for index, present in included if index > first
    )
    entered_then_disappeared = first is not None and not final
    return {
        "observed": first is not None,
        "first_layer_number": first + 1 if first is not None else None,
        "first_depth": (first + 1) / n_layers if first is not None else None,
        "persistent_layer_number": persistent + 1 if persistent is not None else None,
        "persistent_depth": (
            (persistent + 1) / n_layers if persistent is not None else None
        ),
        "final_in_topk": final,
        "dropout_after_entry": dropout,
        "entered_then_disappeared": entered_then_disappeared,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - radius, center + radius


def summarize_rank_trajectories(
    samples: Sequence[dict], top_k: int, ignored_layer_index: int | None = None
) -> dict:
    """Aggregate trajectories while keeping never-observed cases censored."""

    trajectories = [
        trajectory_at_k(sample["layer_ranks"], top_k, ignored_layer_index)
        for sample in samples
    ]
    total = len(trajectories)
    if total == 0:
        raise ValueError("samples must not be empty")

    observed = [item for item in trajectories if item["observed"]]
    persistent = [item for item in trajectories if item["persistent_depth"] is not None]
    observed_count = len(observed)
    final_count = sum(item["final_in_topk"] for item in trajectories)
    dropout_count = sum(item["dropout_after_entry"] for item in trajectories)
    disappeared_count = sum(item["entered_then_disappeared"] for item in trajectories)

    def interval(count: int) -> tuple[float, float]:
        return wilson_interval(count, total)

    observed_ci = interval(observed_count)
    final_ci = interval(final_count)
    dropout_ci = interval(dropout_count)
    disappeared_ci = interval(disappeared_count)
    return {
        "n": total,
        "top_k": top_k,
        "observed_pct": observed_count / total,
        "observed_ci_low": observed_ci[0],
        "observed_ci_high": observed_ci[1],
        "never_observed_pct": 1 - observed_count / total,
        "final_topk_pct": final_count / total,
        "final_topk_ci_low": final_ci[0],
        "final_topk_ci_high": final_ci[1],
        "dropout_pct": dropout_count / total,
        "dropout_ci_low": dropout_ci[0],
        "dropout_ci_high": dropout_ci[1],
        "entered_then_disappeared_pct": disappeared_count / total,
        "entered_then_disappeared_ci_low": disappeared_ci[0],
        "entered_then_disappeared_ci_high": disappeared_ci[1],
        "mean_first_depth_observed": (
            fmean(item["first_depth"] for item in observed) if observed else None
        ),
        "mean_persistent_depth_final": (
            fmean(item["persistent_depth"] for item in persistent) if persistent else None
        ),
    }
