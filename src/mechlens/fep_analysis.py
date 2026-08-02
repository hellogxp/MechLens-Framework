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


def trajectory_at_k(layer_ranks: Sequence[int], top_k: int) -> dict:
    """Summarize zero-based vocabulary ranks at a chosen top-k threshold."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not layer_ranks:
        raise ValueError("layer_ranks must not be empty")

    in_topk = [int(rank) < top_k for rank in layer_ranks]
    entries = [index for index, present in enumerate(in_topk) if present]
    first = entries[0] if entries else None
    final = in_topk[-1]

    persistent = None
    if final:
        persistent = len(in_topk) - 1
        while persistent > 0 and in_topk[persistent - 1]:
            persistent -= 1

    dropout = first is not None and any(not value for value in in_topk[first + 1 :])
    return {
        "observed": first is not None,
        "first_layer_number": first + 1 if first is not None else None,
        "first_depth": (first + 1) / len(in_topk) if first is not None else None,
        "persistent_layer_number": persistent + 1 if persistent is not None else None,
        "persistent_depth": (
            (persistent + 1) / len(in_topk) if persistent is not None else None
        ),
        "final_in_topk": final,
        "dropout_after_entry": dropout,
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


def summarize_rank_trajectories(samples: Sequence[dict], top_k: int) -> dict:
    """Aggregate trajectories while keeping never-observed cases censored."""

    trajectories = [trajectory_at_k(sample["layer_ranks"], top_k) for sample in samples]
    total = len(trajectories)
    if total == 0:
        raise ValueError("samples must not be empty")

    observed = [item for item in trajectories if item["observed"]]
    persistent = [item for item in trajectories if item["persistent_depth"] is not None]
    observed_count = len(observed)
    final_count = sum(item["final_in_topk"] for item in trajectories)
    dropout_count = sum(item["dropout_after_entry"] for item in trajectories)

    def interval(count: int) -> tuple[float, float]:
        return wilson_interval(count, total)

    observed_ci = interval(observed_count)
    final_ci = interval(final_count)
    dropout_ci = interval(dropout_count)
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
        "mean_first_depth_observed": (
            fmean(item["first_depth"] for item in observed) if observed else None
        ),
        "mean_persistent_depth_final": (
            fmean(item["persistent_depth"] for item in persistent) if persistent else None
        ),
    }
