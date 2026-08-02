"""Pure helpers for reliable Factual Emergence Point measurements.

The original experiments used the final layer as a sentinel for targets that
never entered the top-k.  That makes a censored observation indistinguishable
from a genuine final-layer entry.  This module keeps those states separate and
contains no torch dependency so the measurement contract can be unit tested on
CPU-only development machines.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import fmean, pstdev


class TokenAlignmentError(ValueError):
    """Raised when a continuation cannot be aligned to the tokenized prompt."""


def normalize_continuation(text: str) -> str:
    """Return a non-empty continuation with exactly one leading space.

    The experiment prompts end in a colon.  Explicitly including the separator
    makes the intended generation target unambiguous across tokenizer families.
    The target token must still be extracted from the tokenized full sequence;
    callers must not tokenize this returned string in isolation.
    """

    stripped = text.strip()
    if not stripped:
        raise TokenAlignmentError("The target continuation is empty")
    return f" {stripped}"


def extract_aligned_continuation_ids(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
) -> list[int]:
    """Extract continuation ids after verifying exact prompt-prefix alignment."""

    prompt = [int(token_id) for token_id in prompt_ids]
    full = [int(token_id) for token_id in full_ids]
    if not prompt:
        raise TokenAlignmentError("The tokenized prompt is empty")
    if len(full) <= len(prompt):
        raise TokenAlignmentError("The continuation added no tokens")
    if full[: len(prompt)] != prompt:
        mismatch = next(
            (
                index
                for index, (prompt_id, full_id) in enumerate(zip(prompt, full))
                if prompt_id != full_id
            ),
            min(len(prompt), len(full)),
        )
        raise TokenAlignmentError(
            "Tokenization changed inside the prompt at index "
            f"{mismatch}; the continuation boundary is not stable"
        )
    return full[len(prompt) :]


def summarize_topk_trajectory(layer_in_topk: Sequence[bool]) -> dict:
    """Summarize an observed top-k trajectory without a final-layer sentinel.

    Layer indices are zero-based.  Human-facing layer numbers are one-based.
    ``fep_layer`` is ``None`` when the target never enters the top-k.
    ``persistent_fep_layer`` is the first layer after which the target remains
    in the top-k through the final layer.
    """

    trajectory = [bool(value) for value in layer_in_topk]
    if not trajectory:
        raise ValueError("The layer trajectory is empty")

    observed_indices = [index for index, in_topk in enumerate(trajectory) if in_topk]
    first_entry = observed_indices[0] if observed_indices else None
    last_entry = observed_indices[-1] if observed_indices else None
    final_in_topk = trajectory[-1]

    persistent_entry = None
    if final_in_topk:
        persistent_entry = len(trajectory) - 1
        while persistent_entry > 0 and trajectory[persistent_entry - 1]:
            persistent_entry -= 1

    has_dropout_after_entry = False
    if first_entry is not None:
        has_dropout_after_entry = any(not value for value in trajectory[first_entry + 1 :])

    return {
        "fep_observed": first_entry is not None,
        "fep_layer": first_entry,
        "fep_layer_number": first_entry + 1 if first_entry is not None else None,
        "fep_depth": (first_entry + 1) / len(trajectory) if first_entry is not None else None,
        "last_topk_layer": last_entry,
        "last_topk_layer_number": last_entry + 1 if last_entry is not None else None,
        "persistent_fep_layer": persistent_entry,
        "persistent_fep_layer_number": (
            persistent_entry + 1 if persistent_entry is not None else None
        ),
        "persistent_fep_depth": (
            (persistent_entry + 1) / len(trajectory)
            if persistent_entry is not None
            else None
        ),
        "final_in_topk": final_in_topk,
        "entered_then_disappeared": first_entry is not None and not final_in_topk,
        "has_dropout_after_entry": has_dropout_after_entry,
    }


def aggregate_fep_results(results: Sequence[dict], n_layers: int) -> dict:
    """Aggregate corrected per-sample FEP results with explicit denominators."""

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    n_samples = len(results)
    if n_samples == 0:
        return {"n_samples": 0, "n_layers": n_layers}

    observed = [result for result in results if result["fep_observed"]]
    persistent = [
        result for result in results if result.get("persistent_fep_layer") is not None
    ]
    observed_numbers = [result["fep_layer_number"] for result in observed]
    persistent_numbers = [result["persistent_fep_layer_number"] for result in persistent]

    distribution = Counter(observed_numbers)
    final_topk_count = sum(bool(result["final_in_topk"]) for result in results)
    final_top1_count = sum(result.get("final_rank") == 0 for result in results)
    disappeared_count = sum(bool(result["entered_then_disappeared"]) for result in results)
    dropout_count = sum(bool(result["has_dropout_after_entry"]) for result in results)

    def fraction(count: int) -> float:
        return count / n_samples

    aggregate = {
        "n_samples": n_samples,
        "n_layers": n_layers,
        "observed_count": len(observed),
        "observed_pct": fraction(len(observed)),
        "never_observed_count": n_samples - len(observed),
        "never_observed_pct": fraction(n_samples - len(observed)),
        "final_topk_count": final_topk_count,
        "final_topk_pct": fraction(final_topk_count),
        "final_top1_count": final_top1_count,
        "final_top1_pct": fraction(final_top1_count),
        "entered_then_disappeared_count": disappeared_count,
        "entered_then_disappeared_pct": fraction(disappeared_count),
        "dropout_after_entry_count": dropout_count,
        "dropout_after_entry_pct": fraction(dropout_count),
        "persistent_count": len(persistent),
        "persistent_pct": fraction(len(persistent)),
        "mean_observed_fep_layer_number": (
            fmean(observed_numbers) if observed_numbers else None
        ),
        "std_observed_fep_layer_number": (
            pstdev(observed_numbers) if len(observed_numbers) > 1 else 0.0
            if observed_numbers
            else None
        ),
        "mean_persistent_fep_layer_number": (
            fmean(persistent_numbers) if persistent_numbers else None
        ),
        "observed_fep_distribution": {
            str(layer): count for layer, count in sorted(distribution.items())
        },
    }
    candidate_scored = [result for result in results if "candidate_correct" in result]
    if candidate_scored:
        candidate_correct_count = sum(
            bool(result["candidate_correct"]) for result in candidate_scored
        )
        aggregate.update(
            {
                "candidate_scored_count": len(candidate_scored),
                "candidate_correct_count": candidate_correct_count,
                "candidate_accuracy": candidate_correct_count / len(candidate_scored),
            }
        )
    return aggregate
