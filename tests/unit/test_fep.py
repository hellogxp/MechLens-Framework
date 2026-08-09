"""Tests for the corrected FEP measurement contract."""

import pytest

from mechlens.fep import (
    TokenAlignmentError,
    aggregate_fep_results,
    extract_aligned_continuation_ids,
    normalize_continuation,
    summarize_topk_trajectory,
)


def test_normalize_continuation_uses_one_leading_space():
    assert normalize_continuation("  Paris  ") == " Paris"


def test_normalize_continuation_rejects_empty_target():
    with pytest.raises(TokenAlignmentError, match="empty"):
        normalize_continuation("   ")


def test_extract_aligned_continuation_ids():
    assert extract_aligned_continuation_ids([1, 2, 3], [1, 2, 3, 8, 9]) == [8, 9]


def test_extract_aligned_continuation_rejects_boundary_retokenization():
    with pytest.raises(TokenAlignmentError, match="index 2"):
        extract_aligned_continuation_ids([1, 2, 3], [1, 2, 7, 8])


def test_never_observed_is_not_a_final_layer_event():
    summary = summarize_topk_trajectory([False, False, False])
    assert summary["fep_observed"] is False
    assert summary["fep_layer"] is None
    assert summary["fep_layer_number"] is None
    assert summary["final_in_topk"] is False


def test_transient_entry_is_distinct_from_persistent_entry():
    summary = summarize_topk_trajectory([False, True, False, False])
    assert summary["fep_layer"] == 1
    assert summary["entered_then_disappeared"] is True
    assert summary["persistent_fep_layer"] is None


def test_persistent_entry_reports_first_stable_layer():
    summary = summarize_topk_trajectory([False, True, False, True, True])
    assert summary["fep_layer"] == 1
    assert summary["persistent_fep_layer"] == 3
    assert summary["has_dropout_after_entry"] is True


def test_aggregate_uses_observed_values_only_for_mean():
    results = []
    for trajectory, final_rank in [
        ([False, False, False], 200),
        ([False, True, True], 3),
        ([True, False, False], 100),
    ]:
        result = summarize_topk_trajectory(trajectory)
        result["final_rank"] = final_rank
        results.append(result)

    aggregate = aggregate_fep_results(results, n_layers=3)
    assert aggregate["observed_count"] == 2
    assert aggregate["never_observed_count"] == 1
    assert aggregate["final_topk_count"] == 1
    assert aggregate["entered_then_disappeared_count"] == 1
    assert aggregate["mean_observed_fep_layer_number"] == 1.5


def test_aggregate_reports_candidate_accuracy_when_available():
    results = []
    for is_correct in [True, False, True]:
        result = summarize_topk_trajectory([False, True])
        result.update({"final_rank": 1, "candidate_correct": is_correct})
        results.append(result)

    aggregate = aggregate_fep_results(results, n_layers=2)
    assert aggregate["candidate_scored_count"] == 3
    assert aggregate["candidate_correct_count"] == 2
    assert aggregate["candidate_accuracy"] == pytest.approx(2 / 3)
