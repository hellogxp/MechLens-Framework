"""Tests for post-hoc trajectory analysis."""

import pytest

from mechlens.fep_analysis import (
    benjamini_hochberg_adjusted_pvalues,
    exact_sign_test_pvalue,
    holm_adjusted_pvalues,
    mcnemar_exact_pvalue,
    summarize_rank_trajectories,
    trajectory_at_k,
    wilson_interval,
)


def test_truthfulqa_prompt_templates_are_semantically_distinct():
    from mechlens.fep import render_truthfulqa_prompt

    question = "Where is Paris?"
    assert render_truthfulqa_prompt(question, "qa") == "Q: Where is Paris?\nA:"
    assert render_truthfulqa_prompt(question, "question_answer").endswith("\nAnswer:")
    assert "truthfully" in render_truthfulqa_prompt(question, "instruction")


def test_trajectory_at_k_keeps_censoring_and_persistence_distinct():
    transient = trajectory_at_k([20, 3, 12, 15], top_k=10)
    assert transient["observed"] is True
    assert transient["first_layer_number"] == 2
    assert transient["persistent_layer_number"] is None
    assert transient["dropout_after_entry"] is True

    censored = trajectory_at_k([20, 30, 12, 15], top_k=10)
    assert censored["observed"] is False
    assert censored["first_layer_number"] is None


def test_trajectory_at_k_finds_stable_suffix():
    result = trajectory_at_k([20, 3, 12, 2, 1], top_k=10)
    assert result["first_layer_number"] == 2
    assert result["persistent_layer_number"] == 4
    assert result["persistent_depth"] == pytest.approx(0.8)


def test_trajectory_mask_treats_one_readout_as_missing_in_original_coordinates():
    result = trajectory_at_k([20, 3, 50, 2, 1], top_k=10, ignored_layer_index=2)
    assert result["dropout_after_entry"] is False
    assert result["persistent_layer_number"] == 2
    assert result["persistent_depth"] == pytest.approx(0.4)


def test_trajectory_distinguishes_any_gap_from_final_disappearance():
    recovered = trajectory_at_k([20, 3, 50, 2], top_k=10)
    disappeared = trajectory_at_k([20, 3, 50, 20], top_k=10)
    assert recovered["dropout_after_entry"] is True
    assert recovered["entered_then_disappeared"] is False
    assert disappeared["entered_then_disappeared"] is True


def test_wilson_interval_contains_empirical_fraction():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high


def test_exact_mcnemar_is_symmetric_and_handles_no_disagreement():
    assert mcnemar_exact_pvalue(0, 0) == 1.0
    assert mcnemar_exact_pvalue(10, 2) == mcnemar_exact_pvalue(2, 10)
    assert mcnemar_exact_pvalue(10, 2) < 0.05


def test_exact_sign_test_discards_ties_before_calling_helper():
    assert exact_sign_test_pvalue(0, 0) == 1.0
    assert exact_sign_test_pvalue(10, 2) == mcnemar_exact_pvalue(10, 2)


def test_multiple_comparison_adjustments_preserve_input_order():
    pvalues = [0.04, 0.001, 0.03]
    assert holm_adjusted_pvalues(pvalues) == pytest.approx([0.06, 0.003, 0.06])
    assert benjamini_hochberg_adjusted_pvalues(pvalues) == pytest.approx(
        [0.04, 0.003, 0.04]
    )


def test_aggregate_conditions_depth_means_on_observed_populations():
    samples = [
        {"layer_ranks": [20, 3, 2, 1]},
        {"layer_ranks": [20, 30, 12, 15]},
        {"layer_ranks": [3, 20, 2, 1]},
    ]
    result = summarize_rank_trajectories(samples, top_k=10)
    assert result["observed_pct"] == pytest.approx(2 / 3)
    assert result["never_observed_pct"] == pytest.approx(1 / 3)
    assert result["mean_first_depth_observed"] == pytest.approx(0.375)
    assert result["mean_persistent_depth_final"] == pytest.approx(0.625)
