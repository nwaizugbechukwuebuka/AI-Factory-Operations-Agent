"""Unit tests for retrieval evaluation metrics."""

import pytest

from evaluations.metrics import (
    mean_reciprocal_rank,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_at_k_when_expected_source_is_first() -> None:
    ranked_sources = ["safety.md", "operations.md", "quality.md"]

    assert recall_at_k(ranked_sources, "safety.md", 3) == 1.0


def test_recall_at_k_when_expected_source_is_missing() -> None:
    ranked_sources = ["operations.md", "quality.md", "maintenance.md"]

    assert recall_at_k(ranked_sources, "safety.md", 3) == 0.0


def test_recall_at_k_only_considers_top_k() -> None:
    ranked_sources = ["operations.md", "quality.md", "safety.md"]

    assert recall_at_k(ranked_sources, "safety.md", 2) == 0.0


def test_recall_at_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["safety.md"], "safety.md", 0)


def test_reciprocal_rank_for_first_result() -> None:
    assert reciprocal_rank(
        ["safety.md", "quality.md"],
        "safety.md",
    ) == 1.0


def test_reciprocal_rank_for_second_result() -> None:
    assert reciprocal_rank(
        ["quality.md", "safety.md"],
        "safety.md",
    ) == 0.5


def test_reciprocal_rank_when_source_is_missing() -> None:
    assert reciprocal_rank(
        ["quality.md", "operations.md"],
        "safety.md",
    ) == 0.0


def test_mean_reciprocal_rank() -> None:
    ranked_results = [
        ["safety.md", "quality.md"],
        ["quality.md", "maintenance.md"],
        ["operations.md", "safety.md"],
    ]

    expected_sources = [
        "safety.md",
        "safety.md",
        "safety.md",
    ]

    assert mean_reciprocal_rank(
        ranked_results,
        expected_sources,
    ) == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_mean_reciprocal_rank_with_no_queries() -> None:
    assert mean_reciprocal_rank([], []) == 0.0


def test_mean_reciprocal_rank_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        mean_reciprocal_rank(
            [["safety.md"]],
            [],
        )