"""Metrics for evaluating factory knowledge retrieval quality."""

from collections.abc import Sequence


def recall_at_k(
    ranked_sources: Sequence[str],
    expected_source: str,
    k: int,
) -> float:
    """Return 1.0 if the expected source appears in the top k results."""
    if k <= 0:
        raise ValueError("k must be a positive integer")

    return float(expected_source in ranked_sources[:k])


def reciprocal_rank(
    ranked_sources: Sequence[str],
    expected_source: str,
) -> float:
    """Return the reciprocal rank of the first occurrence of the expected source."""
    for rank, source in enumerate(ranked_sources, start=1):
        if source == expected_source:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_sources_list: Sequence[Sequence[str]],
    expected_sources: Sequence[str],
) -> float:
    """Calculate MRR across multiple queries and their expected source documents."""
    if len(ranked_sources_list) != len(expected_sources):
        raise ValueError(
            "ranked_sources_list and expected_sources must have equal lengths"
        )

    if not expected_sources:
        raise ValueError("Evaluation dataset must not be empty")

    reciprocal_ranks = [
        reciprocal_rank(ranked_sources, expected_source)
        for ranked_sources, expected_source in zip(
            ranked_sources_list,
            expected_sources,
            strict=True,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def mean_recall_at_k(
    ranked_sources_list: Sequence[Sequence[str]],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Calculate average Recall@K across multiple queries."""
    if k <= 0:
        raise ValueError("k must be a positive integer")

    if len(ranked_sources_list) != len(expected_sources):
        raise ValueError(
            "ranked_sources_list and expected_sources must have equal lengths"
        )

    if not expected_sources:
        raise ValueError("Evaluation dataset must not be empty")

    recall_scores = [
        recall_at_k(ranked_sources, expected_source, k)
        for ranked_sources, expected_source in zip(
            ranked_sources_list,
            expected_sources,
            strict=True,
        )
    ]

    return sum(recall_scores) / len(recall_scores)