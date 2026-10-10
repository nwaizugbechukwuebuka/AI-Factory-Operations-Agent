from collections.abc import Sequence


def recall_at_k(
    retrieved_sources: Sequence[str],
    expected_source: str,
    k: int,
) -> float:
    """Return 1.0 if the expected source appears in the top k results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    return float(expected_source in retrieved_sources[:k])


def reciprocal_rank(
    retrieved_sources: Sequence[str],
    expected_source: str,
) -> float:
    """Return the reciprocal rank of the first expected source."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_source_lists: Sequence[Sequence[str]],
    expected_sources: Sequence[str],
) -> float:
    """Calculate MRR across a collection of evaluation queries."""
    if len(ranked_source_lists) != len(expected_sources):
        raise ValueError(
            "Each query must have exactly one expected source"
        )

    if not ranked_source_lists:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(ranked_sources, expected_source)
        for ranked_sources, expected_source in zip(
            ranked_source_lists,
            expected_sources,
            strict=True,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def mean_recall_at_k(
    ranked_source_lists: Sequence[Sequence[str]],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Calculate average Recall@k across evaluation queries."""
    if len(ranked_source_lists) != len(expected_sources):
        raise ValueError(
            "Each query must have exactly one expected source"
        )

    if not ranked_source_lists:
        raise ValueError("At least one evaluation query is required")

    recalls = [
        recall_at_k(sources, expected, k)
        for sources, expected in zip(
            ranked_source_lists,
            expected_sources,
            strict=True,
        )
    ]

    return sum(recalls) / len(recalls)