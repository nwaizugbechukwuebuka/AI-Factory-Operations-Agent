"""Integration tests for factory knowledge retrieval quality."""

import json
from pathlib import Path

import pytest

from backend.app.ai.embeddings import EmbeddingModel
from backend.app.ai.retrieval import VectorStore
from evaluations.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "factory"
CASES_PATH = (
    PROJECT_ROOT
    / "evaluations"
    / "cases"
    / "retrieval_cases.json"
)

TOP_K = 3


@pytest.fixture(scope="module")
def evaluation_cases() -> list[dict[str, str]]:
    """Load the factory retrieval evaluation dataset."""
    if not CASES_PATH.is_file():
        pytest.fail(
            f"Evaluation dataset not found: {CASES_PATH}"
        )

    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(cases, list) or not cases:
        pytest.fail(
            "The evaluation dataset must be a non-empty JSON list."
        )

    required_fields = {
        "id",
        "query",
        "expected_source",
    }

    for case in cases:
        if not isinstance(case, dict):
            pytest.fail(
                "Every evaluation case must be a JSON object."
            )

        missing_fields = required_fields - case.keys()

        if missing_fields:
            pytest.fail(
                f"Evaluation case is missing fields: "
                f"{sorted(missing_fields)}"
            )

        for field in required_fields:
            if not isinstance(case[field], str) or not case[field].strip():
                pytest.fail(
                    f"Case {case.get('id', '<unknown>')} "
                    f"has an invalid {field!r}."
                )

    case_ids = [case["id"] for case in cases]

    if len(case_ids) != len(set(case_ids)):
        pytest.fail(
            "Evaluation case IDs must be unique."
        )

    return cases


@pytest.fixture(scope="module")
def retrieval_components() -> tuple[EmbeddingModel, VectorStore]:
    """Load the embedding model and existing FAISS index."""
    if not (INDEX_PATH / "factory.index").is_file():
        pytest.fail(
            "Factory FAISS index not found. "
            "Build it first with: "
            "python -m scripts.build_index"
        )

    if not (INDEX_PATH / "chunks.json").is_file():
        pytest.fail(
            "Factory chunk metadata not found. "
            "Rebuild the index with: "
            "python -m scripts.build_index"
        )

    embedding_model = EmbeddingModel()
    vector_store = VectorStore.load(INDEX_PATH)

    if vector_store.index.ntotal == 0:
        pytest.fail(
            "The factory vector index contains no embeddings."
        )

    return embedding_model, vector_store


@pytest.fixture(scope="module")
def retrieval_results(
    evaluation_cases: list[dict[str, str]],
    retrieval_components: tuple[EmbeddingModel, VectorStore],
) -> list[dict[str, object]]:
    """Retrieve and score documents for every evaluation query."""
    embedding_model, vector_store = retrieval_components
    results: list[dict[str, object]] = []

    available_sources = {
        chunk["source"]
        for chunk in vector_store.chunks
    }

    for case in evaluation_cases:
        expected_source = case["expected_source"]

        if expected_source not in available_sources:
            pytest.fail(
                f"Case {case['id']} expects {expected_source!r}, "
                "but that source is absent from the loaded index. "
                "Check data/documents and rebuild the index."
            )

        query_embedding = embedding_model.encode(
            [case["query"]]
        )[0]

        retrieved_chunks = vector_store.search(
            query_embedding,
            top_k=TOP_K,
        )

        ranked_sources = [
            str(chunk["source"])
            for chunk in retrieved_chunks
        ]

        results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_source": expected_source,
                "ranked_sources": ranked_sources,
                "recall_at_1": recall_at_k(
                    ranked_sources,
                    expected_source,
                    1,
                ),
                "recall_at_3": recall_at_k(
                    ranked_sources,
                    expected_source,
                    TOP_K,
                ),
                "reciprocal_rank": reciprocal_rank(
                    ranked_sources,
                    expected_source,
                ),
            }
        )

    return results


def test_evaluation_dataset_is_not_empty(
    evaluation_cases: list[dict[str, str]],
) -> None:
    """Verify that evaluation cases have been configured."""
    assert evaluation_cases


def test_factory_index_contains_documents(
    retrieval_components: tuple[EmbeddingModel, VectorStore],
) -> None:
    """Verify that the loaded index contains searchable chunks."""
    _, vector_store = retrieval_components

    assert vector_store.index.ntotal > 0
    assert len(vector_store.chunks) == vector_store.index.ntotal


def test_every_query_returns_results(
    retrieval_results: list[dict[str, object]],
) -> None:
    """Verify that every evaluation query returns search results."""
    for result in retrieval_results:
        assert result["ranked_sources"], (
            f"{result['id']} returned no documents: "
            f"{result['query']}"
        )


def test_expected_documents_are_retrieved_in_top_three(
    retrieval_results: list[dict[str, object]],
) -> None:
    """Verify that every expected source appears in the top three."""
    failures = [
        (
            f"{result['id']}: expected "
            f"{result['expected_source']!r}, got "
            f"{result['ranked_sources']!r}"
        )
        for result in retrieval_results
        if result["recall_at_3"] != 1.0
    ]

    assert not failures, (
        "Retrieval evaluation failed:\n"
        + "\n".join(failures)
    )


def test_recall_at_one_meets_baseline(
    retrieval_results: list[dict[str, object]],
) -> None:
    """Require at least 80% Recall@1 across evaluation queries."""
    recall_at_one = sum(
        float(result["recall_at_1"])
        for result in retrieval_results
    ) / len(retrieval_results)

    assert recall_at_one >= 0.80, (
        f"Recall@1 was {recall_at_one:.3f}; "
        "the required baseline is 0.800."
    )


def test_mean_reciprocal_rank_meets_baseline(
    retrieval_results: list[dict[str, object]],
) -> None:
    """Require MRR of at least 0.80."""
    ranked_results = [
        result["ranked_sources"]
        for result in retrieval_results
    ]

    expected_sources = [
        str(result["expected_source"])
        for result in retrieval_results
    ]

    mrr = mean_reciprocal_rank(
        ranked_results,
        expected_sources,
    )

    assert mrr >= 0.80, (
        f"Mean Reciprocal Rank was {mrr:.3f}; "
        "the required baseline is 0.800."
    )


def test_retrieval_scores_are_valid(
    retrieval_results: list[dict[str, object]],
) -> None:
    """Verify that retrieval metrics remain within valid bounds."""
    for result in retrieval_results:
        assert 0.0 <= float(result["recall_at_1"]) <= 1.0
        assert 0.0 <= float(result["recall_at_3"]) <= 1.0
        assert 0.0 <= float(result["reciprocal_rank"]) <= 1.0