import json
from pathlib import Path

import pytest

from backend.app.ai.embeddings import EmbeddingModel
from backend.app.ai.retrieval import VectorStore
from backend.app.data.chunking import chunk_text
from backend.app.data.ingestion import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_PATH = PROJECT_ROOT / "data" / "documents"
EVALUATION_PATH = PROJECT_ROOT / "evaluations" / "retrieval_cases.json"


@pytest.fixture(scope="module")
def retrieval_system() -> tuple[EmbeddingModel, VectorStore]:
    documents = load_documents(DOCUMENT_PATH)

    chunks: list[dict[str, str]] = []

    for document in documents:
        for number, text in enumerate(
            chunk_text(document["text"]),
            start=1,
        ):
            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": (
                        f"{Path(document['source']).stem}-{number:03d}"
                    ),
                    "text": text,
                }
            )

    if not chunks:
        pytest.fail("No document chunks were generated.")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks]
    )

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_documents(chunks, embeddings)

    return embedding_model, vector_store


def load_evaluation_cases() -> list[dict[str, str]]:
    return json.loads(
        EVALUATION_PATH.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "case",
    load_evaluation_cases(),
    ids=lambda case: case["expected_source"],
)
def test_expected_factory_document_is_retrieved(
    retrieval_system: tuple[EmbeddingModel, VectorStore],
    case: dict[str, str],
) -> None:
    embedding_model, vector_store = retrieval_system

    query_embedding = embedding_model.encode([case["query"]])

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    retrieved_sources = [
        result["source"] for result in results
    ]

    assert case["expected_source"] in retrieved_sources, (
        f"Expected {case['expected_source']} in the top 3 results "
        f"for query: {case['query']!r}. "
        f"Retrieved sources: {retrieved_sources}"
    )