import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.embeddings import EmbeddingModel
from backend.app.ai.retrieval import VectorStore
from evaluations.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)

INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "factory"
CASES_PATH = PROJECT_ROOT / "evaluations" / "retrieval_cases.json"
REPORT_PATH = (
    PROJECT_ROOT
    / "evaluations"
    / "results"
    / "retrieval_metrics.json"
)


def load_cases() -> list[dict[str, str]]:
    """Load and validate the retrieval evaluation dataset."""
    if not CASES_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {CASES_PATH}"
        )

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a nonempty JSON list")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each evaluation case must be a JSON object")

        query = case.get("query")
        expected_source = case.get("expected_source")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Each evaluation case needs a nonempty query"
            )

        if (
            not isinstance(expected_source, str)
            or not expected_source.strip()
        ):
            raise ValueError(
                "Each evaluation case needs an expected_source"
            )

    return cases


def main() -> None:
    cases = load_cases()

    print("Loading saved factory index...")
    vector_store = VectorStore.load(INDEX_PATH)

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    results: list[dict[str, object]] = []
    ranked_sources: list[list[str]] = []
    expected_sources: list[str] = []

    for case in cases:
        query = case["query"]
        expected_source = case["expected_source"]

        query_embedding = embedding_model.encode([query])

        retrieved = vector_store.search(
            query_embedding=query_embedding,
            top_k=3,
        )

        sources = [str(item["source"]) for item in retrieved]

        ranked_sources.append(sources)
        expected_sources.append(expected_source)

        result = {
            "query": query,
            "expected_source": expected_source,
            "retrieved_sources": sources,
            "recall_at_1": recall_at_k(
                sources,
                expected_source,
                k=1,
            ),
            "recall_at_3": recall_at_k(
                sources,
                expected_source,
                k=3,
            ),
            "reciprocal_rank": reciprocal_rank(
                sources,
                expected_source,
            ),
        }

        results.append(result)

        print(
            f"\nQuery: {query}\n"
            f"Expected: {expected_source}\n"
            f"Retrieved: {sources}\n"
            f"Reciprocal rank: {result['reciprocal_rank']:.3f}"
        )

    metrics = {
        "evaluation_timestamp": datetime.now(UTC).isoformat(),
        "evaluation_cases": len(cases),
        "recall_at_1": mean_recall_at_k(
            ranked_sources,
            expected_sources,
            k=1,
        ),
        "recall_at_3": mean_recall_at_k(
            ranked_sources,
            expected_sources,
            k=3,
        ),
        "mean_reciprocal_rank": mean_reciprocal_rank(
            ranked_sources,
            expected_sources,
        ),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Evaluation cases: {metrics['evaluation_cases']}")
    print(f"Recall@1: {metrics['recall_at_1']:.3f}")
    print(f"Recall@3: {metrics['recall_at_3']:.3f}")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.3f}")
    print(f"\nReport saved to: {REPORT_PATH}")
