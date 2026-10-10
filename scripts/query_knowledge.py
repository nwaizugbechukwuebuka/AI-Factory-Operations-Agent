from pathlib import Path

from backend.app.ai.embeddings import EmbeddingModel
from backend.app.ai.retrieval import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "factory"


def main() -> None:
    query = input("Ask a factory operations question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    embedding_model = EmbeddingModel()
    vector_store = VectorStore.load(INDEX_PATH)

    query_embedding = embedding_model.encode([query])

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    if not results:
        print("No matching documents were found.")
        return

    print("\nRetrieved factory knowledge:\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Source: {result['source']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print("-" * 60)


if __name__ == "__main__":
    main()