from pathlib import Path

from backend.app.ai.embeddings import EmbeddingModel
from backend.app.ai.retrieval import VectorStore
from backend.app.data.chunking import chunk_text
from backend.app.data.ingestion import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = PROJECT_ROOT / "data" / "documents"
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "factory"


def main() -> None:
    print("Loading factory documents...")

    documents = load_documents(DOCUMENT_PATH)

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {DOCUMENT_PATH}"
        )

    print(f"Loaded {len(documents)} documents.")

    document_chunks: list[dict[str, str]] = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for chunk_number, chunk in enumerate(chunks, start=1):
            document_chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": (
                        f"{Path(document['source']).stem}-{chunk_number:03d}"
                    ),
                    "text": chunk,
                }
            )

    if not document_chunks:
        raise RuntimeError("Document processing produced no chunks.")

    print(f"Created {len(document_chunks)} text chunks.")
    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in document_chunks]
    )

    print(f"Generated embeddings with shape {embeddings.shape}.")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_documents(
        chunks=document_chunks,
        embeddings=embeddings,
    )

    vector_store.save(INDEX_PATH)

    print(f"Saved vector index to: {INDEX_PATH}")
    print("Knowledge index built successfully.")


if __name__ == "__main__":
    main()