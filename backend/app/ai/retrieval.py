import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Store text embeddings and retrieve the most similar chunks."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[dict[str, str]] = []

    def add_documents(
        self,
        chunks: list[dict[str, str]],
        embeddings: np.ndarray,
    ) -> None:
        """Add document chunks and their corresponding embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")

        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Expected vectors of dimension {self.index.d}, "
                f"received {vectors.shape[1]}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numerical values")

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms == 0):
            raise ValueError("Embeddings must not contain zero vectors")

        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, str | float]]:
        """Retrieve the top matching document chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("Provide exactly one query embedding")

        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Expected query dimension {self.index.d}, "
                f"received {query.shape[1]}"
            )

        if not np.isfinite(query).all():
            raise ValueError("Query embedding contains invalid values")

        if np.linalg.norm(query) == 0:
            raise ValueError("Query embedding must not be a zero vector")

        faiss.normalize_L2(query)

        result_count = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, result_count)

        results: list[dict[str, str | float]] = []

        for score, index_position in zip(
            scores[0],
            indices[0],
            strict=True,
        ):
            if index_position < 0:
                continue

            chunk = self.chunks[int(index_position)]

            results.append(
                {
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )

        return results

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and matching chunk metadata."""
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        index_path = output_directory / "factory.index"
        metadata_path = output_directory / "chunks.json"

        faiss.write_index(self.index, str(index_path))

        temporary_path = metadata_path.with_suffix(".json.tmp")

        temporary_path.write_text(
            json.dumps(self.chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        temporary_path.replace(metadata_path)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a persisted index and its chunk metadata."""
        input_directory = Path(directory)

        index_path = input_directory / "factory.index"
        metadata_path = input_directory / "chunks.json"

        if not index_path.is_file():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = faiss.read_index(str(index_path))

        store = cls(dimension=index.d)
        store.index = index
        store.chunks = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if store.index.ntotal != len(store.chunks):
            raise ValueError(
                "Index vector count does not match chunk metadata count"
            )

        return store