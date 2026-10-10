import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate normalized semantic embeddings for text."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Convert text into normalized embedding vectors."""
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()),
                            dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)