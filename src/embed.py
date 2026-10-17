from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """Wrapper around sentence-transformers for generating embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of text strings into a 2D NumPy array.

        Args:
            texts: The document contents to encode.
            batch_size: Batch size for encoding to prevent OOM.

        Returns:
            np.ndarray of shape (num_documents, embedding_dimension).
        """
        if not texts:
            # Return empty array with correct dim if model is loaded
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        # Ensure 2D array
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings
