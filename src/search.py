import json
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embed import DocumentEmbedder


class SemanticSearcher:
    """Loads index and performs cosine-similarity search."""

    def __init__(self, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir

        emb_path = os.path.join(index_dir, "embeddings.npy")
        meta_path = os.path.join(index_dir, "metadata.json")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.embeddings: np.ndarray = np.load(emb_path)
        # Ensure 2D
        if self.embeddings.ndim == 1 and self.embeddings.size > 0:
            self.embeddings = self.embeddings.reshape(1, -1)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        # Initialize embedder (loads model once)
        self.embedder = DocumentEmbedder(model_name=model_name)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a search against the index.

        Args:
            query: The user's search string.
            k: Number of top results to return.

        Returns:
            List of dicts with keys 'id', 'score', 'snippet' sorted descending by score.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")

        if self.embeddings.shape[0] == 0 or len(self.metadata) == 0:
            return []

        query = query.strip()
        if not query:
            return []

        # Embed query -> shape (1, D)
        query_vec = self.embedder.generate_embeddings([query])
        # query_vec is (1, D), embeddings is (N, D)
        scores = cosine_similarity(query_vec, self.embeddings)
        # scores shape (1, N)
        scores = scores.flatten()  # (N,)

        # Top-K descending
        k = min(k, len(scores))
        # argsort ascending, take last k reversed
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            meta = self.metadata[int(idx)]
            results.append({
                "id": meta["id"],
                "score": float(scores[int(idx)]),
                "snippet": meta.get("snippet", ""),
            })
        return results
