import json
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.ingest import ingest_documents
from src.embed import DocumentEmbedder


def build_index(data_dir: str, output_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Orchestrates ingestion, embedding generation, and persistence.

    Args:
        data_dir: Where raw .txt documents live.
        output_dir: Where to save embeddings.npy and metadata.json
    """
    t0 = time.perf_counter()
    print(f"[Indexing] Ingesting documents from '{data_dir}'...")
    docs = ingest_documents(data_dir)
    t_ingest = time.perf_counter()

    if not docs:
        print(f"[Indexing] No .txt documents found in '{data_dir}'. Creating empty index.")
        # Still create output dir with empty index
        os.makedirs(output_dir, exist_ok=True)
        # Need embedding dimension
        embedder = DocumentEmbedder(model_name=model_name)
        dim = embedder.model.get_sentence_embedding_dimension()
        empty = np.empty((0, dim), dtype=np.float32)
        np.save(os.path.join(output_dir, "embeddings.npy"), empty)
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print(f"[Indexing] Saved empty index to '{output_dir}'.")
        return

    print(f"[Indexing] Found {len(docs)} document(s). Generating embeddings with '{model_name}'...")

    texts = [d["content"] for d in docs]
    ids = [d["id"] for d in docs]

    embedder = DocumentEmbedder(model_name=model_name)
    t_embed_start = time.perf_counter()
    embeddings = embedder.generate_embeddings(texts)
    t_embed_end = time.perf_counter()

    print(f"[Indexing] Embeddings shape: {embeddings.shape}")

    # Create metadata list (aligned with embeddings rows)
    metadata: List[Dict] = []
    for doc_id, content in zip(ids, texts):
        snippet = content[:200].replace("\n", " ").strip()
        if len(content) > 200:
            snippet += "..."
        metadata.append({"id": doc_id, "snippet": snippet})

    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings.npy
    emb_path = os.path.join(output_dir, "embeddings.npy")
    meta_path = os.path.join(output_dir, "metadata.json")

    t_save_start = time.perf_counter()
    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    t_save_end = time.perf_counter()

    total = time.perf_counter() - t0
    print(f"[Indexing] Saved embeddings to '{emb_path}'")
    print(f"[Indexing] Saved metadata to '{meta_path}'")
    print(f"[Benchmark] Ingestion: {t_ingest - t0:.3f}s | Embedding: {t_embed_end - t_embed_start:.3f}s | "
          f"Saving: {t_save_end - t_save_start:.3f}s | Total: {total:.3f}s")
    print(f"[Indexing] Done. Indexed {len(docs)} documents.")
