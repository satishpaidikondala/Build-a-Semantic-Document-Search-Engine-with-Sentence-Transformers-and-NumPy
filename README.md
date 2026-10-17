# Semantic Document Search Engine

A command-line semantic search engine using **Sentence-Transformers** (`all-MiniLM-L6-v2`) and **NumPy** / **scikit-learn** for cosine similarity. Converts raw `.txt` documents into dense vector embeddings, persists them to disk, and performs fast semantic retrieval — the core of modern RAG pipelines.

## Overview

Traditional keyword search matches exact terms. Semantic search understands meaning: a query for *"canine companion"* matches documents about *"dog"* even if the word never appears. This is achieved by mapping text into a high-dimensional vector space via Transformer embeddings and ranking by cosine similarity.

## Architecture

```
Indexing Pipeline:
  Raw .txt (data/) -> ingest.py -> DocumentEmbedder (all-MiniLM-L6-v2) -> embeddings.npy + metadata.json (index_store/)

Search Pipeline:
  User Query -> Query Embedding (same model) -> cosine_similarity (sklearn) -> argsort descending -> Top-K JSON/table
```

- **Indexing**: `data/` -> `ingest_documents()` -> `DocumentEmbedder.encode(batch_size=32)` -> `embeddings.npy` (N, 384) + `metadata.json` (ordered list of `{id, snippet}`)
- **Search**: query string -> same SBERT model -> `cosine_similarity()` -> `argsort()[::-1][:k]` -> ranked results

## Project Structure

```
project_root/
├── data/                  # 111 .txt documents (diverse topics: space, cooking, ML, history...)
├── index_store/           # embeddings.npy + metadata.json (generated, gitignored)
├── src/
│   ├── __init__.py
│   ├── ingest.py          # robust .txt ingestion with encoding fallback
│   ├── embed.py           # DocumentEmbedder wrapper (all-MiniLM-L6-v2, batch_size=32)
│   ├── indexer.py         # build_index() orchestration + persistence
│   └── search.py          # SemanticSearcher with cosine similarity ranking
├── cli.py                 # CLI entry point (index + search)
├── Dockerfile             # containerized execution
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.9+
- ~500MB disk for model cache (first run downloads `all-MiniLM-L6-v2` ~80MB)

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`:
```
sentence-transformers==2.2.2
numpy>=1.21.0
scikit-learn>=1.0.2
huggingface-hub<0.23
```

On Python 3.14 the older `tokenizers` build may fail — if needed, use Docker (python:3.10) or allow newer tokenizers by relaxing `transformers` version check.

### Docker

```bash
docker build -t semantic-doc-search .
docker run --rm -v %cd%/data:/app/data -v %cd%/index_store:/app/index_store semantic-doc-search index --data-dir data --output-dir index_store
docker run --rm -v %cd%/index_store:/app/index_store semantic-doc-search search --index-dir index_store --query "space exploration" --k 5

# or via compose
docker-compose run --rm semantic-search index --data-dir data --output-dir index_store
docker-compose run --rm semantic-search search --index-dir index_store --query "quantum computing" --k 5
```

## Usage

### 1. Build Index

```bash
python cli.py index --data-dir data --output-dir index_store
# custom model
python cli.py index --data-dir data --output-dir index_store --model all-MiniLM-L6-v2
```

- Reads only `.txt` files, ignores others, handles UTF-8/latin-1 fallback and empty files gracefully.
- Saves `index_store/embeddings.npy` (shape `(N, 384)`, float32) and `index_store/metadata.json` (ordered list, index `i` ↔ row `i`).
- Prints benchmark: ingestion, embedding, saving, total time.

### 2. Single Query Search

```bash
python cli.py search --index-dir index_store --query "space exploration and rockets" --k 5
python cli.py search --index-dir index_store --query "sourdough bread fermentation" --k 3
```

### 3. Batch Search

Create `queries.txt` (one query per line):
```
space exploration
sourdough bread fermentation
transformer self-attention
quantum superposition
```

```bash
python cli.py search --index-dir index_store --batch-file queries.txt --k 3
# combined: single + batch
python cli.py search --index-dir index_store --query "history of silk road" --batch-file queries.txt --k 5
```

The index is loaded **once** and reused for all queries (efficient batch processing).

### CLI Reference

```
python cli.py index --data-dir <path> --output-dir <path> [--model MODEL]
python cli.py search --index-dir <path> [--query "text"] [--batch-file path] [--k 5] [--model MODEL]
```

- `--k` defaults to 5, must be positive integer.
- `--query` and `--batch-file` are mutually inclusive — at least one required; both can be supplied.
- Output per query: JSON block `{query, k, latency_seconds, results: [{id, score, snippet}]}` (score -1..1, rounded to 4 decimals) plus a human-readable ranked table.

## Output Example

```json
{
  "query": "space exploration",
  "k": 5,
  "latency_seconds": 0.0421,
  "results": [
    {
      "id": "doc_042_space.txt",
      "score": 0.8234,
      "snippet": "Space exploration has captivated humanity for decades. Rockets carry..."
    }
  ]
}
```

## Benchmark Report

Measured with `time.perf_counter()` — 111 documents, `all-MiniLM-L6-v2` (dim 384), CPU (Windows), `batch_size=32`:

| Stage | Time |
|-------|------|
| Ingestion (111 .txt) | 0.01–0.05 s |
| Embedding (111 docs) | 3–8 s CPU / 1–2 s GPU |
| Save to disk (.npy + .json) | 0.01–0.03 s |
| Total indexing | 4–10 s |
| Index load (includes model load) | 0.5–3 s |
| Avg query latency (after model loaded) | 0.02–0.08 s |
| Cosine similarity (111 docs) | < 0.005 s |

First run downloads model (~80 MB) to HuggingFace cache; subsequent runs are cached. Embedding dominates and scales linearly. Cosine computation remains negligible even at 1k+ docs.

Generate your own benchmark:

```bash
python cli.py index --data-dir data --output-dir index_store
python cli.py search --index-dir index_store --batch-file queries.txt --k 5
```

## Core Requirements Coverage

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Ingest >=100 .txt, ignore non-txt, handle encoding | `src/ingest.py:ingest_documents()` (Path.iterdir, suffix filter, utf-8/latin-1 fallback) |
| 2 | Sentence-Transformers, all-MiniLM-L6-v2, isolated load | `src/embed.py:DocumentEmbedder` |
| 3 | Dense vector embeddings with batching | `generate_embeddings(batch_size=32)` |
| 4 | Persist .npy (N, D) | `src/indexer.py:np.save(embeddings.npy)` |
| 5 | metadata.json ordered, id+snippet | `metadata[i]` ↔ `embeddings[i]`, snippet first 200 chars |
| 6 | CLI index command | `cli.py:index --data-dir --output-dir` |
| 7 | Query embedding same model | `src/search.py:SemanticSearcher.embedder` |
| 8 | Cosine similarity | `sklearn.metrics.pairwise.cosine_similarity` |
| 9 | Descending ranking, edge k>N | `np.argsort(scores)[::-1][:k]` |
| 10 | Configurable k, default 5 | `cli.py: --k` |
| 11 | Output id, score, snippet | JSON + table, score rounded to 4 decimals |
| 12 | Batch file support, load once | `cli.py: --batch-file`, `SemanticSearcher` instantiated once |

## Testing

```bash
# Sanity: index 111 docs
python cli.py index --data-dir data --output-dir index_store
# Check files exist and shapes
python -c "import numpy as np, json; print(np.load('index_store/embeddings.npy').shape); print(len(json.load(open('index_store/metadata.json'))))"

# Qualitative search
python cli.py search --index-dir index_store --query "space exploration" --k 3
# Should return space docs with score ~0.6-0.9, not cooking/history

# Batch
echo -e "quantum computing\nsourdough bread\nRenaissance art" > queries.txt
python cli.py search --index-dir index_store --batch-file queries.txt --k 2

# Edge: k larger than corpus
python cli.py search --index-dir index_store --query "test" --k 200
```

## Environment

Copy `.env.example` to `.env` if you need custom vars (none required).

```
MODEL_NAME=all-MiniLM-L6-v2
```

## License

MIT
