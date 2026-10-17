import argparse
import json
import sys
import time
from pathlib import Path

from src.indexer import build_index
from src.search import SemanticSearcher


def main():
    parser = argparse.ArgumentParser(description="Semantic Document Search Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index subcommand
    index_parser = subparsers.add_parser("index", help="Build the search index")
    index_parser.add_argument("--data-dir", required=True, help="Directory containing .txt documents")
    index_parser.add_argument("--output-dir", required=True, help="Directory to save embeddings.npy and metadata.json")
    index_parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Query the search index")
    search_parser.add_argument("--index-dir", required=True, help="Directory containing embeddings.npy and metadata.json")
    search_parser.add_argument("--query", type=str, default=None, help="Single search query string")
    search_parser.add_argument("--batch-file", type=str, default=None, help="Path to text file with one query per line")
    search_parser.add_argument("--k", type=int, default=5, help="Number of top results to return (default: 5)")
    search_parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")

    args = parser.parse_args()

    if args.command == "index":
        build_index(args.data_dir, args.output_dir, model_name=args.model)

    elif args.command == "search":
        # Validate k
        if args.k is not None and args.k <= 0:
            print("Error: --k must be a positive integer.", file=sys.stderr)
            sys.exit(1)

        # Mutually exclusive check: need at least one of --query or --batch-file
        if not args.query and not args.batch_file:
            print("Error: provide either --query or --batch-file.", file=sys.stderr)
            sys.exit(1)

        # Load index once
        print(f"[Search] Loading index from '{args.index_dir}'...")
        t_load_start = time.perf_counter()
        searcher = SemanticSearcher(args.index_dir, model_name=args.model)
        t_load_end = time.perf_counter()
        print(f"[Search] Index loaded: {searcher.embeddings.shape[0]} documents, dim={searcher.embeddings.shape[1] if searcher.embeddings.size else 0} "
              f"(load time: {t_load_end - t_load_start:.3f}s)")

        # Collect queries
        queries = []
        if args.query:
            queries.append(args.query)
        if args.batch_file:
            bf = Path(args.batch_file)
            if not bf.exists():
                print(f"Error: batch file not found: {args.batch_file}", file=sys.stderr)
                sys.exit(1)
            with open(bf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    q = line.strip()
                    if q:
                        queries.append(q)
            if not queries:
                print("Warning: batch file contains no valid queries.")

        # If both --query and --batch-file supplied, we run all queries (query + batch)
        # If only batch, run batch. If only query, run single.

        all_results = {}
        for q in queries:
            t_q_start = time.perf_counter()
            results = searcher.search(q, k=args.k)
            t_q_end = time.perf_counter()
            latency = t_q_end - t_q_start

            # Output to stdout - structured JSON + readable table
            output_block = {
                "query": q,
                "k": args.k,
                "latency_seconds": round(latency, 4),
                "results": [
                    {"id": r["id"], "score": round(r["score"], 4), "snippet": r["snippet"]}
                    for r in results
                ]
            }

            # Print JSON for verifiability
            print(json.dumps(output_block, indent=2, ensure_ascii=False))

            # Also print readable table
            print(f"\nTop {len(results)} results for query: \"{q}\" (latency: {latency:.4f}s)")
            print("-" * 80)
            for rank, r in enumerate(results, 1):
                print(f"  {rank}. [{r['id']}]  score={r['score']:.4f}")
                print(f"     snippet: {r['snippet'][:150]}")
            print("=" * 80 + "\n")

            all_results[q] = output_block

        # If batch mode, average latency
        if len(queries) > 1:
            avg_lat = sum(v["latency_seconds"] for v in all_results.values()) / len(all_results)
            print(f"[Benchmark] Average query latency: {avg_lat:.4f}s over {len(queries)} queries")


if __name__ == "__main__":
    main()
