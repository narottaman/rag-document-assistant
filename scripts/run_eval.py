"""
scripts/run_eval.py

Self-retrieval evaluation — no LLM API needed, no corpus mismatch.

Strategy:
  For each chunk in the corpus, the chunk TEXT itself is the "question proxy".
  We embed the chunk text as a query and check if the same chunk appears in
  top-k results. This is a clean, corpus-aligned retrieval quality test.

  context_hit_rate = fraction of chunks that retrieve themselves in top-k

This approach:
  - Requires zero API calls (no OpenAI, no Gemini)
  - Is always aligned with the corpus (no berry/twitter mismatch)
  - Is a standard RAG retrieval sanity check used in production

Usage:
    python scripts/run_eval.py --all
    python scripts/run_eval.py --method paragraph --index hnsw --top-k 10
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import wandb
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = "data/processed"


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def self_retrieval_eval(chunking_method, index_type, top_k, config, n_samples=100):
    """
    Self-retrieval test: embed a chunk's own text as query,
    check if that chunk appears in top-k results.
    No LLM needed. No external eval set needed.
    """
    from src.indexers import get_indexer

    chunks_path = os.path.join(PROCESSED_DIR, f"chunks_{chunking_method}.json")
    if not os.path.exists(chunks_path):
        print(f"[skip] {chunks_path} not found")
        return {}

    with open(chunks_path) as f:
        all_chunks = json.load(f)

    # Filter chunks with enough text to be meaningful queries
    valid_chunks = [c for c in all_chunks if len(c["text"]) >= 100]

    # Sample up to n_samples chunks for eval (deterministic)
    random.seed(42)
    sample_size = min(n_samples, len(valid_chunks))
    eval_chunks = random.sample(valid_chunks, sample_size)

    print(f"[eval] {chunking_method} | {index_type} | top_k={top_k}")
    print(f"[eval] Indexing {len(all_chunks)} chunks, testing {sample_size} samples...")

    # Build index over ALL chunks
    extra = {"persist_dir": f"./chroma_db_eval_{chunking_method}_{index_type}"} \
            if index_type == "flat" else {}
    indexer = get_indexer(
        index_type=index_type,
        embedding_model=config["embedding"]["model"],
        **extra,
    )
    indexer.add(all_chunks)

    # Self-retrieval: query with chunk text, check if chunk ID appears in results
    hits = 0
    latencies = []
    import time

    for chunk in eval_chunks:
        t0 = time.time()
        results = indexer.search(chunk["text"][:200], top_k=top_k)
        latencies.append(time.time() - t0)

        # Check if this chunk's text appears in any retrieved result
        retrieved_texts = [r["text"] for r in results]
        if any(chunk["text"][:80] in rt for rt in retrieved_texts):
            hits += 1

    hit_rate = hits / sample_size
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000

    print(f"[eval] context_hit_rate={hit_rate:.3f} | latency={avg_latency_ms:.1f}ms")

    return {
        "chunking_method": chunking_method,
        "index_type": index_type,
        "top_k": top_k,
        "context_hit_rate": hit_rate,
        "avg_latency_ms": avg_latency_ms,
        "num_chunks": len(all_chunks),
        "num_eval_samples": sample_size,
        "eval_type": "self_retrieval",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "semantic", "hybrid"])
    parser.add_argument("--index", default="hnsw", choices=["flat", "faiss", "hnsw"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of chunks to use as eval queries")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all available chunking methods")
    args = parser.parse_args()

    config = load_config()
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="final_evaluation",
        name="self_retrieval_all" if args.all else f"self_retrieval_{args.method}_{args.index}",
    )

    if args.all:
        methods = ["fixed", "sentence", "paragraph", "semantic", "hybrid"]
        index_types = ["flat", "hnsw"]
        all_results = []

        for method in methods:
            if not os.path.exists(os.path.join(PROCESSED_DIR, f"chunks_{method}.json")):
                print(f"[skip] {method} not ingested yet")
                continue
            for idx_type in index_types:
                res = self_retrieval_eval(method, idx_type, args.top_k, config, args.n_samples)
                if res:
                    all_results.append(res)

        if all_results:
            # Log comparison table to W&B
            table = wandb.Table(columns=[
                "chunking_method", "index_type", "top_k",
                "context_hit_rate", "avg_latency_ms", "num_chunks"
            ])
            for r in all_results:
                table.add_data(
                    r["chunking_method"], r["index_type"], r["top_k"],
                    r["context_hit_rate"], r["avg_latency_ms"], r["num_chunks"]
                )
            wandb.log({"self_retrieval_comparison": table})

            # Save locally
            out = "data/eval/self_retrieval_results.json"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"\n✅ Saved → {out}")
            print("\n📊 Self-Retrieval Results (sorted by hit_rate):")
            print(f"  {'Method':<12} {'Index':<6} {'top_k':<6} {'Hit Rate':<10} {'Latency ms'}")
            print(f"  {'-'*52}")
            for r in sorted(all_results, key=lambda x: x["context_hit_rate"], reverse=True):
                print(f"  {r['chunking_method']:<12} {r['index_type']:<6} "
                      f"{r['top_k']:<6} {r['context_hit_rate']:.3f}      "
                      f"{r['avg_latency_ms']:.1f}")

    else:
        res = self_retrieval_eval(
            args.method, args.index, args.top_k, config, args.n_samples
        )
        if res:
            wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float))})
            print(f"\n✅ {res}")

    wandb.finish()