"""
scripts/run_query.py

Interactive CLI to query the RAG system after ingestion.
Usage:
    python scripts/run_query.py --method paragraph --index flat
    python scripts/run_query.py --method hybrid --index hnsw --top-k 5
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "semantic", "hybrid"],
                        help="Which chunking method's index to load")
    parser.add_argument("--index", default="flat",
                        choices=["flat", "faiss", "hnsw"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", type=str, default=None,
                        help="Single question (non-interactive mode)")
    args = parser.parse_args()

    config = load_config()

    # Load chunks and build index
    chunks_path = f"data/processed/chunks_{args.method}.json"
    if not os.path.exists(chunks_path):
        print(f"✗ Chunks not found: {chunks_path}")
        print(f"  Run: python scripts/run_ingest.py --method {args.method}")
        sys.exit(1)

    from src.indexers import get_indexer
    from src.generator import Generator

    print(f"[query] Loading {args.index} index for method={args.method}...")
    with open(chunks_path) as f:
        chunks = json.load(f)

    indexer = get_indexer(
        index_type=args.index,
        embedding_model=config["embedding"]["model"],
        persist_dir=f"./chroma_db_{args.method}",
    )
    indexer.add(chunks)
    generator = Generator()

    print(f"[query] Ready. {len(chunks)} chunks indexed.")
    print(f"[query] Model: {config['llm']['model']} | top_k={args.top_k}")

    def ask(question: str):
        retrieved = indexer.search(question, top_k=args.top_k)
        contexts = [r["text"] for r in retrieved]
        answer = generator.generate(question, contexts)

        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print(f"{'─'*60}")
        print(f"A: {answer}")
        print(f"\n--- Retrieved Contexts ---")
        for i, r in enumerate(retrieved):
            print(f"[{i+1}] (score={r.get('score', 0):.3f}) "
                  f"{r.get('title', '')} p.{r.get('page_num', '?')}: "
                  f"{r['text'][:120]}...")
        print()

    if args.question:
        ask(args.question)
    else:
        print("\nType your question (or 'quit' to exit):\n")
        while True:
            try:
                q = input(">>> ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    ask(q)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()