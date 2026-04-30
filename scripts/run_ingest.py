"""
scripts/run_ingest.py

Ingests all PDFs in data/raw/pdfs/ using a chosen chunking method.
Saves chunks to data/processed/chunks_{method}.json
Logs stats to W&B.

Usage:
    python scripts/run_ingest.py --method fixed --chunk-size 512
    python scripts/run_ingest.py --method sentence --sentences-per-chunk 5
    python scripts/run_ingest.py --method paragraph --chunk-size 512
    python scripts/run_ingest.py --method semantic --threshold 0.5
    python scripts/run_ingest.py --method hybrid
    python scripts/run_ingest.py --method all      # run all methods
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import wandb

PDF_DIR = "data/raw/pdfs"
PROCESSED_DIR = "data/processed"


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_paper_metadata(pdf_dir: str) -> dict:
    """Map filename → paper title from metadata JSON."""
    meta_path = os.path.join(pdf_dir, "papers_metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        papers = json.load(f)
    # Build map: arxiv_id → title
    mapping = {}
    for p in papers:
        safe_id = p["arxiv_id"].replace("/", "_")
        mapping[safe_id] = p["title"]
    return mapping


def ingest_with_method(method: str, pdf_dir: str, config: dict, **kwargs) -> list:
    from src.chunkers import get_chunker

    chunker = get_chunker(method, **kwargs)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[ingest] No PDFs found in {pdf_dir}")
        return []

    metadata_map = load_paper_metadata(pdf_dir)
    all_chunks = []
    global_id = 0

    for fname in sorted(pdf_files):
        pdf_path = os.path.join(pdf_dir, fname)
        # Try to get a nice title from metadata
        arxiv_id = fname.split("_")[0]
        title = metadata_map.get(arxiv_id, fname.replace(".pdf", "").replace("_", " "))

        print(f"  Chunking [{method}]: {title[:50]}...")
        t0 = time.time()
        try:
            chunks = chunker.chunk(pdf_path=pdf_path, title=title)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

        # Re-assign global IDs
        for c in chunks:
            c["id"] = global_id
            global_id += 1

        elapsed = time.time() - t0
        print(f"    ✓ {len(chunks)} chunks in {elapsed:.1f}s")
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: list, method: str) -> str:
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"chunks_{method}.json")
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"  Saved {len(chunks)} chunks → {out_path}")
    return out_path


def log_to_wandb(config: dict, method: str, chunks: list, elapsed: float, extra_config: dict):
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="ingestion",
        name=f"ingest_{method}",
        config={
            "chunk_method": method,
            "pdf_dir": PDF_DIR,
            **extra_config,
        },
        reinit=True,
    )

    lengths = [c["chunk_size"] for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    wandb.log({
        "total_chunks": len(chunks),
        "avg_chunk_chars": avg_len,
        "max_chunk_chars": max_len,
        "min_chunk_chars": min_len,
        "ingestion_time_sec": elapsed,
        "chunks_per_sec": len(chunks) / elapsed if elapsed > 0 else 0,
    })

    # Log distribution as histogram
    table = wandb.Table(columns=["chunk_method", "chunk_size", "title"])
    for c in chunks[:500]:  # sample first 500
        table.add_data(c["chunk_method"], c["chunk_size"], c["title"])
    wandb.log({"chunk_size_distribution": table})

    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Method configs
# ─────────────────────────────────────────────────────────────────────────────

METHOD_CONFIGS = {
    "fixed":     {"chunk_size": 512, "chunk_overlap": 50},
    "sentence":  {"sentences_per_chunk": 5, "overlap_sentences": 1},
    "paragraph": {"chunk_size": 512, "chunk_overlap": 50},
    "semantic":  {"similarity_threshold": 0.5},
    "hybrid":    {},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "semantic", "hybrid", "all"],
                        help="Chunking method to use")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--sentences-per-chunk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold for semantic chunker")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    config = load_config()
    methods = list(METHOD_CONFIGS.keys()) if args.method == "all" else [args.method]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Ingesting with method: {method}")
        print(f"{'='*60}")

        # Build kwargs from defaults + CLI overrides
        kwargs = METHOD_CONFIGS[method].copy()
        if method == "fixed":
            kwargs["chunk_size"] = args.chunk_size
            kwargs["chunk_overlap"] = args.chunk_overlap
        elif method == "paragraph":
            kwargs["chunk_size"] = args.chunk_size
            kwargs["chunk_overlap"] = args.chunk_overlap
        elif method == "sentence":
            kwargs["sentences_per_chunk"] = args.sentences_per_chunk
        elif method == "semantic":
            kwargs["similarity_threshold"] = args.threshold
            kwargs["embedding_model"] = config["embedding"]["model"]

        t0 = time.time()
        chunks = ingest_with_method(method, PDF_DIR, config, **kwargs)
        elapsed = time.time() - t0

        if not chunks:
            print(f"[skip] No chunks produced for method={method}")
            continue

        out_path = save_chunks(chunks, method)

        if not args.no_wandb:
            log_to_wandb(config, method, chunks, elapsed, kwargs)
            print(f"  ✓ W&B logged: ingest_{method}")

        print(f"\n✅ [{method}] Done — {len(chunks)} chunks in {elapsed:.1f}s → {out_path}")

    print("\n✅ All ingestion complete. Next: python scripts/generate_eval.py")