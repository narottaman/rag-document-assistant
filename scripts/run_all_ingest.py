"""
scripts/run_all_ingest.py

Alternative to sol/embed_job.slurm — runs on any PC (CPU only).
Runs all chunking methods in sequence and logs stats to W&B.

Usage:
    python scripts/run_all_ingest.py
    python scripts/run_all_ingest.py --methods fixed paragraph sentence
    python scripts/run_all_ingest.py --pdf-dir data/raw/pdfs --no-wandb

What it does:
    1. Reads all PDFs from data/raw/pdfs/
    2. Chunks them with each method (fixed, sentence, paragraph)
    3. Saves chunks to data/processed/chunks_{method}.json
    4. Logs chunk stats (count, avg size, speed) to W&B

GPU not required — runs on CPU.
Semantic and hybrid methods are skipped by default (need GPU for speed).
Add --all to include them (slow on CPU).
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Run all ingestion methods on your PC (no GPU needed)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fixed", "paragraph", "sentence"],
        choices=["fixed", "paragraph", "sentence", "semantic", "hybrid"],
        help="Chunking methods to run (default: fixed paragraph sentence)",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw/pdfs",
        help="Folder containing PDF files (default: data/raw/pdfs)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 methods including semantic and hybrid (slow on CPU)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B logging",
    )
    args = parser.parse_args()

    if args.all:
        args.methods = ["fixed", "paragraph", "sentence", "semantic", "hybrid"]

    print("=" * 60)
    print("RAG Ingestion Pipeline (CPU mode)")
    print("=" * 60)
    print(f"PDF folder : {args.pdf_dir}")
    print(f"Methods    : {args.methods}")
    print(f"W&B logging: {'disabled' if args.no_wandb else 'enabled'}")
    print()

    # Build the command for each method
    import subprocess

    total_start = time.time()
    results = []

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Running: {method}")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            "scripts/run_ingest.py",
            "--method", method,
        ]
        if args.no_wandb:
            cmd.append("--no-wandb")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=os.getcwd())
        elapsed = time.time() - t0

        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        results.append((method, status, elapsed))
        print(f"\n[{method}] {status} in {elapsed:.1f}s")

    # Summary
    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All ingestion complete in {total:.1f}s")
    print(f"{'='*60}")
    for method, status, elapsed in results:
        icon = "OK" if status == "SUCCESS" else "FAIL"
        print(f"  [{icon}] {method:<12} {elapsed:.1f}s")
    print()
    print("Next step: python scripts/run_eval_local.py")


if __name__ == "__main__":
    main()
