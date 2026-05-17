"""
scripts/run_eval_local.py

Alternative to sol/eval_job.slurm — runs on any PC (CPU only).
Runs self-retrieval evaluation across all ingested methods and logs to W&B.

Usage:
    python scripts/run_eval_local.py
    python scripts/run_eval_local.py --top-k 5
    python scripts/run_eval_local.py --methods fixed paragraph --no-wandb

What it does:
    1. Loads each chunks_{method}.json from data/processed/
    2. Builds HNSW and flat indexes for each method
    3. Runs self-retrieval: queries each chunk with its own text
    4. Measures hit rate and latency
    5. Logs comparison table to W&B
    6. Saves results to data/eval/self_retrieval_results.json

No GPU needed. No API key needed. Runs in ~2-5 minutes on CPU.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Run self-retrieval evaluation on your PC (no GPU, no API key)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to evaluate (default: all available in data/processed/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of chunks to use as eval queries (default: 100)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B logging",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Retrieval Evaluation (CPU mode, no API needed)")
    print("=" * 60)

    import subprocess

    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--all",
        "--top-k", str(args.top_k),
        "--n-samples", str(args.n_samples),
    ]

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode == 0:
        print("\nDone. Results saved to data/eval/self_retrieval_results.json")
        print("View W&B dashboard: https://wandb.ai/YOUR_ENTITY/rag-document-assistant")
    else:
        print("\nEvaluation failed. Check logs above.")


if __name__ == "__main__":
    main()
