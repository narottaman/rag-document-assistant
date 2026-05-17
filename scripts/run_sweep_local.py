"""
scripts/run_sweep_local.py

Alternative to sol/sweep_job.slurm — runs on any PC (CPU only).
Runs W&B sweep to find the best chunking + indexing combination.

Usage:
    python scripts/run_sweep_local.py
    python scripts/run_sweep_local.py --count 10
    python scripts/run_sweep_local.py --sweep-id YOUR_EXISTING_SWEEP_ID

What it does:
    1. Creates a new W&B sweep from configs/sweep.yaml
    2. Runs the sweep agent on your PC
    3. Each run tests one combination of chunking x indexing x top_k
    4. Logs context_hit_rate and latency to W&B for every run
    5. W&B dashboard shows which combination is best

Requirements:
    - WANDB_API_KEY set in .env or environment
    - At least one chunks_{method}.json in data/processed/
    - data/eval/hf_qa_subset.json (run download_data.py first)

Note: --count limits how many runs to execute.
Without --count, runs all combinations (36+ on CPU takes ~2-3 hours).
Start with --count 6 to test it works, then run overnight without limit.
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Run W&B sweep on your PC to find best RAG configuration"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Max number of sweep runs (default: all combinations ~36)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to continue (default: creates new sweep)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="ngangada-arizona-state-university",
        help="W&B entity (username or team)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="rag-document-assistant",
        help="W&B project name",
    )
    args = parser.parse_args()

    # Load .env if present
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set.")
        print("Add it to .env file or run: set WANDB_API_KEY=your-key (Windows)")
        sys.exit(1)

    print("=" * 60)
    print("W&B Sweep (CPU mode)")
    print("=" * 60)

    sweep_id = args.sweep_id

    # Step 1: Create sweep if not provided
    if not sweep_id:
        print("\nStep 1: Creating W&B sweep from configs/sweep.yaml...")
        result = subprocess.run(
            ["wandb", "sweep",
             "--entity", args.entity,
             "--project", args.project,
             "configs/sweep.yaml"],
            capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)

        # Parse sweep ID from output
        for line in (result.stdout + result.stderr).split("\n"):
            if "wandb agent" in line:
                sweep_id = line.strip().split()[-1]
                break

        if not sweep_id:
            print("Could not parse sweep ID. Run manually:")
            print("  wandb sweep configs/sweep.yaml")
            print("  wandb agent YOUR_ENTITY/rag-document-assistant/SWEEP_ID")
            sys.exit(1)

        print(f"\nSweep created: {sweep_id}")

    # Step 2: Run the agent
    print(f"\nStep 2: Running sweep agent...")
    print(f"View at: https://wandb.ai/{args.entity}/{args.project}/sweeps/")
    print("Press Ctrl+C to stop early.\n")

    agent_cmd = ["wandb", "agent", sweep_id]
    if args.count:
        agent_cmd += ["--count", str(args.count)]

    subprocess.run(agent_cmd)


if __name__ == "__main__":
    main()
