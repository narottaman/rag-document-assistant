"""
scripts/run_eval.py

Final evaluation — runs all chunking methods, logs comparison table to W&B.
Uses local Qwen2.5-3B-Instruct generator — no API key needed.

Usage:
    python scripts/run_eval.py --all
    python scripts/run_eval.py --method paragraph --index flat --top-k 5
"""

import os
import sys
import json
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


def evaluate_config(chunking_method, index_type, top_k, config, qa_path):
    from src.indexers import get_indexer
    from src.generator import Generator

    chunks_path = os.path.join(PROCESSED_DIR, f"chunks_{chunking_method}.json")
    if not os.path.exists(chunks_path):
        print(f"[skip] {chunks_path} not found")
        return {}
    if not os.path.exists(qa_path):
        print(f"[skip] {qa_path} not found")
        return {}

    with open(chunks_path) as f:
        chunks = json.load(f)
    with open(qa_path) as f:
        qa_pairs = json.load(f)

    extra = {"persist_dir": f"./chroma_db_{chunking_method}"} if index_type == "flat" else {}
    indexer = get_indexer(
        index_type=index_type,
        embedding_model=config["embedding"]["model"],
        **extra,
    )
    indexer.add(chunks)
    generator = Generator()

    results = []
    for qa in qa_pairs:
        retrieved = indexer.search(qa["question"], top_k=top_k)
        contexts = [r["text"] for r in retrieved]
        answer = generator.generate(qa["question"], contexts)
        results.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth_context": qa.get("ground_truth_context", ""),
        })

    # Context hit rate
    hits = sum(
        1 for qa, r in zip(qa_pairs, results)
        if qa.get("ground_truth_context", "") and
        any(qa["ground_truth_context"][:80] in ctx for ctx in r["contexts"])
    )
    context_hit_rate = hits / len(qa_pairs) if qa_pairs else 0.0

    return {
        "chunking_method": chunking_method,
        "index_type": index_type,
        "top_k": top_k,
        "context_hit_rate": context_hit_rate,
        "num_chunks": len(chunks),
        "num_qa_pairs": len(qa_pairs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "semantic", "hybrid"])
    parser.add_argument("--index", default="flat", choices=["flat", "faiss", "hnsw"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qa-path", default="data/eval/qa_self_retrieval.json")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = load_config()
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="final_evaluation",
        name="eval_comparison" if args.all else f"eval_{args.method}_{args.index}",
    )

    if args.all:
        methods = ["fixed", "sentence", "paragraph", "semantic", "hybrid"]
        all_results = []
        for method in methods:
            if not os.path.exists(os.path.join(PROCESSED_DIR, f"chunks_{method}.json")):
                print(f"[skip] {method} not ingested yet")
                continue
            print(f"\n[eval] {method} | flat | top_k={args.top_k}")
            res = evaluate_config(method, "flat", args.top_k, config, args.qa_path)
            if res:
                all_results.append(res)

        if all_results:
            table = wandb.Table(columns=["chunking_method", "index_type", "top_k",
                                          "context_hit_rate", "num_chunks"])
            for r in all_results:
                table.add_data(r["chunking_method"], r["index_type"], r["top_k"],
                               r["context_hit_rate"], r["num_chunks"])
            wandb.log({"eval_comparison_table": table})

            out = "data/eval/eval_results.json"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Saved → {out}")
            print("\n📊 Summary (sorted by context_hit_rate):")
            for r in sorted(all_results, key=lambda x: x["context_hit_rate"], reverse=True):
                print(f"  {r['chunking_method']:12s} | hit_rate={r['context_hit_rate']:.3f} | chunks={r['num_chunks']}")
    else:
        res = evaluate_config(args.method, args.index, args.top_k, config, args.qa_path)
        if res:
            wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float))})
            print(f"\n✅ {res}")

    wandb.finish()