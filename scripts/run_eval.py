"""
scripts/run_eval.py

Runs RAGAS evaluation on the best chunking+index config found from sweep,
OR evaluates all configs and produces a comparison table.

Usage:
    # Evaluate a single config
    python scripts/run_eval.py --method hybrid --index hnsw --top-k 5

    # Evaluate all saved chunk methods against flat index (quick comparison)
    python scripts/run_eval.py --all
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
EVAL_DIR = "data/eval"


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_config(
    chunking_method: str,
    index_type: str,
    top_k: int,
    config: dict,
    qa_path: str,
) -> dict:
    from src.indexers import get_indexer
    from src.generator import Generator

    chunks_path = os.path.join(PROCESSED_DIR, f"chunks_{chunking_method}.json")
    if not os.path.exists(chunks_path):
        print(f"[skip] {chunks_path} not found")
        return {}

    with open(chunks_path) as f:
        chunks = json.load(f)
    with open(qa_path) as f:
        qa_pairs = json.load(f)

    indexer = get_indexer(
        index_type=index_type,
        embedding_model=config["embedding"]["model"],
        persist_dir=f"./chroma_db_{chunking_method}",
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
            "ground_truth": qa.get("answer", ""),
        })

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset

        dataset = Dataset.from_list(results)
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
        return {
            "chunking_method": chunking_method,
            "index_type": index_type,
            "top_k": top_k,
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_recall": float(scores["context_recall"]),
            "num_chunks": len(chunks),
        }
    except Exception as e:
        print(f"[warn] RAGAS error: {e}")
        return {"chunking_method": chunking_method, "index_type": index_type, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "semantic", "hybrid"])
    parser.add_argument("--index", default="flat",
                        choices=["flat", "faiss", "hnsw"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qa-path", default="data/eval/qa_pairs_arxiv.json")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all available chunking methods")
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
            chunks_path = os.path.join(PROCESSED_DIR, f"chunks_{method}.json")
            if not os.path.exists(chunks_path):
                print(f"[skip] {method} chunks not found")
                continue
            print(f"\n[eval] method={method}, index=flat, top_k={args.top_k}")
            res = evaluate_config(method, "flat", args.top_k, config, args.qa_path)
            if res:
                all_results.append(res)

        # Log comparison table to W&B
        if all_results:
            table = wandb.Table(
                columns=["chunking_method", "index_type", "top_k",
                         "faithfulness", "answer_relevancy", "context_recall", "num_chunks"]
            )
            for r in all_results:
                table.add_data(
                    r.get("chunking_method", ""),
                    r.get("index_type", ""),
                    r.get("top_k", 0),
                    r.get("faithfulness", 0),
                    r.get("answer_relevancy", 0),
                    r.get("context_recall", 0),
                    r.get("num_chunks", 0),
                )
            wandb.log({"eval_comparison_table": table})

            # Save results locally
            out_path = "data/eval/eval_results.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Results saved → {out_path}")
            print("\n📊 Summary:")
            for r in all_results:
                print(f"  {r.get('chunking_method','?'):12s} | "
                      f"faithfulness={r.get('faithfulness',0):.3f} | "
                      f"recall={r.get('context_recall',0):.3f}")
    else:
        res = evaluate_config(args.method, args.index, args.top_k, config, args.qa_path)
        if res:
            wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float))})
            print(f"\n✅ Eval complete: {res}")

    wandb.finish()