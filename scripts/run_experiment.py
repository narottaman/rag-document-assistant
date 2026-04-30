"""
scripts/run_experiment.py

Runs ONE experiment configuration (called by W&B sweep agent).
Tests a specific chunking_method + index_type + embedding_model + top_k combination.
Logs RAGAS metrics + latency to W&B.

Can also be run standalone:
    python scripts/run_experiment.py \
        --chunking-method paragraph \
        --index-type flat \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --top-k 5

W&B sweep mode (called automatically by wandb agent):
    wandb sweep configs/sweep.yaml
    wandb agent <sweep_id>
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import wandb
from dotenv import load_dotenv

load_dotenv()


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_qa_pairs(qa_path: str) -> list:
    with open(qa_path) as f:
        return json.load(f)


def load_chunks(chunks_path: str) -> list:
    with open(chunks_path) as f:
        return json.load(f)


def run_experiment(
    chunking_method: str,
    index_type: str,
    embedding_model: str,
    top_k: int,
    config: dict,
    qa_path: str,
    processed_dir: str = "data/processed",
):
    from src.indexers import get_indexer
    from src.generator import Generator

    # ── Load chunks ──────────────────────────────────────────────────────────
    chunks_path = os.path.join(processed_dir, f"chunks_{chunking_method}.json")
    if not os.path.exists(chunks_path):
        print(f"[skip] Chunks not found: {chunks_path}. Run run_ingest.py first.")
        return None

    chunks = load_chunks(chunks_path)
    qa_pairs = load_qa_pairs(qa_path)
    print(f"[exp] {len(chunks)} chunks | {len(qa_pairs)} QA pairs")

    # ── Build index ──────────────────────────────────────────────────────────
    print(f"[exp] Building {index_type} index with {embedding_model}...")
    t_index_start = time.time()
    indexer = get_indexer(
        index_type=index_type,
        embedding_model=embedding_model,
        persist_dir=f"./chroma_db_{chunking_method}",  # separate DB per method
    )
    indexer.add(chunks)
    index_build_time = time.time() - t_index_start
    print(f"[exp] Index built in {index_build_time:.1f}s")

    # ── Run retrieval + generation ────────────────────────────────────────────
    generator = Generator()
    results = []
    retrieval_latencies = []

    for qa in qa_pairs:
        t0 = time.time()
        retrieved = indexer.search(qa["question"], top_k=top_k)
        retrieval_latencies.append(time.time() - t0)

        contexts = [r["text"] for r in retrieved]
        answer = generator.generate(qa["question"], contexts)

        results.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": qa.get("answer", qa.get("ground_truth", "")),
            "ground_truth_context": qa.get("ground_truth_context", ""),
        })

    avg_latency = sum(retrieval_latencies) / len(retrieval_latencies)
    print(f"[exp] Retrieval avg latency: {avg_latency*1000:.1f}ms")

    # ── RAGAS Evaluation ─────────────────────────────────────────────────────
    print("[exp] Running RAGAS evaluation...")
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset

        dataset = Dataset.from_list(results)
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

        ragas_scores = {
            "faithfulness":      float(scores["faithfulness"]),
            "answer_relevancy":  float(scores["answer_relevancy"]),
            "context_recall":    float(scores["context_recall"]),
        }
    except Exception as e:
        print(f"[warn] RAGAS failed: {e}. Logging dummy scores.")
        ragas_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

    # ── Context Hit Rate (manual, fast) ──────────────────────────────────────
    hits = 0
    for qa, result in zip(qa_pairs, results):
        gt_ctx = qa.get("ground_truth_context", "")
        if any(gt_ctx[:100] in ctx for ctx in result["contexts"]):
            hits += 1
    context_hit_rate = hits / len(qa_pairs) if qa_pairs else 0.0

    # ── Log everything to W&B ────────────────────────────────────────────────
    all_metrics = {
        **ragas_scores,
        "context_hit_rate":       context_hit_rate,
        "avg_retrieval_latency_ms": avg_latency * 1000,
        "index_build_time_sec":   index_build_time,
        "total_chunks":           len(chunks),
        "num_qa_pairs":           len(qa_pairs),
        "top_k":                  top_k,
    }

    wandb.log(all_metrics)

    # Log result table
    table = wandb.Table(columns=["question", "answer", "faithfulness_note"])
    for r in results[:20]:
        table.add_data(r["question"][:100], r["answer"][:200], "see ragas score")
    wandb.log({"sample_results": table})

    print(f"\n[exp] Results:")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking-method", default="paragraph",
                        choices=["fixed", "sentence", "paragraph", "semantic", "hybrid"])
    parser.add_argument("--index-type", default="flat",
                        choices=["flat", "faiss", "hnsw"])
    parser.add_argument("--embedding-model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qa-path", default="data/eval/qa_pairs_arxiv.json")
    args = parser.parse_args()

    config = load_config()

    # W&B sweep will override these via wandb.config
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="experiment",
        name=f"{args.chunking_method}_{args.index_type}_k{args.top_k}",
        config={
            "chunking_method": args.chunking_method,
            "index_type":      args.index_type,
            "embedding_model": args.embedding_model,
            "top_k":           args.top_k,
        },
    )

    # When run as part of W&B sweep, use sweep config values
    cfg = wandb.config
    chunking_method = cfg.get("chunking_method", args.chunking_method)
    index_type      = cfg.get("index_type",      args.index_type)
    embedding_model = cfg.get("embedding_model", args.embedding_model)
    top_k           = cfg.get("top_k",           args.top_k)

    run_experiment(
        chunking_method=chunking_method,
        index_type=index_type,
        embedding_model=embedding_model,
        top_k=top_k,
        config=config,
        qa_path=args.qa_path,
    )

    wandb.finish()