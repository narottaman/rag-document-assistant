
"""

scripts/run_experiment.py

Runs ONE experiment (called by W&B sweep agent).

Metrics logged: context_hit_rate, avg_retrieval_latency_ms,

                index_build_time_sec, total_chunks, top_k

RAGAS LLM metrics skipped — uses context_hit_rate as primary retrieval metric.

"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, wandb

from dotenv import load_dotenv

load_dotenv()



def load_config(path="configs/config.yaml"):

    with open(path) as f:

        return yaml.safe_load(f)



def run_experiment(chunking_method, index_type, embedding_model, top_k, config, qa_path, processed_dir="data/processed"):

    from src.indexers import get_indexer

    from src.generator import Generator



    chunks_path = os.path.join(processed_dir, f"chunks_{chunking_method}.json")

    if not os.path.exists(chunks_path):

        print(f"[skip] {chunks_path} not found")

        wandb.log({"error": "chunks_not_found"})

        return None

    if not os.path.exists(qa_path):

        print(f"[skip] {qa_path} not found")

        wandb.log({"error": "qa_not_found"})

        return None



    with open(chunks_path) as f:

        chunks = json.load(f)

    with open(qa_path) as f:

        qa_pairs = json.load(f)

    print(f"[exp] {len(chunks)} chunks | {len(qa_pairs)} QA pairs")



    # Build index

    t0 = time.time()

    extra = {"persist_dir": f"./chroma_db_{chunking_method}"} if index_type == "flat" else {}

    indexer = get_indexer(

        index_type=index_type,

        embedding_model=embedding_model,

        **extra,

    )

    indexer.add(chunks)

    index_build_time = time.time() - t0

    print(f"[exp] Index built in {index_build_time:.1f}s")



    # Retrieval + generation

    generator = Generator()

    results, latencies = [], []

    for qa in qa_pairs:

        t1 = time.time()

        retrieved = indexer.search(qa["question"], top_k=top_k)

        latencies.append(time.time() - t1)

        contexts = [r["text"] for r in retrieved]

        answer = generator.generate(qa["question"], contexts)

        results.append({

            "question": qa["question"],

            "answer": answer,

            "contexts": contexts,

            "ground_truth": qa.get("answer", qa.get("ground_truth", "")),

            "ground_truth_context": qa.get("ground_truth_context", ""),

        })



    avg_latency = sum(latencies) / len(latencies)



    # Context hit rate — primary retrieval metric, no LLM needed

    hits = 0

    for qa, result in zip(qa_pairs, results):

        gt_ctx = qa.get("ground_truth_context", "")

        if gt_ctx and any(gt_ctx[:80] in ctx for ctx in result["contexts"]):

            hits += 1

    context_hit_rate = hits / len(qa_pairs) if qa_pairs else 0.0



    metrics = {

        "context_hit_rate":         context_hit_rate,

        "avg_retrieval_latency_ms": avg_latency * 1000,

        "index_build_time_sec":     index_build_time,

        "total_chunks":             len(chunks),

        "num_qa_pairs":             len(qa_pairs),

        "top_k":                    top_k,

    }

    wandb.log(metrics)



    # Sample table

    table = wandb.Table(columns=["question", "answer", "context_hit"])

    for i, r in enumerate(results[:20]):

        gt = qa_pairs[i].get("ground_truth_context", "")

        hit = any(gt[:80] in ctx for ctx in r["contexts"]) if gt else False

        table.add_data(r["question"][:100], r["answer"][:200], str(hit))

    wandb.log({"sample_results": table})



    print(f"[exp] context_hit_rate={context_hit_rate:.3f} | latency={avg_latency*1000:.1f}ms")

    return metrics



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--chunking_method", default="paragraph",

                        choices=["fixed","sentence","paragraph","semantic","hybrid"])

    parser.add_argument("--index_type", default="flat",

                        choices=["flat","faiss","hnsw"])

    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--qa_path", default="data/eval/hf_qa_subset.json")

    args = parser.parse_args()



    config = load_config()

    wandb.init(

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

    cfg = wandb.config

    run_experiment(

        chunking_method=cfg.get("chunking_method", args.chunking_method),

        index_type=cfg.get("index_type", args.index_type),

        embedding_model=cfg.get("embedding_model", args.embedding_model),

        top_k=cfg.get("top_k", args.top_k),

        config=config,

        qa_path=args.qa_path,

    )

    wandb.finish()

