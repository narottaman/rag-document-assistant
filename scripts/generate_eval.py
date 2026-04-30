"""
scripts/generate_eval.py

Generates ground-truth Q&A pairs from chunked ArXiv PDFs using GPT-4o-mini.
Saves to data/eval/qa_pairs.json

Also filters the neural-bridge/rag-dataset-12000 HF dataset
and saves a subset to data/eval/hf_qa_subset.json

Usage:
    python scripts/generate_eval.py --n-questions 60 --chunker hybrid
"""

import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Q&A Generation from ArXiv PDF chunks
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at creating evaluation datasets for RAG systems.
Given a text chunk from an academic paper, generate ONE factual question whose answer
is explicitly contained in the chunk. The question must be specific, not vague.

Respond ONLY with valid JSON — no markdown, no preamble:
{"question": "...", "answer": "...", "chunk_id": <id>}"""


def generate_qa_from_chunk(client: OpenAI, chunk: dict, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Chunk ID: {chunk['id']}\nTitle: {chunk['title']}\n\nText:\n{chunk['text'][:1500]}",
                    },
                ],
                temperature=0.3,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            qa = json.loads(raw)
            qa["source_title"] = chunk["title"]
            qa["source_filename"] = chunk["filename"]
            qa["source_page"] = chunk.get("page_num")
            qa["ground_truth_context"] = chunk["text"]
            qa["chunk_method"] = chunk.get("chunk_method", "")
            return qa
        except Exception as e:
            print(f"  [attempt {attempt+1}] Failed: {e}")
            time.sleep(2)
    return None


def generate_arxiv_qa(
    chunks_path: str,
    out_path: str,
    n_questions: int = 60,
    skip_short: int = 150,
) -> list:
    """Load chunks JSON, sample, call GPT-4o-mini, save QA pairs."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(chunks_path) as f:
        all_chunks = json.load(f)

    # Filter out very short chunks
    valid = [c for c in all_chunks if len(c["text"]) >= skip_short]
    print(f"[eval-gen] {len(valid)} valid chunks (≥{skip_short} chars) from {len(all_chunks)} total")

    # Sample evenly across papers
    import random
    random.seed(42)

    # Try to get equal representation per paper title
    from collections import defaultdict
    by_title = defaultdict(list)
    for c in valid:
        by_title[c["title"]].append(c)

    sampled = []
    per_paper = max(1, n_questions // len(by_title))
    for title, chunks in by_title.items():
        sampled.extend(random.sample(chunks, min(per_paper, len(chunks))))

    # Fill remaining up to n_questions
    remaining = [c for c in valid if c not in sampled]
    if len(sampled) < n_questions:
        extra = random.sample(remaining, min(n_questions - len(sampled), len(remaining)))
        sampled.extend(extra)

    sampled = sampled[:n_questions]
    print(f"[eval-gen] Generating {len(sampled)} Q&A pairs using GPT-4o-mini...")

    qa_pairs = []
    for i, chunk in enumerate(sampled):
        print(f"  [{i+1}/{len(sampled)}] {chunk['title'][:40]}...")
        qa = generate_qa_from_chunk(client, chunk)
        if qa:
            qa_pairs.append(qa)
        time.sleep(0.5)  # rate limit

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\n✅ Saved {len(qa_pairs)} Q&A pairs → {out_path}")
    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Filter HuggingFace dataset subset
# ─────────────────────────────────────────────────────────────────────────────

def filter_hf_dataset(
    hf_path: str = "data/raw/hf_dataset/rag_dataset_12000.json",
    out_path: str = "data/eval/hf_qa_subset.json",
    n: int = 100,
) -> list:
    """
    Load the full HF dataset and take a clean subset.
    HF dataset format: {"question": ..., "context": ..., "answer": ...}
    We rename to match our unified format.
    """
    if not os.path.exists(hf_path):
        print(f"[hf-filter] Not found: {hf_path}. Run scripts/download_data.py first.")
        return []

    with open(hf_path) as f:
        records = json.load(f)

    # Normalize to unified format
    unified = []
    for r in records[:n]:
        unified.append({
            "question": r.get("question", ""),
            "answer": r.get("answer", r.get("ground_truth", "")),
            "ground_truth_context": r.get("context", r.get("contexts", "")),
            "source_title": "HuggingFace RAG Dataset",
            "source_filename": "hf_rag_12000",
            "chunk_method": "hf_prebuilt",
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(unified, f, indent=2)

    print(f"[hf-filter] Saved {len(unified)} records → {out_path}")
    return unified


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate eval Q&A pairs")
    parser.add_argument("--chunks-path", default="data/processed/chunks_hybrid.json",
                        help="Path to chunked JSON (output of run_ingest.py)")
    parser.add_argument("--out-path", default="data/eval/qa_pairs_arxiv.json",
                        help="Where to save generated Q&A pairs")
    parser.add_argument("--n-questions", type=int, default=60,
                        help="Number of Q&A pairs to generate from ArXiv PDFs")
    parser.add_argument("--hf-subset", type=int, default=100,
                        help="Number of Q&A pairs to take from HF dataset")
    args = parser.parse_args()

    # Step 1: ArXiv-based Q&A
    print("=" * 60)
    print("Generating Q&A from ArXiv PDF chunks")
    print("=" * 60)
    if os.path.exists(args.chunks_path):
        generate_arxiv_qa(
            chunks_path=args.chunks_path,
            out_path=args.out_path,
            n_questions=args.n_questions,
        )
    else:
        print(f"Chunks not found at {args.chunks_path}.")
        print("Run: python scripts/run_ingest.py --method hybrid first.")

    # Step 2: HF dataset subset
    print("\n" + "=" * 60)
    print("Filtering HuggingFace RAG dataset")
    print("=" * 60)
    filter_hf_dataset(n=args.hf_subset)

    print("\n✅ Eval data ready. Next: python scripts/run_experiment.py")