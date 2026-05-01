"""
scripts/generate_eval.py

Generates ground-truth Q&A pairs from chunked ArXiv PDFs using Gemini Flash.
Uses new google-genai SDK (not deprecated google.generativeai).

Usage:
    export GOOGLE_API_KEY=your-key
    python scripts/generate_eval.py \
        --chunks-path data/processed/chunks_paragraph.json \
        --out-path data/eval/qa_pairs_arxiv.json \
        --n-questions 60
"""

import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are an expert at creating evaluation datasets for RAG systems.
Given a text chunk from an academic paper, generate ONE factual question whose answer
is explicitly contained in the chunk. The question must be specific, not vague.

Respond ONLY with valid JSON — no markdown, no preamble:
{"question": "...", "answer": "...", "chunk_id": <id>}"""


def get_gemini_client():
    try:
        from google import genai
    except ImportError:
        raise ImportError("Run: pip install google-genai")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set: export GOOGLE_API_KEY=your-key")
    return genai.Client(api_key=api_key)


def generate_qa_from_chunk(client, chunk: dict, retries: int = 3) -> dict | None:
    prompt = f"""{SYSTEM_PROMPT}

Chunk ID: {chunk['id']}
Title: {chunk['title']}

Text:
{chunk['text'][:1500]}"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            qa = json.loads(raw)
            qa["source_title"] = chunk["title"]
            qa["source_filename"] = chunk["filename"]
            qa["source_page"] = chunk.get("page_num")
            qa["ground_truth_context"] = chunk["text"]
            qa["chunk_method"] = chunk.get("chunk_method", "")
            return qa
        except json.JSONDecodeError as e:
            print(f"  [attempt {attempt+1}] JSON parse failed: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  [attempt {attempt+1}] Failed: {e}")
            time.sleep(2)
    return None


def generate_arxiv_qa(chunks_path, out_path, n_questions=60, skip_short=150):
    client = get_gemini_client()

    with open(chunks_path) as f:
        all_chunks = json.load(f)

    valid = [c for c in all_chunks if len(c["text"]) >= skip_short]
    print(f"[eval-gen] {len(valid)} valid chunks from {len(all_chunks)} total")

    import random
    from collections import defaultdict
    random.seed(42)

    by_title = defaultdict(list)
    for c in valid:
        by_title[c["title"]].append(c)

    sampled = []
    per_paper = max(1, n_questions // len(by_title))
    for title, chunks in by_title.items():
        sampled.extend(random.sample(chunks, min(per_paper, len(chunks))))

    already = set(id(c) for c in sampled)
    remaining = [c for c in valid if id(c) not in already]
    if len(sampled) < n_questions:
        extra = random.sample(remaining, min(n_questions - len(sampled), len(remaining)))
        sampled.extend(extra)

    sampled = sampled[:n_questions]
    print(f"[eval-gen] Generating {len(sampled)} Q&A pairs using {GEMINI_MODEL}...")

    qa_pairs = []
    for i, chunk in enumerate(sampled):
        print(f"  [{i+1}/{len(sampled)}] {chunk['title'][:40]}...")
        qa = generate_qa_from_chunk(client, chunk)
        if qa:
            qa_pairs.append(qa)
        time.sleep(0.3)  # free tier: 15 RPM limit

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\n✅ Saved {len(qa_pairs)} Q&A pairs → {out_path}")
    return qa_pairs


def filter_hf_dataset(hf_path="data/raw/hf_dataset/rag_dataset_12000.json",
                       out_path="data/eval/hf_qa_subset.json", n=100):
    if not os.path.exists(hf_path):
        print(f"[hf-filter] Not found: {hf_path}")
        return []

    with open(hf_path) as f:
        records = json.load(f)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", default="data/processed/chunks_paragraph.json")
    parser.add_argument("--out-path", default="data/eval/qa_pairs_arxiv.json")
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--hf-subset", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Q&A from ArXiv PDF chunks")
    print("=" * 60)
    if os.path.exists(args.chunks_path):
        generate_arxiv_qa(args.chunks_path, args.out_path, args.n_questions)
    else:
        print(f"Chunks not found: {args.chunks_path}")

    print("\n" + "=" * 60)
    print("Filtering HuggingFace RAG dataset")
    print("=" * 60)
    filter_hf_dataset(n=args.hf_subset)

    print("\n✅ Eval data ready.")