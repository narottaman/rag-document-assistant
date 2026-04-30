"""
scripts/download_data.py

Downloads:
1. ArXiv PDF papers → data/raw/pdfs/
2. neural-bridge/rag-dataset-12000 from HuggingFace → data/raw/hf_dataset/
"""

import os
import requests
import json
import time
from pathlib import Path

# ─── ArXiv Papers ────────────────────────────────────────────────────────────

PAPERS = [
    {"title": "Attention Is All You Need",   "arxiv_id": "1706.03762"},
    {"title": "BERT",                         "arxiv_id": "1810.04805"},
    {"title": "GPT-3",                        "arxiv_id": "2005.14165"},
    {"title": "Deep Residual Learning",       "arxiv_id": "1512.03385"},
    {"title": "Adam Optimizer",               "arxiv_id": "1412.6980"},
    {"title": "GANs",                         "arxiv_id": "1406.2661"},
    {"title": "DALL-E 2",                     "arxiv_id": "2204.06125"},
    {"title": "Stable Diffusion",             "arxiv_id": "2112.10752"},
    {"title": "LoRA",                         "arxiv_id": "2106.09685"},
    {"title": "LLaMA",                        "arxiv_id": "2302.13971"},
]


def download_pdfs(out_dir: str = "data/raw/pdfs") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for paper in PAPERS:
        fname = f"{paper['arxiv_id'].replace('/', '_')}_{paper['title'].replace(' ', '_')}.pdf"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"[skip] {paper['title']} already downloaded")
            continue
        url = f"https://arxiv.org/pdf/{paper['arxiv_id']}"
        print(f"[download] {paper['title']} from {url}")
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            with open(fpath, "wb") as f:
                f.write(resp.content)
            print(f"  ✓ Saved → {fpath} ({len(resp.content) // 1024} KB)")
            time.sleep(2)   # be polite to arxiv
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Save metadata
    meta_path = os.path.join(out_dir, "papers_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(PAPERS, f, indent=2)
    print(f"\n[done] Metadata saved → {meta_path}")


# ─── HuggingFace RAG Dataset ─────────────────────────────────────────────────

def download_hf_dataset(out_dir: str = "data/raw/hf_dataset") -> None:
    """
    Downloads neural-bridge/rag-dataset-12000.
    Falls back to manual HTTP download if datasets library not available.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "rag_dataset_12000.json")

    if os.path.exists(out_path):
        print(f"[skip] HF dataset already downloaded at {out_path}")
        return

    try:
        from datasets import load_dataset
        print("[download] neural-bridge/rag-dataset-12000 from HuggingFace...")
        ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
        records = [dict(row) for row in ds]
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  ✓ Saved {len(records)} records → {out_path}")
    except Exception as e:
        print(f"  ✗ Failed to download HF dataset: {e}")
        print("  → Install with: pip install datasets")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Downloading ArXiv PDFs")
    print("=" * 60)
    download_pdfs()

    print("\n" + "=" * 60)
    print("Step 2: Downloading HuggingFace RAG Dataset")
    print("=" * 60)
    download_hf_dataset()

    print("\n✅ All data downloaded. Next step: python scripts/run_ingest.py")