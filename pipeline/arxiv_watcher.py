"""
pipeline/arxiv_watcher.py

Prefect pipeline — watches ArXiv RSS, downloads new papers,
chunks them, embeds, and adds to ChromaDB automatically.

Run once:    python pipeline/arxiv_watcher.py
Run on cron: add to crontab or use Prefect serve

Install: pip install prefect feedparser
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prefect import flow, task
import feedparser
import requests
import wandb

# ArXiv categories to watch
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]
PDF_DIR   = Path("data/raw/pdfs")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Track which papers we already ingested
SEEN_FILE = Path("data/processed/seen_papers.json")


def load_seen() -> set:
    if SEEN_FILE.exists():
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen(seen: set):
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen), f)


@task(retries=3, retry_delay_seconds=30)
def fetch_new_papers(category: str, seen: set) -> list[dict]:
    """Fetch latest papers from ArXiv RSS, skip already-seen ones."""
    url = f"https://export.arxiv.org/rss/{category}"
    feed = feedparser.parse(url)

    new_papers = []
    for entry in feed.entries[:20]:
        arxiv_id = entry.id.split("/abs/")[-1].strip()
        if arxiv_id not in seen:
            new_papers.append({
                "arxiv_id": arxiv_id,
                "title": entry.title.replace("\n", " ").strip(),
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            })

    print(f"[fetch] {category}: {len(new_papers)} new papers")
    return new_papers


@task(retries=2, retry_delay_seconds=60)
def download_pdf(paper: dict) -> str | None:
    """Download PDF to data/raw/pdfs/. Returns local path or None on failure."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = paper["arxiv_id"].replace("/", "_")
    fpath = PDF_DIR / f"{safe_id}.pdf"

    if fpath.exists():
        return str(fpath)

    try:
        resp = requests.get(
            paper["pdf_url"], timeout=60,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        fpath.write_bytes(resp.content)
        print(f"[download] {paper['title'][:50]} → {fpath.name}")
        time.sleep(2)  # polite to ArXiv
        return str(fpath)
    except Exception as e:
        print(f"[download] Failed {paper['arxiv_id']}: {e}")
        return None


@task
def chunk_and_store(pdf_path: str, title: str) -> int:
    """
    Chunk PDF with paragraph chunker, embed, add to ChromaDB.
    Returns number of chunks added.
    """
    from src.chunkers import get_chunker
    from sentence_transformers import SentenceTransformer
    import chromadb

    chunker = get_chunker("paragraph", chunk_size=512, chunk_overlap=50)
    chunks  = chunker.chunk(pdf_path=pdf_path, title=title)

    if not chunks:
        return 0

    # Embed
    model = SentenceTransformer(EMBED_MODEL)
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Store in ChromaDB
    chroma_path = os.environ.get("CHROMA_PATH", "./chroma_db_pipeline")
    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        "rag_papers",
        metadata={"hnsw:space": "cosine"}
    )

    ids       = [f"{title[:15]}_{c['id']}" for c in chunks]
    metadatas = [{
        "title":        c["title"],
        "page_num":     str(c.get("page_num", "")),
        "chunk_method": c["chunk_method"],
    } for c in chunks]

    # Batch insert
    for i in range(0, len(texts), 500):
        collection.add(
            documents=texts[i:i+500],
            embeddings=embeddings[i:i+500].tolist(),
            ids=ids[i:i+500],
            metadatas=metadatas[i:i+500],
        )

    print(f"[store] {title[:40]}: {len(chunks)} chunks added")
    return len(chunks)


@task
def log_pipeline_run(papers_processed: int, chunks_added: int):
    """Log pipeline stats to W&B."""
    try:
        wandb.init(
            project="rag-document-assistant",
            entity=os.environ.get("WANDB_ENTITY", "ngangada-arizona-state-university"),
            job_type="pipeline",
            reinit=True,
        )
        wandb.log({
            "pipeline/papers_processed": papers_processed,
            "pipeline/chunks_added": chunks_added,
        })
        wandb.finish()
    except Exception as e:
        print(f"[wandb] Logging failed: {e}")


@flow(name="arxiv-rag-pipeline", log_prints=True)
def run_pipeline():
    """
    Main Prefect flow:
    1. Fetch new papers from ArXiv RSS
    2. Download PDFs
    3. Chunk + embed + store in ChromaDB
    4. Log to W&B
    """
    seen = load_seen()
    all_new_papers = []

    for category in ARXIV_CATEGORIES:
        papers = fetch_new_papers(category, seen)
        all_new_papers.extend(papers)

    if not all_new_papers:
        print("[pipeline] No new papers found.")
        return

    total_chunks = 0
    processed    = 0

    for paper in all_new_papers:
        pdf_path = download_pdf(paper)
        if pdf_path:
            chunk_count = chunk_and_store(pdf_path, paper["title"])
            total_chunks += chunk_count
            seen.add(paper["arxiv_id"])
            processed += 1

    save_seen(seen)
    log_pipeline_run(processed, total_chunks)

    print(f"\n✅ Pipeline complete: {processed} papers, {total_chunks} chunks added")


if __name__ == "__main__":
    run_pipeline()