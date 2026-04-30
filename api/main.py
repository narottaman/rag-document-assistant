"""
api/main.py

FastAPI server for the RAG Document Assistant.
Endpoints:
  GET  /health          — health check
  POST /query           — ask a question
  GET  /stats           — index stats
  POST /ingest          — trigger ingestion (dev only)

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import json
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import yaml

load_dotenv()

app = FastAPI(
    title="RAG Document Assistant",
    description="Production RAG pipeline over ArXiv AI papers with RAGAS evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Config + lazy-loaded pipeline ───────────────────────────────────────────

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

_pipeline = None
_index_stats = {}


def get_pipeline(chunking_method: str = "paragraph", index_type: str = "flat"):
    global _pipeline, _index_stats

    if _pipeline is not None:
        return _pipeline

    from src.indexers import get_indexer
    from src.generator import Generator

    config = load_config()
    chunks_path = f"data/processed/chunks_{chunking_method}.json"

    if not os.path.exists(chunks_path):
        raise RuntimeError(
            f"Chunks not found: {chunks_path}. "
            f"Run: python scripts/run_ingest.py --method {chunking_method}"
        )

    with open(chunks_path) as f:
        chunks = json.load(f)

    indexer = get_indexer(
        index_type=index_type,
        embedding_model=config["embedding"]["model"],
        persist_dir=f"./chroma_db_{chunking_method}",
    )
    indexer.add(chunks)

    _index_stats = {
        "chunking_method": chunking_method,
        "index_type": index_type,
        "total_chunks": len(chunks),
        "embedding_model": config["embedding"]["model"],
    }

    class Pipeline:
        def __init__(self, idx, gen, top_k):
            self.indexer = idx
            self.generator = gen
            self.top_k = top_k

        def query(self, question: str) -> dict:
            retrieved = self.indexer.search(question, top_k=self.top_k)
            contexts = [r["text"] for r in retrieved]
            answer = self.generator.generate(question, contexts)
            return {
                "answer": answer,
                "contexts": contexts,
                "sources": [
                    {
                        "title": r.get("title", ""),
                        "page": r.get("page_num", ""),
                        "score": round(r.get("score", 0), 4),
                        "snippet": r["text"][:200],
                    }
                    for r in retrieved
                ],
            }

    _pipeline = Pipeline(
        idx=indexer,
        gen=Generator(),
        top_k=config["retriever"]["top_k"],
    )
    return _pipeline


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    chunking_method: Optional[str] = "paragraph"
    index_type: Optional[str] = "flat"


class QueryResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float
    sources: list
    contexts: list


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-document-assistant"}


@app.get("/stats")
def stats():
    return {"index_stats": _index_stats, "pipeline_loaded": _pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        pipeline = get_pipeline(
            chunking_method=req.chunking_method or "paragraph",
            index_type=req.index_type or "flat",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    t0 = time.time()
    result = pipeline.query(req.question)
    latency_ms = (time.time() - t0) * 1000

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        latency_ms=round(latency_ms, 2),
        sources=result["sources"],
        contexts=result["contexts"],
    )


# ─── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)