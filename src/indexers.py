"""
src/indexers.py

Three index backends behind a unified interface:
  - ChromaFlat   : ChromaDB with flat (brute-force) index — exact, small corpus
  - FAISSIndex   : FAISS IVF — approximate, large corpus, GPU-ready on Sol
  - HNSWIndex    : hnswlib HNSW — approximate, fast in-memory

All indexers expose:
  .add(chunks: list[dict])                 → build the index
  .search(query: str, top_k: int)          → list[dict]  retrieved chunks
  .save(path: str) / .load(path: str)      → persistence
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import json
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────

class BaseIndexer(ABC):

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self._embed_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._get_embed_model()
        return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    @abstractmethod
    def add(self, chunks: List[Dict[str, Any]]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# CHROMA FLAT (exact cosine search)
# ─────────────────────────────────────────────────────────────────────────────

class ChromaFlatIndexer(BaseIndexer):
    """
    ChromaDB persistent store with default HNSW index (exact-ish).
    Best for < 50K chunks. Zero external deps beyond chromadb.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_chunks",
    ):
        super().__init__(embedding_model_name)
        import chromadb
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = self._encode(texts)
        ids = [f"chunk_{c['id']}" for c in chunks]
        metadatas = [
            {
                "title": c.get("title", ""),
                "filename": c.get("filename", ""),
                "heading": c.get("heading") or "",
                "page_num": str(c.get("page_num") or ""),
                "chunk_method": c.get("chunk_method", ""),
            }
            for c in chunks
        ]
        # Add in batches of 500 to avoid memory issues
        batch = 500
        for i in range(0, len(texts), batch):
            self.collection.add(
                documents=texts[i: i + batch],
                embeddings=embeddings[i: i + batch].tolist(),
                ids=ids[i: i + batch],
                metadatas=metadatas[i: i + batch],
            )
        print(f"[ChromaFlat] Indexed {len(texts)} chunks → {self.persist_dir}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self._encode([query])
        results = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "score": 1 - dist, **meta})
        return output

    def save(self, path: str) -> None:
        # ChromaDB persists automatically
        print(f"[ChromaFlat] Already persisted at {self.persist_dir}")

    def load(self, path: str) -> None:
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(self.collection_name)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS IVF (approximate, scales to millions)
# ─────────────────────────────────────────────────────────────────────────────

class FAISSIndexer(BaseIndexer):
    """
    FAISS IndexIVFFlat — inverted file index.
    Clusters vectors into n_lists buckets; searches only nearest buckets.
    GPU-accelerated on Sol with faiss-gpu.
    Requires: pip install faiss-cpu  (or faiss-gpu on Sol)
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_lists: int = 100,          # number of Voronoi cells
        n_probe: int = 10,           # cells to search at query time
        use_gpu: bool = False,
    ):
        super().__init__(embedding_model_name)
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.use_gpu = use_gpu
        self.index = None
        self.chunks_store: List[Dict[str, Any]] = []  # parallel list to FAISS ids

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        import faiss

        texts = [c["text"] for c in chunks]
        embeddings = self._encode(texts).astype("float32")
        dim = embeddings.shape[1]

        quantizer = faiss.IndexFlatIP(dim)          # inner product (cosine after normalize)
        self.index = faiss.IndexIVFFlat(quantizer, dim, min(self.n_lists, len(chunks)))

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("[FAISS] Using GPU")
            except Exception:
                print("[FAISS] GPU not available, falling back to CPU")

        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = self.n_probe
        self.chunks_store = chunks
        print(f"[FAISS-IVF] Indexed {len(texts)} chunks (n_lists={self.n_lists}, nprobe={self.n_probe})")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self._encode([query]).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        output = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks_store[idx].copy()
            chunk["score"] = float(score)
            output.append(chunk)
        return output

    def save(self, path: str) -> None:
        import faiss
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(self.chunks_store, f)
        print(f"[FAISS] Saved → {path}")

    def load(self, path: str) -> None:
        import faiss
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        self.index.nprobe = self.n_probe
        with open(os.path.join(path, "chunks.json")) as f:
            self.chunks_store = json.load(f)
        print(f"[FAISS] Loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HNSW (Hierarchical Navigable Small World)
# ─────────────────────────────────────────────────────────────────────────────

class HNSWIndexer(BaseIndexer):
    """
    hnswlib HNSW index — graph-based approximate nearest neighbor.
    Best latency at query time. Pure CPU but extremely fast.
    Requires: pip install hnswlib
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        space: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ):
        super().__init__(embedding_model_name)
        self.space = space
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.index = None
        self.chunks_store: List[Dict[str, Any]] = []

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        import hnswlib

        texts = [c["text"] for c in chunks]
        embeddings = self._encode(texts).astype("float32")
        dim = embeddings.shape[1]
        n = len(texts)

        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(
            max_elements=n,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.add_items(embeddings, list(range(n)))
        self.index.set_ef(self.ef_search)
        self.chunks_store = chunks
        print(f"[HNSW] Indexed {n} chunks (M={self.M}, ef_construction={self.ef_construction})")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self._encode([query]).astype("float32")
        labels, distances = self.index.knn_query(q_emb, k=top_k)
        output = []
        for idx, dist in zip(labels[0], distances[0]):
            chunk = self.chunks_store[idx].copy()
            chunk["score"] = float(1 - dist)  # cosine distance → similarity
            output.append(chunk)
        return output

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.index.save_index(os.path.join(path, "hnsw.bin"))
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(self.chunks_store, f)
        print(f"[HNSW] Saved → {path}")

    def load(self, path: str) -> None:
        import hnswlib
        with open(os.path.join(path, "chunks.json")) as f:
            self.chunks_store = json.load(f)
        dim = len(self._encode(["warmup"])[0])
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.load_index(os.path.join(path, "hnsw.bin"), max_elements=len(self.chunks_store))
        self.index.set_ef(self.ef_search)
        print(f"[HNSW] Loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_indexer(index_type: str, embedding_model: str, **kwargs) -> BaseIndexer:
    """
    Factory function.
    Usage: get_indexer("flat", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
           get_indexer("faiss", embedding_model=..., n_lists=50, use_gpu=True)
           get_indexer("hnsw",  embedding_model=..., M=32)
    """
    mapping = {
        "flat":  ChromaFlatIndexer,
        "faiss": FAISSIndexer,
        "hnsw":  HNSWIndexer,
    }
    if index_type not in mapping:
        raise ValueError(f"Unknown index '{index_type}'. Choose from: {list(mapping.keys())}")
    return mapping[index_type](embedding_model_name=embedding_model, **kwargs)