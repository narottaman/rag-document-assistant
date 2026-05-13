# src/vectorstore.py — single source of truth
import chromadb

CHROMA_PATH = "/data/chromadb"  # persistent Docker volume

def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name="rag_papers"):
    client = get_client()
    return client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"}
    )