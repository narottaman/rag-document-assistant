import chromadb
from src.embeddings import EmbeddingModel
import yaml

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

class Retriever:
    def __init__(self):
        config = load_config()
        self.top_k = config["retriever"]["top_k"]
        self.embed_model = EmbeddingModel(config["embedding"]["model"])
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("documents")

    def add_documents(self, chunks: list):
        texts = [c.page_content for c in chunks]
        embeddings = self.embed_model.embed(texts)
        ids = [f"chunk_{i}" for i in range(len(texts))]
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=ids
        )

    def retrieve(self, query: str) -> list[str]:
        query_embedding = self.embed_model.embed([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k
        )
        return results["documents"][0]