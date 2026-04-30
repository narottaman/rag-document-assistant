from sentence_transformers import SentenceTransformer
import wandb
import time

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list:
        start = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        elapsed = time.time() - start

        # W&B: log embedding speed
        if wandb.run:
            wandb.log({
                "embedding_time_sec": elapsed,
                "texts_per_sec": len(texts) / elapsed,
                "embedding_model": self.model_name
            })
        return embeddings