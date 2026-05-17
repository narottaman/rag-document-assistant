import os
import sys
import json
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexers import get_indexer
from src.generator_gemini import GeminiGenerator

CHUNKS_PATH = os.environ.get("CHUNKS_PATH", "data/processed/chunks_paragraph.json")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "5"))

print("[chat] Loading chunks from " + CHUNKS_PATH)
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

print("[chat] Building HNSW index over " + str(len(chunks)) + " chunks...")
indexer = get_indexer(index_type="hnsw", embedding_model=EMBED_MODEL)
indexer.add(chunks)

generator = GeminiGenerator()
print("[chat] Ready.")


def chat(message, history):
    if not message.strip():
        return "Please ask a question."
    retrieved = indexer.search(message, top_k=TOP_K)
    contexts = [r["text"] for r in retrieved]
    sources = list({
        r.get("title", "Unknown") + " p." + str(r.get("page_num", "?"))
        for r in retrieved
    })
    answer = generator.generate(message, contexts)
    return answer + "\n\nSources: " + " | ".join(sources[:3])


with gr.Blocks(title="RAG Document Assistant") as demo:
    gr.Markdown("# RAG Document Assistant\nAsk questions about 10 ArXiv AI papers: Attention, BERT, GPT-3, ResNet, Adam, GANs, LoRA, LLaMA, DALL-E 2, Stable Diffusion\n\nPowered by sentence-transformers + HNSW + Gemini 2.5 Flash Lite")
    gr.ChatInterface(
        fn=chat,
        examples=[
            "What is multi-head attention?",
            "How does LoRA reduce the number of trainable parameters?",
            "What optimizer does GPT-3 use?",
            "How does BERT differ from GPT?",
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )