"""
app/chat.py

Gradio live chat interface for the RAG Document Assistant.
Uses Gemini 2.5 Flash Lite (cheapest stable model).
Runs in Docker on your laptop — no Sol GPU needed.

Run locally:  python app/chat.py
Run in Docker: handled by docker-compose.yml

Access: http://localhost:7860
"""

import os
import sys
import json
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexers import get_indexer
from src.generator_gemini import GeminiGenerator

# ── Load index at startup ─────────────────────────────────────────────────────

CHUNKS_PATH = os.environ.get("CHUNKS_PATH", "data/processed/chunks_paragraph.json")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K        = int(os.environ.get("TOP_K", "5"))

print(f"[chat] Loading chunks from {CHUNKS_PATH}...")
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

print(f"[chat] Building HNSW index over {len(chunks)} chunks...")
indexer = get_indexer(
    index_type="hnsw",
    embedding_model=EMBED_MODEL,
)
indexer.add(chunks)

generator = GeminiGenerator()
print("[chat] Ready.")

# ── Chat function ─────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """Multi-turn RAG chat — retrieves context, generates with Gemini."""
    if not message.strip():
        return "Please ask a question."

    # Retrieve relevant chunks
    retrieved = indexer.search(message, top_k=TOP_K)
    contexts  = [r["text"] for r in retrieved]
    sources   = list({
        f"{r.get('title', 'Unknown')} (p.{r.get('page_num', '?')})"
        for r in retrieved
    })

    # Generate answer
    answer = generator.generate(message, contexts)

    # Append source citations
    sources_text = "\n\n📄 **Sources:** " + " · ".join(sources[:3])
    return answer + sources_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="RAG Document Assistant") as demo:
    gr.Markdown("""
    # 📚 RAG Document Assistant
    Ask questions about 10 ArXiv AI papers:
    **Attention, BERT, GPT-3, ResNet, Adam, GANs, LoRA, LLaMA, DALL-E 2, Stable Diffusion**

    Powered by **sentence-transformers + HNSW + Gemini 2.5 Flash Lite**
    """)

    chatbot = gr.ChatInterface(
        fn=chat,
        examples=[
            "What is multi-head attention and why does it help?",
            "How does LoRA reduce the number of trainable parameters?",
            "What optimizer does GPT-3 use and what are its hyperparameters?",
            "How does BERT differ from GPT in its training objective?",
            "What is the U-Net architecture used for?",
        ],
        retry_btn=None,
        undo_btn=None,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   # set True to get a public gradio.live URL
    )