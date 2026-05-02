# RAG Document Assistant

> Production-grade Retrieval-Augmented Generation pipeline over ArXiv AI papers.  
> Systematically evaluates **3 chunking strategies × 2 index types × 2 embedding models × 3 top-k values** — 36+ experiment runs tracked via W&B Sweeps on ASU Sol HPC (A100 GPU).

[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-orange)](https://wandb.ai/ngangada-arizona-state-university/rag-document-assistant)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRE-PRODUCTION                          │
│                                                             │
│  ArXiv PDFs ──► Chunker ──► Embedding Model ──► Vector DB   │
│  (10 papers)   [5 methods]   [2 models]        [3 indexes]  │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      IN PRODUCTION                          │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ▼                                                      │
│  Embed Query ──► Vector Search ──► Top-K Chunks             │
│                                        │                    │
│                                        ▼                    │
│                               LLM Prompt Builder            │
│                                        │                    │
│                                        ▼                    │
│                            Qwen2.5-3B / GPT-4o-mini         │
│                                        │                    │
│                                        ▼                    │
│                                   Final Answer              │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiment Results

> 36+ runs tracked via W&B Sweeps on ASU Sol HPC (A100 GPU).  
> Metric: **Context Hit Rate** — whether the ground-truth context appeared in top-k retrieved chunks.  
> Eval set: 100 Q&A pairs from `neural-bridge/rag-dataset-12000` (HuggingFace).

### Full Results Table

| Chunking | Index | Embedding Model | top_k | Context Hit Rate | Latency (ms) |
|----------|-------|----------------|-------|-----------------|-------------|
| **paragraph** | **HNSW** | **bge-small-en-v1.5** | **10** | **0.87 🏆** | **16.3** |
| paragraph | HNSW | bge-small-en-v1.5 | 5 | 0.86 | 15.9 |
| paragraph | HNSW | all-MiniLM-L6-v2 | 10 | 0.86 | 14.0 |
| paragraph | flat | all-MiniLM-L6-v2 | 10 | 0.85 | 30.2 |
| paragraph | HNSW | all-MiniLM-L6-v2 | 5 | 0.78 | 13.1 |
| paragraph | flat | all-MiniLM-L6-v2 | 5 | 0.78 | 31.4 |
| paragraph | HNSW | bge-small-en-v1.5 | 3 | 0.81 | 15.6 |
| paragraph | flat | all-MiniLM-L6-v2 | 3 | 0.75 | 39.5 |
| fixed | HNSW | all-MiniLM-L6-v2 | 10 | 0.60 | 14.3 |
| fixed | flat | all-MiniLM-L6-v2 | 10 | 0.60 | 36.4 |
| fixed | HNSW | all-MiniLM-L6-v2 | 5 | 0.54 | 13.4 |
| fixed | flat | all-MiniLM-L6-v2 | 5 | 0.56 | 40.1 |
| sentence | HNSW | all-MiniLM-L6-v2 | 10 | 0.54 | 12.5 |
| sentence | flat | all-MiniLM-L6-v2 | 10 | 0.55 | 36.7 |
| sentence | HNSW | bge-small-en-v1.5 | 10 | 0.54 | 15.7 |
| sentence | flat | all-MiniLM-L6-v2 | 5 | 0.43 | 42.8 |

> Note: BAAI/bge-small-en-v1.5 with flat (ChromaDB) index showed anomalously low hit rates (0.02–0.16)
> due to a known embedding normalization mismatch with ChromaDB's cosine space — HNSW handled it correctly.

### Key Findings

**1. Paragraph chunking dominates.**  
Recursive splitting on `\n\n → \n → ". "` preserves semantic units that align naturally with how questions are asked. Fixed-size chunking cuts mid-sentence, losing context. Sentence chunking produces chunks too short to contain complete answers.

**2. HNSW beats flat index on latency with no accuracy loss.**  
HNSW averages **13–16ms** retrieval vs **30–50ms** for ChromaDB flat — a 2–3× speedup at identical hit rates. For production systems this matters at scale.

**3. all-MiniLM-L6-v2 is more robust than bge-small-en-v1.5 for this corpus.**  
bge performs well with HNSW (0.81–0.87) but collapses with ChromaDB flat (0.02–0.16). MiniLM is consistent across both index types. For production, bge + HNSW is the best combination; MiniLM is the safer default.

**4. top_k=10 consistently outperforms top_k=3.**  
Hit rate improves 10–15% going from k=3 to k=10 with minimal latency cost on HNSW. Flat index latency barely changes since it's brute-force regardless.

**Best config:** `paragraph + HNSW + bge-small-en-v1.5 + top_k=10` → **87% context hit rate at 16ms**

---

## Chunking Strategies

| Method | Overlap | Description |
|--------|---------|-------------|
| **Fixed** | 50 chars | Splits every N characters regardless of sentence boundaries |
| **Sentence** | 1 sentence | Groups N sentences per chunk, preserves grammatical units |
| **Paragraph** | 50 chars | Recursive splitting: `\n\n → \n → ". " → " "` — best for prose |
| **Semantic** | None | Groups sentences by cosine similarity drop (GPU-accelerated on Sol) |
| **Hybrid (Docling)** | None | Layout-aware: respects headings, tables, figure captions |

---

## Indexing Strategies

| Index | Algorithm | Best For | Query Latency |
|-------|-----------|----------|---------------|
| **Flat** (ChromaDB) | Brute-force cosine | Small corpus <50K chunks, exact search | 30–50ms |
| **HNSW** (hnswlib) | Graph-based ANN | Production, fast approximate search | 12–16ms |
| **FAISS-IVF** | Inverted file index | Very large corpus, GPU-accelerated | varies |

---

## Corpus

**10 ArXiv AI papers** parsed with `pypdf` (torch-free for Sol HPC compatibility):

| Paper | ArXiv ID |
|-------|----------|
| Attention Is All You Need | 1706.03762 |
| BERT | 1810.04805 |
| GPT-3 | 2005.14165 |
| Deep Residual Learning (ResNet) | 1512.03385 |
| Adam Optimizer | 1412.6980 |
| GANs | 1406.2661 |
| DALL-E 2 | 2204.06125 |
| Stable Diffusion | 2112.10752 |
| LoRA | 2106.09685 |
| LLaMA | 2302.13971 |

**Eval set:** 100 Q&A pairs from `neural-bridge/rag-dataset-12000` (HuggingFace)

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-document-assistant
cd rag-document-assistant
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with OPENAI_API_KEY (or GOOGLE_API_KEY) and WANDB_API_KEY

# 3. Download data
python scripts/download_data.py

# 4. Ingest with all chunking methods
python scripts/run_ingest.py --method fixed
python scripts/run_ingest.py --method sentence
python scripts/run_ingest.py --method paragraph

# 5. Generate eval QA pairs
python scripts/generate_eval.py \
    --chunks-path data/processed/chunks_paragraph.json \
    --out-path data/eval/qa_pairs_arxiv.json

# 6. Run a question interactively
python scripts/run_query.py --method paragraph --index hnsw

# 7. Run full evaluation
python scripts/run_eval.py --all

# 8. Run W&B sweep (all combinations)
wandb sweep configs/sweep.yaml
wandb agent YOUR_ENTITY/rag-document-assistant/SWEEP_ID

# 9. Start API server
uvicorn api.main:app --reload --port 8000
```

---

## W&B Experiment Tracking

Every run logs:

| Metric | Description |
|--------|-------------|
| `context_hit_rate` | % of questions where ground-truth chunk was in top-k results |
| `avg_retrieval_latency_ms` | Vector search speed per query |
| `index_build_time_sec` | Time to embed + index all chunks |
| `total_chunks` | Chunks produced by this chunking method |
| `num_qa_pairs` | Size of evaluation set |
| `top_k` | Number of chunks retrieved per query |

**W&B Project:** [ngangada-arizona-state-university/rag-document-assistant](https://wandb.ai/ngangada-arizona-state-university/rag-document-assistant)

---

## Sol HPC Setup (GPU-Accelerated)

Semantic chunking and FAISS-GPU indexing run on ASU Sol HPC (A100):

```bash
# Copy project to Sol
scp -r . YOUR_ASURITE@sol.asu.edu:~/rag-document-assistant

# First-time venv setup on Sol
source $HOME/.venv/bin/activate
pip install pypdf nltk faiss-cpu hnswlib chromadb \
            sentence-transformers langchain langchain-community \
            openai wandb fastapi uvicorn python-dotenv pyyaml \
            ragas datasets google-genai

# Submit embedding job
sbatch sol/embed_job.slurm

# Monitor
squeue -u $USER
tail -f logs/embed_<job_id>.out
```

GPU metrics (utilization, memory, temperature) are captured automatically in W&B's System tab.

**Important:** Use `$HOME/.venv` not base conda — Sol's base conda has a broken torchvision/sqlite3 that conflicts with PyPDF and NLTK imports. The project uses `pypdf` directly (no LangChain PDF loader) to avoid this.

---

## Project Structure

```
rag-document-assistant/
├── data/
│   ├── raw/pdfs/           # Downloaded ArXiv PDFs
│   ├── raw/hf_dataset/     # HuggingFace RAG dataset
│   ├── processed/          # chunks_{method}.json files
│   └── eval/               # qa_pairs_arxiv.json, eval_results.json
├── src/
│   ├── chunkers.py         # 5 chunking strategies (torch-free PDF loading)
│   ├── indexers.py         # ChromaFlat, FAISS-IVF, HNSW
│   ├── embeddings.py       # Embedding model + W&B speed logging
│   ├── generator.py        # LLM answer generation
│   ├── pipeline.py         # End-to-end RAG pipeline
│   └── evaluate.py         # RAGAS evaluation runner
├── scripts/
│   ├── download_data.py    # Download PDFs + HF dataset
│   ├── run_ingest.py       # Chunk PDFs → save JSON + log to W&B
│   ├── generate_eval.py    # Generate Q&A eval pairs (Gemini/OpenAI)
│   ├── run_experiment.py   # Single experiment (W&B sweep target)
│   ├── run_query.py        # Interactive query CLI
│   └── run_eval.py         # Full evaluation + comparison table
├── api/
│   └── main.py             # FastAPI server
├── configs/
│   ├── config.yaml         # Model + retriever config
│   └── sweep.yaml          # W&B grid sweep definition
├── sol/
│   └── embed_job.slurm     # ASU Sol HPC SLURM job script
└── tests/
    └── test_pipeline.py    # Unit tests for chunkers + indexers
```

---

## Tech Stack

`pypdf` · `LangChain` · `ChromaDB` · `FAISS` · `hnswlib` · `Docling`  
`sentence-transformers` · `RAGAS` · `Weights & Biases` · `FastAPI`  
`Qwen2.5-3B-Instruct` · `NLTK` · `Google Gemini Flash`

---

## License

MIT © 2026 narottaman