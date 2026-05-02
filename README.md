# RAG Document Assistant

> Production-grade Retrieval-Augmented Generation pipeline over ArXiv AI papers.  
> Evaluates **3 chunking strategies × 2 index types × 2 embedding models × 3 top-k values** — 36+ experiment runs tracked via W&B Sweeps on ASU Sol HPC (A100 GPU).

[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-orange)](https://wandb.ai/ngangada-arizona-state-university/rag-document-assistant)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRE-PRODUCTION                          │
│                                                             │
│  ArXiv PDFs ──► Chunker ──► Embedding Model ──► Vector DB  │
│  (10 papers)   [5 methods]   [2 models]        [3 indexes] │
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
│                            Qwen2.5-3B-Instruct (local)      │
│                                        │                    │
│                                        ▼                    │
│                                   Final Answer              │
└─────────────────────────────────────────────────────────────┘
```

---

## Evaluation — Two Complementary Tests

### Test 1: Cross-Corpus Retrieval (W&B Sweep — 36+ runs)

Questions from an external dataset queried against ArXiv paper chunks.
Measures real-world retrieval quality — whether semantically related chunks
surface when the question is not verbatim in the corpus.

**Metric:** Context Hit Rate — fraction of questions where the relevant chunk appeared in top-k results.

| Chunking | Index | Embedding Model | top_k | Context Hit Rate | Latency (ms) |
|----------|-------|----------------|-------|-----------------|-------------|
| **paragraph** | **HNSW** | **bge-small-en-v1.5** | **10** | **0.87 🏆** | **16.3** |
| paragraph | HNSW | bge-small-en-v1.5 | 5 | 0.86 | 15.9 |
| paragraph | HNSW | all-MiniLM-L6-v2 | 10 | 0.86 | 14.0 |
| paragraph | flat | all-MiniLM-L6-v2 | 10 | 0.85 | 30.2 |
| paragraph | HNSW | bge-small-en-v1.5 | 3 | 0.81 | 15.6 |
| paragraph | flat | all-MiniLM-L6-v2 | 5 | 0.78 | 31.4 |
| fixed | HNSW | all-MiniLM-L6-v2 | 10 | 0.60 | 14.3 |
| fixed | flat | all-MiniLM-L6-v2 | 5 | 0.56 | 40.1 |
| sentence | HNSW | all-MiniLM-L6-v2 | 10 | 0.54 | 12.5 |
| sentence | flat | all-MiniLM-L6-v2 | 5 | 0.43 | 42.8 |

> **Anomaly:** bge-small-en-v1.5 + ChromaDB flat collapsed to 0.02–0.16 hit rate due to
> embedding normalization mismatch. bge + HNSW worked correctly (0.81–0.87).
> MiniLM was consistent across both index types.

---

### Test 2: Self-Retrieval Sanity Check (corpus-aligned, no API needed)

Each chunk's own text is used as a query. Verifies that the index can
retrieve a piece of text given a semantically identical query.
Ran on Sol A100 in **42 seconds** — no LLM calls, pure embedding + search.

| Chunking | Index | top_k | Hit Rate | Latency (ms) |
|----------|-------|-------|----------|-------------|
| fixed | flat | 10 | 1.000 | 12.5 |
| fixed | HNSW | 10 | 1.000 | 5.9 |
| sentence | flat | 10 | 1.000 | 11.0 |
| sentence | HNSW | 10 | 1.000 | 5.7 |
| paragraph | flat | 10 | 1.000 | 13.6 |
| paragraph | HNSW | 10 | 0.990 | 5.7 |

All indexers pass self-retrieval, confirming correct embedding and search
implementation. Hit rate of 1.0 is expected at top_k=10 on a corpus of
~1,500–2,000 chunks — this is an indexer correctness test, not a
generalization test. Cross-corpus sweep (Test 1) is the generalization measure.

**Key latency finding:** HNSW is consistently **2× faster** than flat search
(5–6ms vs 11–14ms) with equivalent or better accuracy.

---

## Key Findings

**1. Paragraph chunking dominates (+27–44pp over alternatives).**
Recursive splitting on `\n\n → \n → ". "` preserves semantic units that
align naturally with how questions are phrased. Fixed-size chunking cuts
mid-sentence; sentence chunking produces units too short for multi-sentence answers.

**2. HNSW beats flat search 2× on latency with no accuracy trade-off.**
5–6ms vs 11–14ms on self-retrieval. 13–16ms vs 30–50ms on cross-corpus sweep.
At scale (10K+ chunks) HNSW is the only viable production choice.

**3. Embedding model × index type interaction is non-obvious.**
bge-small-en-v1.5 ranks higher on MTEB benchmarks than all-MiniLM-L6-v2,
but collapses to 2% hit rate with ChromaDB flat due to normalization mismatch.
Empirical sweep caught this — benchmark scores alone would have missed it.

**4. top_k=10 is worth the context window cost.**
10–15pp improvement from k=3 to k=10, with HNSW adding only ~2ms.

**Best config:** `paragraph + HNSW + bge-small-en-v1.5 + top_k=10` → **87% cross-corpus hit rate at 16ms**

---

## Chunking Strategies

| Method | Chunk Size | Overlap | Avg Chars | Total Chunks | Strategy |
|--------|-----------|---------|-----------|-------------|---------|
| **Fixed** | 512 chars | 50 chars | 474 | 1,847 | Sliding window — last 50 chars of chunk N start chunk N+1 |
| **Sentence** | 5 sentences | 1 sentence | 639 | 1,498 | Last sentence of chunk N repeated at start of chunk N+1 |
| **Paragraph** | 512 chars | 50 chars | 445 | 1,983 | Recursive `\n\n → \n → ". "` with 50-char carry-forward |
| **Semantic** | dynamic | none | — | pending | Splits at cosine similarity drop — boundary IS the semantic break |
| **Hybrid (Docling)** | layout-aware | none | — | pending | Section/heading boundaries, table-aware |

---

## Indexing Strategies

| Index | Algorithm | Query Latency | Best For |
|-------|-----------|-------------|---------|
| **Flat** (ChromaDB) | Brute-force cosine | 11–14ms | Small corpus, exact search |
| **HNSW** (hnswlib) | Graph-based ANN | 5–6ms | Production — 2× faster |
| **FAISS-IVF** | Inverted file index | varies | Very large corpus (1M+ chunks) |

---

## Corpus

10 ArXiv AI papers parsed with `pypdf` (torch-free, no LangChain dependency):

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

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-document-assistant
cd rag-document-assistant
pip install -r requirements.txt

# 2. Set env vars
cp .env.example .env
# Add WANDB_API_KEY — no OpenAI key needed

# 3. Download ArXiv papers
python scripts/download_data.py

# 4. Ingest with all chunking methods
python scripts/run_ingest.py --method fixed
python scripts/run_ingest.py --method sentence
python scripts/run_ingest.py --method paragraph

# 5. Run self-retrieval evaluation (no API needed)
python scripts/run_eval.py --all --top-k 10

# 6. Ask a question interactively
python scripts/run_query.py --method paragraph --index hnsw

# 7. Run W&B sweep (cross-corpus retrieval experiment)
wandb sweep configs/sweep.yaml
wandb agent YOUR_ENTITY/rag-document-assistant/SWEEP_ID

# 8. Start API server
uvicorn api.main:app --reload --port 8000
```

---

## W&B Experiment Tracking

| Metric | Description |
|--------|-------------|
| `context_hit_rate` | Fraction of queries where correct chunk in top-k |
| `avg_retrieval_latency_ms` | Vector search speed per query |
| `index_build_time_sec` | Time to embed + index all chunks |
| `total_chunks` | Chunks produced by chunking method |
| `avg_chunk_chars` | Average chunk character length |

**W&B Project:** [ngangada-arizona-state-university/rag-document-assistant](https://wandb.ai/ngangada-arizona-state-university/rag-document-assistant)

---

## Sol HPC Setup

```bash
# Activate existing venv (do NOT use base conda)
source $HOME/.venv/bin/activate

# Install RAG dependencies
pip install pypdf nltk faiss-cpu hnswlib chromadb \
            sentence-transformers wandb fastapi uvicorn \
            python-dotenv pyyaml datasets google-genai

# Submit sweep job
sbatch sol/sweep_job.slurm

# Submit eval job
sbatch sol/eval_job.slurm
```

> Sol HPC note: Uses `pypdf` directly instead of LangChain's `PyPDFLoader`
> to avoid the torchvision circular import caused by Sol's mismatched
> system torch+torchvision. Sentence tokenization uses a regex fallback
> that works without SSL access (Sol login nodes block outbound SSL).

---

## Project Structure

```
rag-document-assistant/
├── data/
│   ├── raw/pdfs/                    # Downloaded ArXiv PDFs
│   ├── processed/                   # chunks_{method}.json
│   └── eval/                        # self_retrieval_results.json
├── src/
│   ├── chunkers.py                  # 5 strategies, torch-free PDF loading
│   ├── indexers.py                  # ChromaFlat, FAISS-IVF, HNSW
│   ├── generator.py                 # Qwen2.5-3B local inference
│   └── embeddings.py                # Embedding model + W&B logging
├── scripts/
│   ├── download_data.py             # Download PDFs + HF dataset
│   ├── run_ingest.py                # Chunk PDFs → JSON + W&B logging
│   ├── run_experiment.py            # Single sweep run
│   ├── run_eval.py                  # Self-retrieval eval (no API needed)
│   └── run_query.py                 # Interactive query CLI
├── api/main.py                      # FastAPI server
├── configs/
│   ├── config.yaml                  # Model + retriever config
│   └── sweep.yaml                   # W&B grid sweep
└── sol/
    ├── sweep_job.slurm              # Run W&B sweep on A100
    └── eval_job.slurm              # Run self-retrieval eval
```

---

## Tech Stack

`pypdf` · `ChromaDB` · `FAISS` · `hnswlib` · `sentence-transformers`  
`Qwen2.5-3B-Instruct` · `Weights & Biases` · `FastAPI` · `Docling`

---

## License

MIT © 2026 narottaman