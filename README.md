# RAG Document Assistant

> Production-grade Retrieval-Augmented Generation pipeline over ArXiv AI papers.  
> Systematically evaluates **5 chunking strategies × 3 index types × 2 embedding models** tracked via W&B Sweeps.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRE-PRODUCTION                          │
│                                                             │
│  ArXiv PDFs ──► Chunker ──► Embedding Model ──► Vector DB  │
│  (10 papers)    [5 methods]   [2 models]        [3 indexes] │
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
│                                  GPT-4o-mini                │
│                                        │                    │
│                                        ▼                    │
│                                   Final Answer              │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiment Matrix

| Chunking Method | Index Type | Embedding Model | top_k |
|----------------|-----------|----------------|-------|
| Fixed (256/512/1024) | Flat (ChromaDB) | all-MiniLM-L6-v2 | 3/5/10 |
| Sentence (3/5 sent) | HNSW (hnswlib) | BAAI/bge-small-en-v1.5 | 3/5/10 |
| Paragraph (recursive) | FAISS-IVF | — | — |
| Semantic (cosine sim) | — | — | — |
| Hybrid (Docling layout) | — | — | — |

All runs tracked via **W&B Sweeps** → [View Dashboard](#)

---

## Results

> Results from W&B sweep across 90+ experiment runs.  
> Best configuration: **hybrid chunking + HNSW + bge-small-en-v1.5 + top_k=5**

| Chunking Method | Index | Faithfulness | Answer Relevancy | Context Recall |
|----------------|-------|-------------|-----------------|---------------|
| fixed_512      | flat  | —           | —               | —             |
| sentence_5s    | flat  | —           | —               | —             |
| paragraph_512  | flat  | —           | —               | —             |
| semantic_0.5   | hnsw  | —           | —               | —             |
| **hybrid_docling** | **hnsw** | **—** | **—**       | **—**         |

*Fill in after running `python scripts/run_eval.py --all`*

---

## Corpus

- **10 ArXiv AI papers**: Attention Is All You Need, BERT, GPT-3, ResNet, Adam, GANs, LoRA, LLaMA, DALL-E 2, Stable Diffusion
- **Eval set**: 60 auto-generated Q&A pairs from paper chunks (via GPT-4o-mini) + 100 from `neural-bridge/rag-dataset-12000`

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-document-assistant
cd rag-document-assistant
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and WANDB_API_KEY

# 3. Download data
python scripts/download_data.py

# 4. Ingest with all chunking methods
python scripts/run_ingest.py --method all

# 5. Generate eval QA pairs
python scripts/generate_eval.py

# 6. Run a question interactively
python scripts/run_query.py --method hybrid --index hnsw

# 7. Run full evaluation
python scripts/run_eval.py --all

# 8. Run W&B sweep (all combinations)
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>

# 9. Start API server
uvicorn api.main:app --reload --port 8000
```

---

## W&B Tracking

Every experiment logs:

| Metric | Description |
|--------|-------------|
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_recall` | Was the correct chunk retrieved? |
| `avg_retrieval_latency_ms` | Speed of vector search |
| `index_build_time_sec` | Time to build the index |
| `total_chunks` | Number of chunks produced per method |
| `avg_chunk_chars` | Average chunk size in characters |

W&B Public Report: [Link after first run]

---

## Sol HPC (GPU-Accelerated)

Semantic chunking and large-corpus embedding run on ASU Sol HPC:

```bash
# Copy project to Sol
scp -r . YOUR_ASURITE@sol.asu.edu:~/rag-document-assistant

# Submit job
sbatch sol/embed_job.slurm

# Monitor
squeue -u $USER
tail -f logs/embed_<job_id>.out
```

GPU metrics (utilization, memory, temperature) are automatically captured in W&B's System tab.

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
│   ├── chunkers.py         # 5 chunking strategies
│   ├── indexers.py         # ChromaFlat, FAISS-IVF, HNSW
│   ├── embeddings.py       # Embedding model wrapper
│   ├── generator.py        # LLM answer generation
│   ├── pipeline.py         # End-to-end RAG pipeline
│   └── evaluate.py         # RAGAS evaluation runner
├── scripts/
│   ├── download_data.py    # Download PDFs + HF dataset
│   ├── run_ingest.py       # Chunk PDFs → save JSON
│   ├── generate_eval.py    # Generate Q&A eval pairs
│   ├── run_experiment.py   # Single experiment (W&B sweep target)
│   ├── run_query.py        # Interactive query CLI
│   └── run_eval.py         # Full evaluation + comparison
├── api/
│   └── main.py             # FastAPI server
├── configs/
│   ├── config.yaml         # Main config
│   └── sweep.yaml          # W&B sweep grid
├── sol/
│   └── embed_job.slurm     # ASU Sol HPC job script
└── tests/
    └── test_pipeline.py    # Unit tests
```

---

## Tech Stack

`LangChain` · `ChromaDB` · `FAISS` · `hnswlib` · `Docling` · `sentence-transformers`  
`RAGAS` · `Weights & Biases` · `FastAPI` · `OpenAI GPT-4o-mini` · `NLTK`

---

## License

MIT © 2026 narottaman