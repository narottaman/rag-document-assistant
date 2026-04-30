#!/bin/bash
#SBATCH --job-name=rag-embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_ASU_EMAIL@asu.edu

# ─────────────────────────────────────────────────────────────────────────────
# RAG Document Assistant — GPU Embedding + Semantic Chunking Job
# Submit: sbatch sol/embed_job.slurm
# Check:  squeue -u $USER
# Logs:   tail -f logs/embed_<job_id>.out
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU(s):    $SLURM_JOB_GPUS"
echo "Start:     $(date)"
echo "=========================================="

# Load ASU Sol modules
module load python/3.11
module load cuda/12.1

# Activate virtual environment (create it first time with setup below)
source ~/envs/rag/bin/activate

# Set API keys from your Sol .bashrc or pass them here
export WANDB_API_KEY="${WANDB_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

# Log GPU info to stdout (captured by W&B system tab automatically)
nvidia-smi
echo ""

# Move to project directory
cd ~/rag-document-assistant

# ─── Step 1: Fixed + Paragraph chunking (CPU, fast) ──────────────────────────
echo "--- Step 1: Fixed chunking ---"
python scripts/run_ingest.py --method fixed --chunk-size 256
python scripts/run_ingest.py --method fixed --chunk-size 512
python scripts/run_ingest.py --method fixed --chunk-size 1024

echo "--- Step 2: Paragraph chunking ---"
python scripts/run_ingest.py --method paragraph --chunk-size 512
python scripts/run_ingest.py --method paragraph --chunk-size 1024

echo "--- Step 3: Sentence chunking ---"
python scripts/run_ingest.py --method sentence --sentences-per-chunk 3
python scripts/run_ingest.py --method sentence --sentences-per-chunk 5

# ─── Step 4: Semantic chunking (GPU-accelerated) ─────────────────────────────
echo "--- Step 4: Semantic chunking (GPU) ---"
# sentence-transformers uses GPU automatically via PyTorch
python scripts/run_ingest.py --method semantic --threshold 0.4
python scripts/run_ingest.py --method semantic --threshold 0.5
python scripts/run_ingest.py --method semantic --threshold 0.6

# ─── Step 5: Hybrid Docling chunking ─────────────────────────────────────────
echo "--- Step 5: Hybrid Docling chunking ---"
python scripts/run_ingest.py --method hybrid

# ─── Step 6: Generate eval QA pairs ─────────────────────────────────────────
echo "--- Step 6: Generate eval QA pairs ---"
python scripts/generate_eval.py \
    --chunks-path data/processed/chunks_hybrid.json \
    --n-questions 60

echo ""
echo "=========================================="
echo "All ingestion complete."
echo "End: $(date)"
echo "=========================================="

# ─────────────────────────────────────────────────────────────────────────────
# FIRST-TIME SETUP ON SOL (run these manually before submitting the job):
#
# module load python/3.11
# module load cuda/12.1
# python -m venv ~/envs/rag
# source ~/envs/rag/bin/activate
# pip install --upgrade pip
# pip install -r requirements.txt
# pip install faiss-gpu hnswlib docling
# pip install wandb
# wandb login   # paste your W&B API key
#
# Then copy your project to Sol:
# scp -r ./rag-document-assistant YOUR_ASURITE@sol.asu.edu:~/
#
# Then submit:
# sbatch sol/embed_job.slurm
# ─────────────────────────────────────────────────────────────────────────────
