#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "=========================================="

nvidia-smi

source $HOME/.venv/bin/activate

export WANDB_API_KEY="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"
#export GOOGLE_API_KEY="your_google_key_here"
export HF_HOME="/scratch/ngangada/huggingface_cache"
#export OPENAI_API_KEY="your_openai_key_here"

cd /scratch/ngangada/portfolio/rag-document-assistant

mkdir -p logs

echo "--- Running fixed + HNSW + top_k=10 ---"
python scripts/run_eval.py \
    --method fixed \
    --index hnsw \
    --top-k 10 \
    --qa-path data/eval/hf_qa_subset.json

echo "--- Running paragraph + HNSW + top_k=10 ---"
python scripts/run_eval.py \
    --method paragraph \
    --index hnsw \
    --top-k 10 \
    --qa-path data/eval/hf_qa_subset.json

echo "--- Running sentence + HNSW + top_k=10 ---"
python scripts/run_eval.py \
    --method sentence \
    --index hnsw \
    --top-k 10 \
    --qa-path data/eval/hf_qa_subset.json

echo ""
echo "=========================================="
echo "All eval complete."
echo "End: $(date)"
echo "=========================================="