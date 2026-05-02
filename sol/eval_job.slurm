#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
nvidia-smi

source $HOME/.venv/bin/activate
export WANDB_API_KEY="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"
export HF_HOME="/scratch/ngangada/huggingface_cache"

cd /scratch/ngangada/portfolio/rag-document-assistant
mkdir -p logs

python scripts/run_eval.py --all --top-k 10 --n-samples 100