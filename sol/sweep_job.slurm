#!/bin/bash

#SBATCH -A grp_cbaral

#SBATCH -N 1

#SBATCH -c 16

#SBATCH -t 0-06:00:00

#SBATCH -p public

#SBATCH -q public

#SBATCH --mem=64G

#SBATCH --gres=gpu:1

#SBATCH -o logs/sweep_%j.out

#SBATCH -e logs/sweep_%j.err

#SBATCH --mail-type=ALL

#SBATCH --mail-user=ngangada@asu.edu

#SBATCH --export=NONE

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"

nvidia-smi

source $HOME/.venv/bin/activate

export WANDB_API_KEY="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"

export HF_HOME="/scratch/ngangada/huggingface_cache"   # use existing cache

cd /scratch/ngangada/portfolio/rag-document-assistant

wandb agent ngangada-arizona-state-university/rag-document-assistant/orkg3mdt

