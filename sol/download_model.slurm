#!/bin/bash

#SBATCH -A grp_cbaral

#SBATCH -N 1

#SBATCH -c 8

#SBATCH -t 0-01:00:00

#SBATCH -p general

#SBATCH -q public

#SBATCH --mem=32G

#SBATCH -o logs/download_%j.out

#SBATCH -e logs/download_%j.err

#SBATCH --export=NONE



source $HOME/.venv/bin/activate



python3 -c "

from huggingface_hub import snapshot_download

print('Downloading Qwen2.5-3B-Instruct...')

snapshot_download(

    repo_id='Qwen/Qwen2.5-3B-Instruct',

    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],

)

print('Done.')

"

