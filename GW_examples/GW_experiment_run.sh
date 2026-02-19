#!/bin/bash -l
#SBATCH -J GW_experiment
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --mem-per-gpu=20G
#SBATCH -t 01:00:00
#SBATCH --output=GW_log.out
#SBATCH --error=GW_log.err
#SBATCH --export=NONE

set -euo pipefail

source ~/.bashrc
conda activate jaxpsmc-gpu

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$SLURM_SUBMIT_DIR"

if [ -d "$SLURM_SUBMIT_DIR/sampler" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(realpath "$SLURM_SUBMIT_DIR/..")"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -u "$ROOT/GW_examples/GW_experiment.py"