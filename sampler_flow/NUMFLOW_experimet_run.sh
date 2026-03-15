#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_a100
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --mem-per-gpu=20G
#SBATCH --job-name=NUM_experiment
#SBATCH --output=log.out
#SBATCH --error=log.err

set -euo pipefail

echo "$(date)"
echo "HOST: $(hostname)"
echo "WORKDIR: ${SLURM_SUBMIT_DIR:-$(pwd)}"

source ~/.bashrc
conda activate jaxpsmc-gpu

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$SLURM_SUBMIT_DIR"

# repo root = submit dir OR parent dir (if submitted from numerical_experiments/)
if [ -d "$SLURM_SUBMIT_DIR/sampler" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(realpath "$SLURM_SUBMIT_DIR/..")"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -u "$ROOT/sampler_flow/NUMFLOWnumerical_experiments.py"
