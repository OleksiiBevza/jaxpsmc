#!/bin/bash -l
#SBATCH -J GW_experiment
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --mem-per-gpu=40G
#SBATCH -t 20:00:00
#SBATCH --output=GW_log.out
#SBATCH --error=GW_log.err
#SBATCH --export=NONE

set -euo pipefail

# Load your environment
source ~/.bashrc
conda activate jaxpsmc-gpu

# Go to submit directory
cd "$SLURM_SUBMIT_DIR"

# Detect project root 
if [ -d "$SLURM_SUBMIT_DIR/sampler" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(realpath "$SLURM_SUBMIT_DIR/..")"
fi

# Python can find your local packages 
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Run 
python -u "$ROOT/GW_examples/GW_experiment.py"







