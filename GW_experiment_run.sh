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

echo "$(date)"
echo "HOST: $(hostname)"
echo "WORKDIR: $SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader


source ~/.bashrc
conda activate jaxpsmc-gpu


module purge 2>/dev/null || true
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$(python - <<'PY'
import site, glob, os
sp = site.getsitepackages()[0]
paths = glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
print(":".join(paths))
PY
):${LD_LIBRARY_PATH:-}"


export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$SLURM_SUBMIT_DIR"


python GW_experiment.py

echo "JOB DONE"