#!/bin/bash -l
#SBATCH --job-name=recon5_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_parti
set -euo pipefail
run_dir=${1:?Run directory required}
source "$run_dir/input/hpc/common.sh"
phase5_thread_limits
phase5_load_runtime "$run_dir"
cd "$run_dir/input"
exec "$PHASE5_RUNTIME_PYTHON" hpc/run_task.py --run-dir "$run_dir" --index "${SLURM_ARRAY_TASK_ID:?}"
