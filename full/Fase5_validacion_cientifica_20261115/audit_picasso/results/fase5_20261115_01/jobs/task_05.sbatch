#!/bin/bash -l
set -euo pipefail
run_dir=/mnt/fscratch/users/tic_118_uma/frv/Reconexiones_Fase5_20261114/results/fase5_20261115_01
index=5
trap 'rc=$?; printf "%s\n" "$rc" > "$run_dir/logs/task_${index}_${SLURM_JOB_ID:-local}.shell_exit"' EXIT
printf 'START job=%s index=%s host=%s\n' "${SLURM_JOB_ID:-local}" "$index" "$(hostname)"
source "$run_dir/input/hpc/common.sh"
phase5_load_runtime "$run_dir"
phase5_thread_limits
cd "$run_dir/input"
"$PHASE5_RUNTIME_PYTHON" hpc/run_task.py --run-dir "$run_dir" --index "$index"
