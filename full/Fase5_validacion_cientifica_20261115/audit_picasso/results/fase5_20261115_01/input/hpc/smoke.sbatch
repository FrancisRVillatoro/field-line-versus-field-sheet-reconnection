#!/bin/bash -l
#SBATCH --job-name=recon5_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_parti
set -euo pipefail
run_dir=${1:?Run directory required}
source "$run_dir/input/hpc/common.sh"
phase5_thread_limits
cd "$run_dir/input"
try_python() {
    local candidate=$1 selected_module=${2:-}
    [[ -x "$candidate" ]] || return 1
    printf 'Probing %s (module=%s)\n' "$candidate" "$selected_module"
    "$candidate" hpc/probe_python.py --run-dir "$run_dir" --module "$selected_module"
}
selected=false
python_override=$(cat "$run_dir/python_override.txt")
module_override=$(cat "$run_dir/module_override.txt")
if [[ -n "$python_override" ]]; then
    if [[ -n "$module_override" ]]; then module load "$module_override"; fi
    try_python "$python_override" "$module_override" && selected=true
elif [[ -n "$module_override" ]]; then
    module load "$module_override"
    try_python "$(command -v python3)" "$module_override" && selected=true
else
    # Module names are listed by the official SCBI catalogue; imports decide availability.
    if type module >/dev/null 2>&1; then
        module -t avail python > "$run_dir/logs/python_modules.txt" 2>&1 || true
        for mod in python/3.11.4 python/3.14.3; do
            if (module load "$mod" && try_python "$(command -v python3)" "$mod"); then selected=true; break; fi
        done
    fi
    if [[ "$selected" != true ]]; then
        candidates=(/usr/bin/python3 /mnt/home/soft/python/programs/x86_64_sles156/3.14.3/bin/python3)
        current_python=$(command -v python3 || true)
        [[ -z "$current_python" ]] || candidates+=("$current_python")
        for candidate in "${candidates[@]}"; do
            if try_python "$candidate"; then selected=true; break; fi
        done
    fi
fi
if [[ "$selected" != true ]]; then
    printf 'No tested interpreter has the required Python/NumPy. No installation was attempted.\n' >&2
    printf 'NO_COMPATIBLE_NUMPY\n' > "$run_dir/smoke_failed.txt"
    exit 10
fi
phase5_load_runtime "$run_dir"
"$PHASE5_RUNTIME_PYTHON" hpc/run_task.py --run-dir "$run_dir" --self-test
printf 'SMOKE_OK\n' > "$run_dir/smoke_ok.txt"
