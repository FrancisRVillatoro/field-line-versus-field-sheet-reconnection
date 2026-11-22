#!/bin/bash -l
#SBATCH --job-name=recon5_summary
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
if phase5_load_runtime "$run_dir"; then
    status_python=$PHASE5_RUNTIME_PYTHON
else
    # The status report uses only the standard library, even after a NumPy smoke failure.
    status_python=""
    for candidate in /usr/bin/python3 /mnt/home/soft/python/programs/x86_64_sles156/3.14.3/bin/python3; do
        if [[ -x "$candidate" ]] && "$candidate" -c 'import sys,json,pathlib; assert sys.version_info >= (3,8)'; then status_python=$candidate; break; fi
    done
    [[ -n "$status_python" ]] || { printf 'No standard-library Python for status report\n' > "$run_dir/summary_failure.txt"; exit 11; }
fi
archive_on_exit() {
    local summary_rc=$? archive_rc
    trap - EXIT
    set +e
    "$status_python" hpc/archive_results.py --run-dir "$run_dir" --summary-exit-code "$summary_rc"
    archive_rc=$?
    if ((archive_rc != 0)); then
        printf 'Archiving failed with exit code %s; the explicit run directory remains available.\n' "$archive_rc" >&2
    fi
    if ((summary_rc != 0)); then exit "$summary_rc"; fi
    exit "$archive_rc"
}
trap archive_on_exit EXIT
"$status_python" hpc/summarize_campaign.py --run-dir "$run_dir"
if [[ -f analyze_campaign.py && -f "$run_dir/smoke_ok.txt" ]]; then
    set +e
    "$status_python" code/campaign_audit.py --run-dir "$run_dir"
    analysis_rc=$?
    set -e
    printf '%s\n' "$analysis_rc" > "$run_dir/analysis_exit_code.txt"
    if ((analysis_rc != 0)); then printf 'Scientific analysis failed; campaign_status.json and partial outputs remain available.\n' >&2; fi
else
    printf 'SKIPPED: no successful smoke or no analysis entry point.\n' > "$run_dir/analysis_skipped.txt"
fi
