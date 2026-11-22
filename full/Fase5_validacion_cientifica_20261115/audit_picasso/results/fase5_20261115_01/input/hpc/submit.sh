#!/usr/bin/env bash
# Lightweight submission only: no solver or NumPy execution on the login node.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
run_id="run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
python_override=""
module_override=""
while (($#)); do
    case "$1" in
        --run-id) run_id=${2:?Missing run id}; shift 2 ;;
        --python) python_override=${2:?Missing interpreter path}; shift 2 ;;
        --module) module_override=${2:?Missing module name}; shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { printf 'Invalid run id\n' >&2; exit 2; }
[[ "$python_override" != *$'\n'* && "$module_override" != *$'\n'* ]] || exit 2
command -v sbatch >/dev/null || { printf 'sbatch is unavailable\n' >&2; exit 2; }
for item in solver.py code data configs/tasks.list hpc/smoke.sbatch hpc/array.sbatch hpc/summary.sbatch; do
    [[ -e "$project_root/$item" ]] || { printf 'Missing %s\n' "$item" >&2; exit 2; }
done
configs=()
while IFS= read -r entry || [[ -n "$entry" ]]; do
    entry=${entry%$'\r'}
    [[ -z "$entry" || "$entry" == \#* ]] && continue
    [[ "$entry" == configs/*.json && "$entry" != *'/../'* && "$entry" != *$'\t'* ]] || { printf 'Invalid config path: %s\n' "$entry" >&2; exit 2; }
    [[ -f "$project_root/$entry" ]] || { printf 'Missing config: %s\n' "$entry" >&2; exit 2; }
    configs+=("$entry")
done < "$project_root/configs/tasks.list"
((${#configs[@]} > 0)) || { printf 'Empty task list\n' >&2; exit 2; }
run_dir="$project_root/results/$run_id"
mkdir -p "$project_root/results"
mkdir "$run_dir" || { printf 'Run already exists: %s\n' "$run_dir" >&2; exit 2; }
mkdir "$run_dir/input" "$run_dir/logs"
for item in solver.py code data configs hpc; do cp -a "$project_root/$item" "$run_dir/input/"; done
if [[ -f "$project_root/analyze_campaign.py" ]]; then cp -a "$project_root/analyze_campaign.py" "$run_dir/input/"; fi
printf '%s\n' "${configs[@]}" > "$run_dir/input/configs/tasks.list"
printf '%s\n' "$run_id" > "$run_dir/run_id.txt"
printf '%s\n' "$run_id" > "$project_root/RUN_ID.txt"
# Convenience markers describe the latest submission; per-run manifests remain authoritative.
for marker in SMOKE_JOBID ARRAY_JOBID SUMMARY_JOBID; do printf '\n' > "$project_root/$marker.txt"; done
printf '%s\n' "${#configs[@]}" > "$run_dir/task_count.txt"
printf '%s\n' "$python_override" > "$run_dir/python_override.txt"
printf '%s\n' "$module_override" > "$run_dir/module_override.txt"
printf 'run_dir\t%s\nsubmitted_utc\t%s\n' "$run_dir" "$(date -u +%FT%TZ)" > "$run_dir/submission.tsv"
(cd "$run_dir/input" && find . -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$run_dir/input_SHA256SUMS.txt"
smoke_raw=$(sbatch --parsable --chdir="$run_dir/input" --output="$run_dir/logs/smoke_%j.out" --error="$run_dir/logs/smoke_%j.err" "$run_dir/input/hpc/smoke.sbatch" "$run_dir")
smoke_id=${smoke_raw%%;*}
[[ "$smoke_id" =~ ^[0-9]+$ ]] || { printf 'Invalid smoke job id: %s\n' "$smoke_raw" >&2; exit 1; }
printf 'smoke_job\t%s\n' "$smoke_id" >> "$run_dir/submission.tsv"
printf '%s\n' "$smoke_id" > "$project_root/SMOKE_JOBID.txt"
array_id=""
if array_raw=$(sbatch --parsable --array="0-$((${#configs[@]} - 1))" --dependency="afterok:$smoke_id" --kill-on-invalid-dep=yes --chdir="$run_dir/input" --output="$run_dir/logs/task_%A_%a.out" --error="$run_dir/logs/task_%A_%a.err" "$run_dir/input/hpc/array.sbatch" "$run_dir"); then
    array_id=${array_raw%%;*}
    [[ "$array_id" =~ ^[0-9]+$ ]] || { printf 'Invalid array job id\n' >&2; exit 1; }
    printf 'array_job\t%s\n' "$array_id" >> "$run_dir/submission.tsv"
    printf '%s\n' "$array_id" > "$project_root/ARRAY_JOBID.txt"
else
    printf 'array_submission_failed\ttrue\n' >> "$run_dir/submission.tsv"
fi
summary_dependency="afterany:$smoke_id"
[[ -z "$array_id" ]] || summary_dependency+=":$array_id"
summary_raw=$(sbatch --parsable --dependency="$summary_dependency" --chdir="$run_dir/input" --output="$run_dir/logs/summary_%j.out" --error="$run_dir/logs/summary_%j.err" "$run_dir/input/hpc/summary.sbatch" "$run_dir")
summary_id=${summary_raw%%;*}
[[ "$summary_id" =~ ^[0-9]+$ ]] || { printf 'Invalid summary job id\n' >&2; exit 1; }
printf 'summary_job\t%s\n' "$summary_id" >> "$run_dir/submission.tsv"
printf '%s\n' "$summary_id" > "$project_root/SUMMARY_JOBID.txt"
printf '\nRUN_ID=%s\nRUN_DIR=%s\nSMOKE_JOB=%s\nARRAY_JOB=%s\nSUMMARY_JOB=%s\n' "$run_id" "$run_dir" "$smoke_id" "$array_id" "$summary_id"
printf 'Monitor: squeue -l -j %s,%s,%s\n' "$smoke_id" "${array_id:-$smoke_id}" "$summary_id"
printf 'Results belong only to this explicit run directory.\n'
[[ -n "$array_id" ]]
