#!/bin/bash -l
set -euo pipefail
run_dir=/mnt/fscratch/users/tic_118_uma/frv/Reconexiones_Fase5_20261114/results/fase5_20261115_01
source "$run_dir/input/hpc/common.sh"
phase5_load_runtime "$run_dir"
phase5_thread_limits
# La contabilidad incluye los 24 JOBID individuales.
sacct -j "$(cat "$run_dir/INDIVIDUAL_JOBIDS.txt")" --format=JobID,State%30,ExitCode,Elapsed,MaxRSS > "$run/individual_accounting.txt" || true
# Conservar primero el informe y ZIP, incluso si los resultados son parciales.
set +e
bash -l "$run_dir/input/hpc/summary.sbatch" "$run_dir"
report_rc=$?
"$PHASE5_RUNTIME_PYTHON" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);bad=[]
for i in range(24):
    p=r/f'task_{i:03d}'
    try:
        w=json.loads((p/'wrapper_status.json').read_text())
        s=json.loads((p/'summary.json').read_text())
        assert w['status']=='SUCCESS' and w['exit_code']==0
        assert s['status']=='SUCCESS' and (p/'evolution.npz').is_file()
    except (OSError,ValueError,KeyError,TypeError,AssertionError):bad.append(i)
try:
    ok=int((r/'analysis_exit_code.txt').read_text())==0
    a=json.loads((r/'numerical_audit.json').read_text())
    ok=ok and a['successful_numerical_tasks']==24
except (OSError,ValueError,KeyError,TypeError):ok=False
print('Tareas ausentes o fallidas:',bad,flush=True)
print('Analisis completo:',ok,flush=True)
# Verifica ejecucion y archivos; no equivale a aceptar las predicciones fisicas.
sys.exit(0 if not bad and ok else 20)
PY
check_rc=$?
final_rc=$report_rc
if ((final_rc == 0)); then final_rc=$check_rc; fi
printf '%s\n' "$final_rc" > "$run_dir/recovery_summary_exit_code.txt"
"$PHASE5_RUNTIME_PYTHON" "$run_dir/input/hpc/archive_results.py" --run-dir "$run_dir" --summary-exit-code "$final_rc"
archive_rc=$?
if ((final_rc != 0)); then exit "$final_rc"; fi
exit "$archive_rc"
