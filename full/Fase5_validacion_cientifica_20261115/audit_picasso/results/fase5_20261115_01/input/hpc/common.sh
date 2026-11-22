#!/usr/bin/env bash
# Shared helpers, sourced only inside allocated batch jobs.
phase5_thread_limits() {
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
}
phase5_load_runtime() {
    local run_dir=$1 selected_module
    [[ -s "$run_dir/runtime.python" ]] || return 1
    selected_module=$(cat "$run_dir/runtime.module")
    if [[ -n "$selected_module" ]]; then
        type module >/dev/null 2>&1 || return 1
        module load "$selected_module" || return 1
    fi
    PHASE5_RUNTIME_PYTHON=$(cat "$run_dir/runtime.python")
    [[ -x "$PHASE5_RUNTIME_PYTHON" ]] || return 1
    export PHASE5_RUNTIME_PYTHON
}
