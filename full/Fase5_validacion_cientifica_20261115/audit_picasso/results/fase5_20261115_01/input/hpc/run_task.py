#!/usr/bin/env python3
"""Run one frozen campaign task and preserve execution status on ordinary failures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--index', type=int)
    g.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    run = Path(a.run_dir).resolve()
    src = run/'input'
    entries = (src/'configs/tasks.list').read_text().splitlines()
    task = 'smoke' if a.self_test else 'task_{:03d}'.format(a.index)
    out = run/task
    out.mkdir(exist_ok=False)
    record = {'task': task, 'started_utc': now(), 'status': 'STARTED', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'), 'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')}
    status_path = out/'wrapper_status.json'
    write_json(status_path, record)
    start = time.monotonic()
    rc = 125
    try:
        import numpy as np
        expected = json.loads((run/'runtime.json').read_text())
        if list(sys.version_info[:3]) != expected['python_version_tuple'] or np.__version__ != expected['numpy_version']:
            raise RuntimeError('Runtime differs from successful smoke probe.')
        cmd = [sys.executable, str(src/'solver.py')]
        if a.self_test:
            cmd += ['--self-test']
        else:
            if a.index < 0 or a.index >= len(entries):
                raise ValueError('Task index outside the frozen task list.')
            config = (src/entries[a.index]).resolve()
            config.relative_to(src)
            record['config'] = entries[a.index]
            record['config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
            cmd += ['--config', str(config)]
        cmd += ['--output', str(out)]
        record['command'] = cmd
        write_json(status_path, record)
        print('Running:', ' '.join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=src, check=False).returncode
        record['status'] = 'SUCCESS' if rc == 0 else 'FAILED'
    except BaseException as exc:
        record['status'] = 'FAILED'
        record['error'] = '{}: {}'.format(type(exc).__name__, exc)
        print(record['error'], file=sys.stderr, flush=True)
    finally:
        record['exit_code'] = rc
        record['finished_utc'] = now()
        record['wall_seconds'] = time.monotonic()-start
        write_json(status_path, record)
    return rc if rc >= 0 else 128-rc


if __name__ == '__main__':
    raise SystemExit(main())
