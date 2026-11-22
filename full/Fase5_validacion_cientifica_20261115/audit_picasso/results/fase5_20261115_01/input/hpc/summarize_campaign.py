#!/usr/bin/env python3
"""Standard-library status report that remains useful after failed/partial runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def read_json(path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object.')
        return value, None
    except (OSError, ValueError) as exc:
        return None, '{}: {}'.format(type(exc).__name__, exc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    a = p.parse_args()
    run = Path(a.run_dir).resolve()
    entries = (run/'input/configs/tasks.list').read_text().splitlines()
    submission = {}
    for line in (run/'submission.tsv').read_text().splitlines():
        key, value = line.split('\t', 1)
        submission[key] = value
    report = {'run_id': (run/'run_id.txt').read_text().strip(), 'run_dir': str(run), 'generated_utc': datetime.now(timezone.utc).isoformat(), 'submission': submission, 'smoke_ok': (run/'smoke_ok.txt').is_file(), 'tasks': []}
    for index, config in enumerate(entries):
        task = 'task_{:03d}'.format(index)
        path = run/task
        status, status_error = read_json(path/'wrapper_status.json')
        result, result_error = read_json(path/'summary.json')
        report['tasks'].append({'index': index, 'task': task, 'config': config, 'wrapper': status, 'wrapper_read_error': status_error, 'summary': result, 'summary_read_error': result_error, 'evolution_present': (path/'evolution.npz').is_file()})
    report['successful_processes'] = sum(x['wrapper'] is not None and x['wrapper'].get('status') == 'SUCCESS' for x in report['tasks'])
    report['readable_summaries'] = sum(x['summary'] is not None for x in report['tasks'])
    report['expected_tasks'] = len(entries)
    report['complete_process_campaign'] = report['smoke_ok'] and report['successful_processes'] == len(entries)
    job_ids = [submission[k] for k in ('smoke_job', 'array_job') if k in submission]
    if job_ids:
        try:
            r = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(job_ids), '--format=JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS'], text=True, capture_output=True, timeout=30, check=False)
            (run/'slurm_accounting.txt').write_text(r.stdout)
            report['sacct_exit_code'] = r.returncode
            if r.stderr:
                (run/'slurm_accounting_error.txt').write_text(r.stderr)
        except (OSError, subprocess.SubprocessError) as exc:
            report['sacct_error'] = str(exc)
    (run/'campaign_status.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = [
        'RUN_ID='+report['run_id'],
        'Expected tasks: {}'.format(len(entries)),
        'Successful solver processes: {}'.format(report['successful_processes']),
        'Readable scientific summaries: {}'.format(report['readable_summaries']),
        'Smoke succeeded: {}'.format(report['smoke_ok']),
        'Process completion is not a scientific convergence or acceptance verdict.',
        'Partial outputs are retained. Inspect campaign_status.json and slurm_accounting.txt.',
    ]
    (run/'campaign_status.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
