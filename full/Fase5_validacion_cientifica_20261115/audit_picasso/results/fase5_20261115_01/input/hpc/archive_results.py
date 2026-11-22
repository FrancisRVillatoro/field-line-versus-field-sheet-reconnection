#!/usr/bin/env python3
"""Archive one explicit run, including partial results; update a predictable alias."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import zipfile


PREFIX = 'Reconexiones_Fase5_resultados_20261114'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--summary-exit-code', type=int, default=0)
    a = p.parse_args()
    run = Path(a.run_dir).resolve()
    run_id = (run/'run_id.txt').read_text().strip()
    if run.name != run_id or run.parent.name != 'results':
        raise ValueError('The explicit run directory does not match its recorded run id.')
    root = run.parent.parent
    status = {}
    try:
        status = json.loads((run/'campaign_status.json').read_text())
        if not isinstance(status, dict):
            status = {}
    except (OSError, ValueError):
        pass
    try:
        analysis_rc = int((run/'analysis_exit_code.txt').read_text().strip())
    except (OSError, ValueError):
        analysis_rc = None
    state = 'COMPLETE_PROCESS_CAMPAIGN' if status.get('complete_process_campaign') else 'PARTIAL_OR_FAILED'
    archive = root/(PREFIX+'_'+run_id+'.zip')
    alias = root/(PREFIX+'.zip')
    metadata = {
        'run_id': run_id,
        'snapshot_utc': datetime.now(timezone.utc).isoformat(),
        'state': state,
        'expected_tasks': status.get('expected_tasks'),
        'successful_processes': status.get('successful_processes'),
        'readable_summaries': status.get('readable_summaries'),
        'smoke_ok': status.get('smoke_ok', False),
        'summary_job_exit_code': a.summary_exit_code,
        'scientific_analysis_exit_code': analysis_rc,
        'archive_filename': archive.name,
        'predictable_alias': alias.name,
        'scope': 'Only results/'+run_id+'/. Process completion is not a scientific acceptance verdict.',
        'logs': 'Copied at the end of the summary job; final scheduler accounting or later log bytes may not yet exist.',
    }
    (run/'archive_status.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (run/'archive_status.txt').write_text('RUN_ID='+run_id+'\nSTATE='+state+'\nARCHIVE='+archive.name+'\nScientific acceptance requires the separate diagnostic checks.\n')
    temporary = archive.with_name(archive.name+'.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            kind = zipfile.ZIP_STORED if path.suffix.lower() in ('.npz', '.zip', '.png', '.pdf') else zipfile.ZIP_DEFLATED
            z.write(path, arcname=str(path.relative_to(root)), compress_type=kind)
    temporary.replace(archive)
    temporary_alias = alias.with_name(alias.name+'.tmp.'+run_id)
    if temporary_alias.is_symlink() or temporary_alias.exists():
        temporary_alias.unlink()
    os.symlink(archive.name, temporary_alias)
    os.replace(temporary_alias, alias)
    print('Archived run:', run_id)
    print('Archive state:', state)
    print('Archive:', archive)
    print('Predictable alias:', alias)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
