#!/usr/bin/env python3
"""A bounded compute-node import/numerical probe; never installs packages."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--run-dir', required=True)
p.add_argument('--module', default='')
a = p.parse_args()
if sys.version_info < (3, 10):
    raise SystemExit('Python >=3.10 is required.')
import numpy as np
m = np.array([[2., .2, 0.], [.2, 3., .1], [0., .1, 1.]])
batch = np.stack([m, 2*m])
assert np.max(np.abs(np.einsum('...ij,...jk->...ik', batch, np.linalg.inv(batch))-np.eye(3))) < 1e-12
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'probe.npz'
    np.savez_compressed(path, matrix=batch)
    with np.load(path, allow_pickle=False) as restored:
        assert np.array_equal(restored['matrix'], batch)
run = Path(a.run_dir)
info = {
    'python_executable': os.path.abspath(sys.executable),
    'python_version': sys.version,
    'python_version_tuple': list(sys.version_info[:3]),
    'numpy_version': np.__version__,
    'numpy_file': np.__file__,
    'module': a.module,
    'platform': platform.platform(),
    'machine': platform.machine(),
    'hostname': platform.node(),
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
}
config = io.StringIO()
with contextlib.redirect_stdout(config):
    np.show_config()
(run/'numpy_build_config.txt').write_text(config.getvalue())
(run/'runtime.json').write_text(json.dumps(info, indent=2)+'\n')
(run/'runtime.python').write_text(info['python_executable']+'\n')
(run/'runtime.module').write_text(a.module+'\n')
print(json.dumps(info, indent=2))
