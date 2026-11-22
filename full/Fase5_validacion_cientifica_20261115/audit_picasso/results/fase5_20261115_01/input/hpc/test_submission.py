#!/usr/bin/env python3
"""Local tests with a mocked sbatch: no Slurm or cluster connection."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent


class PackagingTest(unittest.TestCase):
    def test_dependency_chain_and_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'campaign with spaces'
            root.mkdir()
            shutil.copytree(HERE, root/'hpc', ignore=shutil.ignore_patterns('__pycache__'))
            (root/'code').mkdir()
            (root/'data').mkdir()
            (root/'configs').mkdir()
            (root/'solver.py').write_text('raise SystemExit(0)\n')
            for n in range(3):
                (root/'configs'/('case_%d.json' % n)).write_text('{}\n')
            (root/'configs/tasks.list').write_text('# comment\nconfigs/case_0.json\n\nconfigs/case_1.json\nconfigs/case_2.json\n')
            fakebin = Path(td)/'bin'
            fakebin.mkdir()
            stub = fakebin/'sbatch'
            stub.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ["MOCK_CALLS"])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a))\nif os.environ.get("MOCK_FAIL_ARRAY") and any(x.startswith("--array=") for x in sys.argv): raise SystemExit(1)\nprint(1000+len(a))\n')
            stub.chmod(0o755)
            callsfile = Path(td)/'calls.json'
            env = dict(os.environ, PATH=str(fakebin)+os.pathsep+os.environ['PATH'], MOCK_CALLS=str(callsfile))
            r = subprocess.run(['bash', str(root/'hpc/submit.sh'), '--run-id', 'fixed_test'], env=env, text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            calls = json.loads(callsfile.read_text())
            self.assertEqual(len(calls), 3)
            self.assertEqual((root/'RUN_ID.txt').read_text().strip(), 'fixed_test')
            self.assertEqual((root/'SMOKE_JOBID.txt').read_text().strip(), '1001')
            self.assertEqual((root/'ARRAY_JOBID.txt').read_text().strip(), '1002')
            self.assertEqual((root/'SUMMARY_JOBID.txt').read_text().strip(), '1003')
            self.assertIn('squeue -l -j 1001,1002,1003', r.stdout)
            for filename in ('smoke.sbatch', 'array.sbatch', 'summary.sbatch'):
                batch = (root/'hpc'/filename).read_text()
                self.assertIn('#SBATCH --partition=cpu_parti', batch)
                self.assertNotIn('--constraint', batch)
                self.assertIn('#SBATCH --cpus-per-task=1', batch)
            self.assertIn('#SBATCH --time=00:30:00', (root/'hpc/array.sbatch').read_text())
            self.assertIn('--array=0-2', calls[1])
            self.assertIn('--dependency=afterok:1001', calls[1])
            self.assertIn('--kill-on-invalid-dep=yes', calls[1])
            self.assertIn('--dependency=afterany:1001:1002', calls[2])
            self.assertFalse(any('%' in x for x in calls[1] if x.startswith('--array=')))
            run = root/'results/fixed_test'
            self.assertTrue((run/'input_SHA256SUMS.txt').is_file())
            self.assertEqual((run/'task_count.txt').read_text().strip(), '3')
            # Mutating a source config does not change the submitted snapshot.
            (root/'configs/case_0.json').write_text('{"changed": true}\n')
            self.assertEqual((run/'input/configs/case_0.json').read_text(), '{}\n')
            # A duplicate run must fail without submitting additional jobs.
            again = subprocess.run(['bash', str(root/'hpc/submit.sh'), '--run-id', 'fixed_test'], env=env, text=True, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(len(json.loads(callsfile.read_text())), 3)
            # Entirely absent task outputs still produce a partial status report.
            report = subprocess.run([sys.executable, str(run/'input/hpc/summarize_campaign.py'), '--run-dir', str(run)], env=env, text=True, capture_output=True)
            self.assertEqual(report.returncode, 0, report.stderr)
            status = json.loads((run/'campaign_status.json').read_text())
            self.assertEqual(status['expected_tasks'], 3)
            self.assertEqual(status['successful_processes'], 0)
            self.assertFalse(status['complete_process_campaign'])
            # Failed array submission still queues a summary after the smoke job.
            failure = subprocess.run(['bash', str(root/'hpc/submit.sh'), '--run-id', 'array_failure'], env=dict(env, MOCK_FAIL_ARRAY='1'), text=True, capture_output=True)
            self.assertNotEqual(failure.returncode, 0)
            calls = json.loads(callsfile.read_text())
            self.assertEqual(len(calls), 6)
            self.assertIn('--dependency=afterany:1004', calls[5])
            self.assertIn('array_submission_failed\ttrue', (root/'results/array_failure/submission.tsv').read_text())
            self.assertEqual((root/'RUN_ID.txt').read_text().strip(), 'array_failure')
            self.assertEqual((root/'SMOKE_JOBID.txt').read_text().strip(), '1004')
            self.assertEqual((root/'ARRAY_JOBID.txt').read_text().strip(), '')
            self.assertEqual((root/'SUMMARY_JOBID.txt').read_text().strip(), '1006')
            # A damaged scientific result and an interrupted task remain reportable.
            (run/'task_000').mkdir()
            (run/'task_000/wrapper_status.json').write_text('{"status":"SUCCESS"}')
            (run/'task_000/summary.json').write_text('[]')
            (run/'task_001').mkdir()
            (run/'task_001/wrapper_status.json').write_text('{"status":"STARTED"}')
            report = subprocess.run([sys.executable, str(run/'input/hpc/summarize_campaign.py'), '--run-dir', str(run)], env=env, text=True, capture_output=True)
            self.assertEqual(report.returncode, 0, report.stderr)
            status = json.loads((run/'campaign_status.json').read_text())
            self.assertEqual(status['successful_processes'], 1)
            self.assertEqual(status['readable_summaries'], 0)
            self.assertIn('Expected a JSON object', status['tasks'][0]['summary_read_error'])
            self.assertFalse(status['complete_process_campaign'])
            # The archive contains only the exact run and labels incomplete data.
            packed = subprocess.run([sys.executable, str(run/'input/hpc/archive_results.py'), '--run-dir', str(run)], text=True, capture_output=True)
            self.assertEqual(packed.returncode, 0, packed.stderr)
            archive = root/'Reconexiones_Fase5_resultados_20261114_fixed_test.zip'
            alias = root/'Reconexiones_Fase5_resultados_20261114.zip'
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), archive.resolve())
            with zipfile.ZipFile(archive) as z:
                self.assertTrue(all(name.startswith('results/fixed_test/') for name in z.namelist()))
                self.assertEqual(z.testzip(), None)
                archived_status = json.loads(z.read('results/fixed_test/archive_status.json'))
                self.assertEqual(archived_status['state'], 'PARTIAL_OR_FAILED')


if __name__ == '__main__':
    unittest.main()
