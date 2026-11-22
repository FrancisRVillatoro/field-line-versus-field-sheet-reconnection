"""Reproduce supplementary controls locally; needs Python, NumPy and SciPy.
Existing completed outputs are retained. To recompute, move extra/ and near_events/
out of this directory first. This script does not connect to Picasso.
"""
import subprocess,sys,json
from pathlib import Path
B=Path(__file__).resolve().parent
jobs=[]
for e in [.04,.08,.12]:
 tag=f'e{round(e*100):02d}'
 for kind,N,cfl,suffix in [('bvp',65,.075,'N65_dt2'),('bvp',129,.075,'N129_dt2'),('bvp',129,.30,'N129_dt_coarse'),('shoot',129,.075,'shoot_direct')]:jobs.append((kind,e,N,cfl,tag+'_'+suffix))
for e in [.01,.02]:
 for N,cfl in [(129,.075),(97,.075),(129,.15)]:jobs.append(('small',e,N,cfl,f'e{round(e*100):02d}_small_N{N}_cfl{cfl}'))
for j in jobs:
 p=B/'extra'/j[-1]/'summary.json'
 if p.exists():
  assert json.loads(p.read_text())['status']=='SUCCESS'
  continue
 subprocess.run([sys.executable,str(B/'worker.py'),*map(str,j)],check=True)
if not (B/'near_events').exists():subprocess.run([sys.executable,str(B/'near_events.py')],check=True)
for script in ['evaluate.py','evaluate_small.py','retrace.py']:subprocess.run([sys.executable,str(B/script)],check=True)
print('Analysis regenerated; empirical validation is distinct from a certified error bound.')
