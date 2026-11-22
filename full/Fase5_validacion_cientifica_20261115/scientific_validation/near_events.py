import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json
from pathlib import Path
b=Path(__file__).resolve().parent;r=b.parent/'audit_picasso/results/fase5_20261115_01';src=r/'input'
sys.path[:0]=[str(src),str(src/'code')]
from solver import evolve
from audit_slices import audit
analysis=json.loads((r/'analysis.json').read_text())
for idx in [1,4,7]:
 t=next(v for v in analysis['tasks'] if v['task']==f'task_{idx:03d}')
 config=json.loads((r/f'task_{idx:03d}/config.json').read_text())
 events=[v['s_event'] for v in t['trace_analysis']['events']]
 config['slice_targets']=[0]+[x+d for x in events for d in [-.002,.002]];config['trace_slices']=False
 dest=b/'near_events'/f'e{round(config["epsilon"]*100):02d}';dest.mkdir(parents=True,exist_ok=False)
 (dest/'config.json').write_text(json.dumps(config,indent=2));evolve(config,dest)
 z=audit(dest/'evolution.npz',step_scale=.25)
 (dest/'slice_audit.json').write_text(json.dumps(z,indent=2))
 print('NEAR_EVENT_FINISHED',config['epsilon'],flush=True)
