import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b.parent/'audit_picasso/results/fase5_20261115_01/input/code'))
from audit_slices import audit
results=[]
for e in [.04,.08,.12]:
 p=b/'near_events'/f'e{round(e*100):02d}';a=json.loads((p/'slice_audit.json').read_text());coarse=audit(p/'evolution.npz',step_scale=.5)
 changes=[];errors=[]
 for lo,hi in zip(coarse['slices'],a['slices']):
  for x,y in zip(lo['cases'],hi['cases']):
   if x['status']!=y['status'] or x.get('outlet')!=y.get('outlet'):changes.append((lo['s'],x['theta_in']))
   if x['status']=='EXIT' and y['status']=='EXIT':errors.append(abs((x['theta_out']-y['theta_out']+3.141592653589793)%(2*3.141592653589793)-3.141592653589793))
 results.append({'epsilon':e,'classification_changes':changes,'max_endpoint_step_difference':max(errors),'exits_coarse':sum(x['traced_exits'] for x in coarse['slices']),'exits_fine':sum(x['traced_exits'] for x in a['slices'])})
print(results);(b/'tracing_refinement.json').write_text(json.dumps(results,indent=2))
