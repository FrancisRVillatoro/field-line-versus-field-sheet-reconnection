"""Independent shooting check of bundled initial data; LOCAL SciPy preprocessing only."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];z=np.load(ROOT/'data/radial_profiles.npz');reports=[]
for j,e in enumerate(z['epsilon']):
 if e==0:continue
 def source(r,l):return e*e*r*r+np.exp(-2*l)*(e*e*(1-r*r)**2+e**4*(1-r*r)**4/4)
 def shoot(l0,end=2.,dense=False):
  r0=1e-7;A=source(0,l0)
  return solve_ivp(lambda r,v:[v[1],-v[1]/r-source(r,v[0])],(r0,end),[l0-A*r0*r0/4,-A*r0/2],method='DOP853',rtol=3e-14,atol=2e-15,dense_output=dense)
 l0=brentq(lambda l:shoot(l).y[0,-1],0,4*e*e,xtol=1e-15,rtol=1e-14)
 sol=shoot(l0,3.2,True);r=z['r'];v=sol.sol(np.maximum(r,1e-7));v[:,0]=[l0,0]
 reports.append({'epsilon':float(e),'shooting_lambda0':l0,'bvp_lambda0':float(z['lam'][j,0]),'lambda0_difference':float(abs(l0-z['lam'][j,0])),
 'lambda_difference_r_le_2':float(np.max(abs(v[0,r<=2]-z['lam'][j,r<=2]))),'lambda_difference_full_extension':float(np.max(abs(v[0]-z['lam'][j]))),'lambda_r_difference_full_extension':float(np.max(abs(v[1]-z['lam_r'][j])))})
(ROOT/'data/initial_shooting_audit.json').write_text(json.dumps({'scope':'Independent numerical agreement, not an interval-arithmetic error bound.','results':reports},indent=2)+'\n');print(json.dumps(reports,indent=2))
