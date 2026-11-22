import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
BASE=Path(__file__).resolve().parent
SRC=BASE.parent/'audit_picasso/results/fase5_20261115_01/input'
sys.path.insert(0,str(SRC));import solver
kind=sys.argv[1];e=float(sys.argv[2]);N=int(sys.argv[3]);cfl=float(sys.argv[4]);tag=sys.argv[5]
config={'L':2.,'cfl':cfl,'Tmax':.26,'order':6,'nb':256,'slice_targets':[0,.05,.1,.15,.2,.24],'diagnostic_radius':1.1,'trace_slices':False,'epsilon':e,'N':N,'tag':tag}
out=BASE/'extra'/tag;out.mkdir(parents=True,exist_ok=False)
if kind in ('shoot','small'):
 def source(r,l):return e*e*r*r+np.exp(-2*l)*(e*e*(1-r*r)**2+e**4*(1-r*r)**4/4)
 def shoot(l0,end=2.,dense=False):
  r0=1e-7;A=source(0,l0)
  return solve_ivp(lambda r,v:[v[1],-v[1]/r-source(r,v[0])],(r0,end),[l0-A*r0*r0/4,-A*r0/2],method='DOP853',rtol=3e-14,atol=2e-15,dense_output=dense)
 l0=brentq(lambda l:shoot(l).y[0,-1],0,4*e*e,xtol=1e-15,rtol=1e-14)
 sol=shoot(l0,3.2,True)
 assert sol.success
 old_initial=solver.initial
 def independent_initial(config):
  template=dict(config)
  if kind=='small':template['epsilon']=.04
  x,state,boundary,theta=old_initial(template);X,Y=np.meshgrid(x,x,indexing='ij');r=np.hypot(X,Y)
  lam=sol.sol(np.maximum(r,1e-7).ravel())[0].reshape(r.shape);lam[r==0]=l0
  oldgamma=state[3].copy();oldpi=state[13].copy()
  state[3]=state[5]=np.exp(2*lam);state[13]=e*np.exp(-2*lam)*(1-r*r)
  state[10]=-e*e*(1-r*r)**2;state[12]=e*X*Y
  info={'scope':'Independent shooting and direct dense evaluation versus tabulated BVP plus Hermite interpolation; numerical discrepancy, not rigorous bound.','max_metric_difference':float(np.max(abs(state[3]-oldgamma))),'max_phi_velocity_difference':float(np.max(abs(state[13]-oldpi))),'lambda0':l0}
  if kind=='small':info={'lambda0':l0,'method':'Independent shooting, direct dense evaluation; epsilon=.04 was used only to allocate the template before every epsilon-dependent field was replaced. No BVP-versus-shooting discrepancy is inferred for these small-amplitude runs.'}
  (out/'initial_difference.json').write_text(json.dumps(info,indent=2))
  return x,state,boundary,theta
 solver.initial=independent_initial
(out/'config.json').write_text(json.dumps(config,indent=2))
solver.evolve(config,out)
