#!/usr/bin/env python3
"""Compute-node smoke: independent tensors, time integrator, and full PDE path."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'code/audit')]
from gh_core import metric_geometry,rhs,pack
from gh_pointwise_reference import geometry
from solver import evolve
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
testdir=out/'pointwise';testdir.mkdir(exist_ok=True)
for name in ['gh_pointwise_reference.py','test_gh_pointwise_reference.py']:shutil.copy2(ROOT/'code/audit'/name,testdir/name)
with (testdir/'test.log').open('w') as f:subprocess.run([sys.executable,str(testdir/'test_gh_pointwise_reference.py')],check=True,stdout=f,stderr=subprocess.STDOUT)
rng=np.random.default_rng(73);g=np.broadcast_to(np.diag([-1.,1.,1.]),(9,3,3)).copy();z=.03*rng.normal(size=g.shape);g+=(z+z.swapaxes(-1,-2))/2
dg=.1*rng.normal(size=(9,3,3,3));dg=(dg+dg.swapaxes(-1,-2))/2
_,_,_,Q=metric_geometry(g,dg);ref=geometry(g,dg,np.zeros((9,3,3,3,3)))
qerror=float(np.max(abs(Q-ref['reduced_ricci'])));assert qerror<1e-13
# Curved vacuum Kasner uplift: h=(-exp(10T),exp(2T),exp(8T)), phi=2T.
errors=[]
for steps in [10,20,40]:
 Y=np.zeros((15,7,7));Y[0]=-1;Y[3]=Y[5]=1;Y[6]=-10;Y[9]=2;Y[11]=8;Y[13]=2
 dt=.1/steps
 for _ in range(steps):
  f=lambda Z:rhs(Z,1.,6,freeze_width=0)[0]
  k1=f(Y);k2=f(Y+dt*k1/2);k3=f(Y+dt*k2/2);k4=f(Y+dt*k3)
  Y+=dt*(k1+2*k2+2*k3+k4)/6
 exact=np.array([-np.exp(1),0,0,np.exp(.2),0,np.exp(.8)])
 errors.append(float(np.max(abs(Y[:6,3,3]-exact))))
orders=np.log2(np.array(errors[:-1])/np.array(errors[1:]));assert min(orders)>3.7,(errors,orders)
flat={'N':33,'L':2.,'epsilon':0.,'nb':64,'cfl':.15,'Tmax':.03,'order':6,'slice_targets':[0,.02],'diagnostic_radius':.7,'tag':'smoke_flat'}
(out/'flat').mkdir(exist_ok=True);r=evolve(flat,out/'flat')
flatmax=max(max(v['hamiltonian_linf'],v['momentum_linf'],v['gauge_linf']) for v in r['diagnostics']);assert flatmax<1e-12
nonlinear=dict(flat,epsilon=.08,tag='smoke_nonlinear');(out/'nonlinear').mkdir(exist_ok=True);r=evolve(nonlinear,out/'nonlinear');assert r['status']=='SUCCESS' and r['boundary']['max_V_norm']<0
result={'status':'PASS','independent_pointwise_checks':48,'production_Q_independent_error':qerror,'curved_Kasner_RK4_errors':errors,'curved_Kasner_RK4_orders':orders.tolist(),'flat_constraint_max':flatmax,'nonlinear_smoke_status':r['status']}
(out/'self_test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
