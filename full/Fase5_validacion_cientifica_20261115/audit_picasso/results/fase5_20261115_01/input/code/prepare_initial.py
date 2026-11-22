"""Local preprocessing only (SciPy); production solver reads bundled NumPy data."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_bvp,solve_ivp
ROOT=Path(__file__).resolve().parents[1]
r=np.linspace(0,3.2,16385);eps=np.array([0.,.04,.08,.12]);ls=[];ds=[];metadata=[]
for e in eps:
    if e==0:ls.append(r*0);ds.append(r*0);metadata.append({'epsilon':0,'lambda0':0});continue
    def source(r,l):return e*e*r*r+np.exp(-2*l)*(e*e*(1-r*r)**2+e**4*(1-r*r)**4/4)
    rr=np.linspace(0,2,257);L2=16/9-rr*rr/4+rr**4/16-rr**6/36
    init=np.array([e*e*L2,e*e*(-rr/2+rr**3/4-rr**5/6)])
    sol=solve_bvp(lambda r,y:np.vstack((y[1],-source(r,y[0]))),lambda a,b:np.array([a[1],b[0]]),rr,init,S=np.array([[0.,0.],[0.,-1.]]),tol=2e-12,max_nodes=30000)
    assert sol.success
    ext=solve_ivp(lambda r,y:[y[1],-y[1]/r-source(r,y[0])],[2,3.2],sol.sol(2),method='DOP853',rtol=2e-13,atol=2e-15,dense_output=True)
    assert ext.success
    vals=np.empty((2,len(r)));m=r<=2;vals[:,m]=sol.sol(r[m]);vals[:,~m]=ext.sol(r[~m]);ls.append(vals[0]);ds.append(vals[1]);metadata.append({'epsilon':float(e),'lambda0':float(sol.sol(0)[0]),'max_rms_residual':float(max(sol.rms_residuals)),'outer_extension':'Radial ODE continuation of the same constrained initial data beyond r=2; not a new lambda Dirichlet normalization.'})
np.savez_compressed(ROOT/'data/radial_profiles.npz',r=r,epsilon=eps,lam=np.array(ls),lam_r=np.array(ds))
(ROOT/'data/radial_profiles_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(metadata)
