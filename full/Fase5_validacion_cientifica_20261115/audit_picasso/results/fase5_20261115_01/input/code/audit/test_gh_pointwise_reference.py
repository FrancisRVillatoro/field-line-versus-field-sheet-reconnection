#!/usr/bin/env python3
"""Pointwise, batched, and exact-solution tests. Requires Python and NumPy only."""
from pathlib import Path
import json
import platform
import time
import numpy as np
from gh_pointwise_reference import geometry, metric_tt, scalar_tt

HERE=Path(__file__).resolve().parent
checks=[]
def check(name,got,expected,atol=2e-11):
    got=np.asarray(got); expected=np.asarray(expected)
    error=float(np.max(np.abs(got-expected)))
    scale=max(1.,float(np.max(np.abs(expected))))
    assert error<=atol*scale,(name,error,scale)
    checks.append({'name':name,'absolute_residual':error,'relative_scale':scale})

g=np.diag([-1.,1.,1.]); dg=np.zeros((3,3,3)); ddg=np.zeros((3,3,3,3)); dp=np.zeros(3)
flat=geometry(g,dg,ddg)
check('Minkowski_Ricci',flat['ricci'],np.zeros((3,3)))
check('Minkowski_contracted_Gamma',flat['contracted_gamma'],np.zeros(3))
check('Minkowski_metric_RHS',metric_tt(g,dg,ddg,dp),np.zeros((3,3)))

def homogeneous(u):
    g=np.diag([-np.exp(2*u),np.exp(u),np.exp(u)])
    dg=np.zeros((3,3,3)); ddg=np.zeros((3,3,3,3))
    dg[0]=np.diag([-2*np.exp(2*u),np.exp(u),np.exp(u)])
    ddg[0,0]=np.diag([-4*np.exp(2*u),np.exp(u),np.exp(u)])
    return g,dg,ddg,np.array([.5,0.,0.])
for u in [-.7,0.,.6]:
    g,dg,ddg,dp=homogeneous(u)
    exact=geometry(g,dg,ddg)
    check(f'homogeneous_Gamma_{u}',exact['contracted_gamma'],np.zeros(3))
    check(f'homogeneous_Ricci_{u}',exact['ricci'],2*np.outer(dp,dp))
    check(f'homogeneous_metric_RHS_{u}',metric_tt(g,dg,ddg,dp),ddg[0,0])
    check(f'homogeneous_scalar_RHS_{u}',scalar_tt(g,dg,np.zeros((3,3)),dp),0.)

# Anisotropic exact Einstein-scalar family. p=1,q=4 uplifts to curved
# vacuum Kasner with exponents (-1/3,2/3,2/3), not to flat Milne.
for u in [-.05,0.,.08]:
    p=1.; q=4.; rates=np.array([2*(p+q),2*p,2*q])
    diag=np.array([-1.,1.,1.])*np.exp(rates*u)
    g=np.diag(diag); dg=np.zeros((3,3,3)); ddg=np.zeros((3,3,3,3))
    dg[0]=np.diag(rates*diag); ddg[0,0]=np.diag(rates*rates*diag)
    dp=np.array([np.sqrt(p*q),0.,0.]); exact=geometry(g,dg,ddg)
    check(f'curved_Kasner_Gamma_{u}',exact['contracted_gamma'],np.zeros(3))
    check(f'curved_Kasner_Ricci_{u}',exact['ricci'],2*np.outer(dp,dp))
    check(f'curved_Kasner_metric_RHS_{u}',metric_tt(g,dg,ddg,dp),ddg[0,0])
    check(f'curved_Kasner_scalar_RHS_{u}',scalar_tt(g,dg,np.zeros((3,3)),dp),0.)

# Nonhomogeneous flat metric in exact harmonic coordinates:
# T=t, Y=y, X=x-F(t-y), F(v)=alpha*sin(v).
# Every new coordinate is harmonic in Minkowski space. This tests mixed
# metric derivatives, cross terms, inverse metric, and all Ricci components.
def harmonic_flat(t,y,alpha=.4):
    z=t-y; a=alpha*np.cos(z); ap=-alpha*np.sin(z); app=-a
    g=np.array([[-1+a*a,-a,-a*a],[-a,1,a],[-a*a,a,1+a*a]])
    ma=np.array([[2*a,-1,-2*a],[-1,0,1],[-2*a,1,2*a]])
    maa=np.array([[2.,0.,-2.],[0.,0.,0.],[-2.,0.,2.]])
    gu=ma*ap; guu=maa*ap*ap+ma*app
    dg=np.zeros((3,3,3)); ddg=np.zeros((3,3,3,3))
    dg[0]=gu; dg[2]=-gu
    ddg[0,0]=ddg[2,2]=guu
    ddg[0,2]=ddg[2,0]=-guu
    return g,dg,ddg
for t,y in [(.2,-.4),(.8,.3),(-.6,.2)]:
    g,dg,ddg=harmonic_flat(t,y)
    exact=geometry(g,dg,ddg)
    check(f'harmonic_flat_det_{t}',np.linalg.det(g),-1.)
    check(f'harmonic_flat_Gamma_{t}',exact['contracted_gamma'],np.zeros(3))
    check(f'harmonic_flat_Ricci_{t}',exact['ricci'],np.zeros((3,3)))
    check(f'harmonic_flat_metric_RHS_{t}',metric_tt(g,dg,ddg,np.zeros(3)),ddg[0,0])
    # A separate TEST scalar sin(t-y) solves the wave equation. It is not
    # included as a nonzero stress source for the preceding flat metric.
    c=np.cos(t-y); ss=np.sin(t-y); dp=np.array([c,0.,-c])
    d2=np.array([[-ss,0.,ss],[0.,0.,0.],[ss,0.,-ss]])
    check(f'harmonic_flat_test_scalar_{t}',scalar_tt(g,dg,d2,dp),-ss)

# Principal-symbol identity for arbitrary local metric jets. These are
# algebraic inputs, not assumed Einstein data or a harmonic metric.
rng=np.random.default_rng(1937)
batch=17
R=.07*rng.normal(size=(batch,3,3)); R=.5*(R+R.swapaxes(-1,-2))
g=np.broadcast_to(np.diag([-1.,1.,1.]),(batch,3,3)).copy()+R
dg=.2*rng.normal(size=(batch,3,3,3)); dg=.5*(dg+dg.swapaxes(-1,-2))
ddg=.2*rng.normal(size=(batch,3,3,3,3))
ddg=.5*(ddg+ddg.swapaxes(-1,-2)); ddg=.5*(ddg+ddg.swapaxes(-3,-4))
ddg[:,0,0]=0.
V=rng.normal(size=(batch,3,3)); V=.5*(V+V.swapaxes(-1,-2))
withtt=ddg.copy(); withtt[:,0,0]=V
a=geometry(g,dg,ddg); b=geometry(g,dg,withtt)
check('random_jet_principal_symbol',b['reduced_ricci']-a['reduced_ricci'],-.5*a['inverse_metric'][:,0,0,None,None]*V)
dp=.2*rng.normal(size=(batch,3))
predicted=metric_tt(g,dg,ddg,dp)
ddsolve=ddg.copy(); ddsolve[:,0,0]=predicted
solved=geometry(g,dg,ddsolve)
check('random_jet_RHS_substitution',solved['reduced_ricci'],2*np.einsum('...a,...b->...ab',dp,dp))
for j in (0,8,16):
    check('batch_agrees_point_'+str(j),predicted[j],metric_tt(g[j],dg[j],ddg[j],dp[j]))

# Initial harmonic gauge: gamma=e^(2lambda)delta, K trace zero and Kxy=B.
lam=.13; lx=.31; ly=-.17; B=.024; E=np.exp(2*lam)
g=np.diag([-1.,E,E]); dg=np.zeros((3,3,3)); dg[0,1,2]=dg[0,2,1]=-2*B
dg[1,1,1]=dg[1,2,2]=2*E*lx
dg[2,1,1]=dg[2,2,2]=2*E*ly
exact=geometry(g,dg,np.zeros((3,3,3,3)))
check('conformal_tracefree_initial_gauge',exact['contracted_gamma'],np.zeros(3))

# Small throughput probe only, not a benchmark of a PDE implementation.
n=4096
bg=np.broadcast_to(g,(n,3,3)).copy()
bdg=np.broadcast_to(dg,(n,3,3,3)).copy()
bddg=np.zeros((n,3,3,3,3)); bdp=np.zeros((n,3))
start=time.perf_counter(); z=metric_tt(bg,bdg,bddg,bdp); elapsed=time.perf_counter()-start
assert np.all(np.isfinite(z))
result={'status':'PASS','number_of_checks':len(checks),'checks':checks,
        'versions':{'python':platform.python_version(),'numpy':np.__version__},
        'batch_probe':{'points':n,'wall_seconds':elapsed},
        'scope':'Independent pointwise GH RHS and exact-solution tests. No grid evolution, convergence test, or physical reconnection validation.'}
(HERE/'gh_pointwise_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
