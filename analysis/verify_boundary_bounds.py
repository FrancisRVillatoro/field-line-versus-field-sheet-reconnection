"""Exact comparison identities and an independent ODE check for the new lemma.

The inequalities are proved in the manuscript. Sampling here is a sanity check,
not a replacement for the comparison proof and not a finite-epsilon bound.
"""
from pathlib import Path
import json
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
s,u,v=sp.symbols('s u v',real=True)
a=1-2*s;b=1+2*s
D=a*a*u*u+b*b*v*v;Q=a*u*u-b*v*v
mp=(2*s*s-sp.Rational(8,3)*s**3)/(1-2*s)
mm=(2*s*s+sp.Rational(8,3)*s**3)/(1+2*s)
rp=1+sp.Rational(4,3)*s**3/(1-2*s)
rm=1-sp.Rational(4,3)*s**3/(1+2*s)
checks={}
def check(name,val):
    assert sp.factor(val)==0,(name,sp.factor(val));checks[name]=True
check('upper_Rayleigh_gap',D-a*Q-2*b*v*v)
check('lower_Rayleigh_gap',D+b*Q-2*a*u*u)
check('upper_comparison_ODE',sp.diff(mp,s)-4*s-2*mp/(1-2*s))
check('lower_comparison_ODE',sp.diff(mm,s)-4*s+2*mm/(1+2*s))
check('upper_radius_formula',1-2*s*s+mp-rp)
check('lower_radius_formula',1-2*s*s+mm-rm)
S=sp.Rational(1,4)
check('upper_radius_at_S',rp.subs(s,S)-sp.Rational(25,24))
check('lower_radius_at_S',rm.subs(s,S)-sp.Rational(71,72))
check('maximum_m_at_S',mp.subs(s,S)-sp.Rational(1,6))
check('uniform_timelike_margin',-1+(sp.Rational(1,6))**2/(sp.Rational(1,4)*sp.Rational(71,72))+sp.Rational(63,71))
check('crossing_label_at_S',(s-sp.Rational(2,3)*s**3).subs(s,S)-sp.Rational(23,96))
theta=np.linspace(0,2*np.pi,257)[:-1]
X=np.stack([np.cos(theta),np.sin(theta)],axis=0)
def rhs(t,flat):
    x,y=flat.reshape(2,-1);p=x*x+y*y;fs=1-p-2*t*t
    gx=y-2*t*x;gy=x-2*t*y;den=gx*gx+gy*gy
    return np.stack([-fs*gx/den,-fs*gy/den]).ravel()
ts=np.linspace(0,.25,101)
sol=solve_ivp(rhs,[0,.25],X.ravel(),method='DOP853',t_eval=ts,rtol=2e-12,atol=2e-14)
assert sol.success
arr=sol.y.reshape(2,len(theta),len(ts));xx,yy=arr;rr=xx*xx+yy*yy
fs=1-rr-2*ts**2;Dnum=(yy-2*ts*xx)**2+(xx-2*ts*yy)**2
norm=-1+fs*fs/Dnum
lo=1-4/3*ts**3/(1+2*ts);hi=1+4/3*ts**3/(1-2*ts)
label=xx*yy+ts*(1-rr)-2/3*ts**3
numerical={'trajectories':len(theta),'time_samples':len(ts),
 'minimum_radial_lower_slack':float(np.min(rr-lo)),
 'minimum_radial_upper_slack':float(np.min(hi-rr)),
 'max_label_drift':float(np.max(np.abs(label-.5*np.sin(2*theta[:,None])))),
 'maximum_flat_V_norm':float(np.max(norm)),
 'proven_uniform_norm_bound':-63/71}
assert numerical['minimum_radial_lower_slack']>-2e-10
assert numerical['minimum_radial_upper_slack']>-2e-10
assert numerical['max_label_drift']<2e-10
assert numerical['maximum_flat_V_norm']<-63/71+2e-10
result={'status':'PASS','exact_identity_count':len(checks),'checks':checks,'independent_ODE_sanity_check':numerical,
 'scope':'Bounds for the normalized leading field only. Small-amplitude nonlinear persistence is a qualitative theorem; no explicit epsilon_0 is computed.'}
Path(__file__).with_name('boundary_bounds_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
