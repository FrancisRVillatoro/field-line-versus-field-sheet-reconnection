import sys,json,math
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'audit_picasso/results/fase5_20261115_01/input'))
from analyze_campaign import analyze_trace
original=json.loads((B/'scientific_validation.json').read_text());rows=[]
for e in [.01,.02]:
 ds={}
 for N,cfl in [(129,.075),(97,.075),(129,.15)]:
  p=B/'extra'/f'e{round(e*100):02d}_small_N{N}_cfl{cfl}'
  with np.load(p/'evolution.npz') as z:t=z['center_trace']
  ds[(N,cfl)]={v['c']:v for v in analyze_trace(t,e)['events']}
  if (N,cfl)==(129,.075):fine_trace=t
 for c in [.04,.12,.18]:
  spl=[CubicSpline(fine_trace[:,0],fine_trace[:,i]) for i in [1,2,4]]
  root=brentq(lambda t:float(spl[1](t))-e*c,fine_trace[0,0],fine_trace[-1,0],xtol=1e-15)
  v=ds[(129,.075)][c];time0=brentq(lambda t:float(spl[0](t))-v['s0'],fine_trace[0,0],fine_trace[-1,0],xtol=1e-15)
  alt={'s_event':float(spl[0](root)),'paired_tau_event':float(spl[2](root)),'fixed_s_a_over_eps':float(spl[1](time0))/e}
  for key,res in [('s_event','s_error'),('paired_tau_event','paired_tau_error'),('fixed_s_a_over_eps','fixed_s_corrected_residual')]:
   space=abs(v[key]-ds[(97,.075)][c][key]);time=abs(v[key]-ds[(129,.15)][c][key]);interp=abs(v[key]-alt[key]);U=10*(space+time+interp)+64*np.finfo(float).eps
   rows.append({'epsilon':e,'c':c,'observable':key,'observed':v[key],'residual':v[res],'space_difference':space,'time_difference':time,'interpolation_difference':interp,'sensitivity_indicator':U,'residual_to_indicator':abs(v[res])/U,'note':'Small-amplitude extension uses shooting initial data; indicator includes space/time/interpolation only. Initial-method comparison was performed for the three original amplitudes.'})
slopes=[]
for c in [.04,.12,.18]:
 for key in ['s_event','paired_tau_event','fixed_s_a_over_eps']:
  q=sorted([v for v in rows+original['observables'] if v['c']==c and v['observable']==key],key=lambda v:v['epsilon']);sl=[]
  for x,y in zip(q[:-1],q[1:]):sl.append({'epsilon_pair':[x['epsilon'],y['epsilon']],'power':math.log(abs(y['residual']/x['residual']))/math.log(y['epsilon']/x['epsilon'])})
  slopes.append({'c':c,'observable':key,'slopes':sl})
result={'observables':rows,'five_amplitude_scalings':slopes,'minimum_residual_to_indicator':min(v['residual_to_indicator'] for v in rows)}
(B/'small_amplitude_validation.json').write_text(json.dumps(result,indent=2,default=lambda v:v.item()))
print('MIN_RESIDUAL_RATIO',result['minimum_residual_to_indicator'])
for r in slopes:
 if r['c']==.12:print(r)
