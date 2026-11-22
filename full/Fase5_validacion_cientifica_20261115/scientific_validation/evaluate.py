import sys,json,math,collections
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
B=Path(__file__).resolve().parent;R=B.parent/'audit_picasso/results/fase5_20261115_01';S=R/'input';sys.path.insert(0,str(S))
from analyze_campaign import analyze_trace,predictions,observed_order
labels=[.04,.12,.18];rows=[];refinements=[]
def data(p):
 with np.load(p/'evolution.npz') as z:trace=z['center_trace'].copy()
 s=json.loads((p/'summary.json').read_text());e=s['config']['epsilon'];a=analyze_trace(trace,e,labels)
 return {v['c']:v for v in a['events']},trace,s
for k,e in enumerate([.04,.08,.12]):
 tag=f'e{round(e*100):02d}'
 paths={'fine':B/'extra'/f'{tag}_N129_dt2','midspace':R/f'task_{[10,12,14][k]:03d}','coarsespace':B/'extra'/f'{tag}_N65_dt2','midtime':R/f'task_{[2,5,8][k]:03d}','coarsetime':B/'extra'/f'{tag}_N129_dt_coarse','shoot':B/'extra'/f'{tag}_shoot_direct','domain':R/f'task_{15+k:03d}','base':R/f'task_{1+3*k:03d}'}
 D={name:data(p) for name,p in paths.items()}
 for c in labels:
  fine,trace,summary=D['fine'];ev=fine[c];spl=[CubicSpline(trace[:,0],trace[:,i]) for i in [1,2,4]]
  time=brentq(lambda t:float(spl[1](t))-e*c,trace[0,0],trace[-1,0],xtol=1e-15)
  fixed=brentq(lambda t:float(spl[0](t))-ev['s0'],trace[0,0],trace[-1,0],xtol=1e-15)
  independent={'s_event':float(spl[0](time)),'paired_tau_event':float(spl[2](time)),'fixed_s_a_over_eps':float(spl[1](fixed))/e}
  for key,residual,leading in [('s_event','s_error','leading_s_delay'),('paired_tau_event','paired_tau_error','leading_paired_tau_correction'),('fixed_s_a_over_eps','fixed_s_corrected_residual',None)]:
   ds=abs(ev[key]-D['midspace'][0][c][key]);dt=abs(ev[key]-D['midtime'][0][c][key]);di=abs(ev[key]-D['shoot'][0][c][key]);dip=abs(ev[key]-independent[key]);dd=abs(D['domain'][0][c][key]-D['base'][0][c][key]);floor=64*np.finfo(float).eps*max(1,abs(ev[key]));U=10*(ds+dt+di+dip+dd)+floor
   row={'epsilon':e,'c':c,'observable':key,'observed':ev[key],'residual':ev[residual],'space_difference':ds,'time_difference':dt,'initial_data_propagated_difference':di,'interpolation_difference':dip,'domain_difference_proxy':dd,'roundoff_floor':floor,'sensitivity_indicator':U,'residual_to_indicator':abs(ev[residual])/U,'residual_resolved':abs(ev[residual])>5*U}
   if leading:row.update(leading_correction=ev[leading],prediction_correction_resolved=U<.1*abs(ev[leading]))
   rows.append(row)
   for kind,names in [('space_time',['coarsespace','midspace','fine']),('time',['coarsetime','midtime','fine'])]:
    h=[D[n][2]['h' if kind=='space_time' else 'dt'] for n in names];values=[D[n][0][c][key] for n in names]
    refinements.append({'epsilon':e,'c':c,'observable':key,'kind':kind,'levels':h,'values':values,'diagnostic':observed_order(h,values,difference_floor=1e-12)})
scalings=[]
for c in labels:
 for key in ['s_event','paired_tau_event','fixed_s_a_over_eps']:
  a=sorted([r for r in rows if r['c']==c and r['observable']==key],key=lambda r:r['epsilon']);sl=[];bands=[]
  for i in [0,1]:
   x,y=a[i:i+2];ratio=y['epsilon']/x['epsilon'];ax,ay=abs(x['residual']),abs(y['residual']);ux,uy=x['sensitivity_indicator'],y['sensitivity_indicator']
   sl.append(math.log(ay/ax)/math.log(ratio));bands.append([math.log((ay-uy)/(ax+ux))/math.log(ratio),math.log((ay+uy)/(ax-ux))/math.log(ratio)])
  scalings.append({'c':c,'observable':key,'adjacent_slopes':sl,'sensitivity_intervals_not_confidence_intervals':bands})
# Actual traced outlets before and after each directly measured event.
switches=[]
for e in [.04,.08,.12]:
 a=json.loads((B/'near_events'/f'e{round(e*100):02d}'/'slice_audit.json').read_text())
 for j,c in enumerate(labels):
  lo,hi=a['slices'][1+2*j:3+2*j]
  for inlet in ['N','S']:
   before=next(v for v in lo['cases'] if abs(v['c']-c)<1e-12 and v['inlet']==inlet);after=next(v for v in hi['cases'] if abs(v['c']-c)<1e-12 and v['inlet']==inlet)
   switches.append({'epsilon':e,'c':c,'inlet':inlet,'s_before':lo['s'],'s_after':hi['s'],'before_status':before['status'],'after_status':after['status'],'before_outlet':before.get('outlet'),'after_outlet':after.get('outlet'),'switched':before.get('outlet')!=after.get('outlet'),'minimum_gap_over_guard':min(abs(before['gap'])/before['guard'],abs(after['gap'])/after['guard']),'max_label_error':max(before.get('label_error',0),after.get('label_error',0))})
# Constraints, clocks, causality for all original and successful supplementary runs.
controls=[]
paths=[p for p in R.glob('task_*') if p.is_dir()]+[p for p in (B/'extra').iterdir() if (p/'summary.json').exists() and 'shoot_N129' not in p.name]+list((B/'near_events').glob('*'))
for p in paths:
 s=json.loads((p/'summary.json').read_text());controls.append({'case':p.name,'epsilon':s['config']['epsilon'],'H':max(d['hamiltonian_linf'] for d in s['diagnostics']),'M':max(d['momentum_linf'] for d in s['diagnostics']),'C':max(d['gauge_linf'] for d in s['diagnostics']),'min_lapse':min(d['min_lapse'] for d in s['diagnostics']),'min_spatial_eigenvalue':min(d['min_spatial_metric_eigenvalue'] for d in s['diagnostics']),'center_clock_discrepancy':s['center_gaussian_eikonal_discrepancy'],'max_V_norm':s['boundary'].get('max_V_norm'),'min_Dphi_squared':s['boundary'].get('min_spatial_gradient2'),'slices_complete':all(s['slices_complete_in_diagnostic_disk']) and all(s['boundary_slices_complete'])})
result={'scope':'Empirical numerical validation of sampled regular connectivity and reserved finite-amplitude predictions. Not interval-certified error bounds or a global topology theorem.','protocol':{'labels':labels,'amplitudes':[.04,.08,.12],'reference':'N129 CFL0.075','indicator':'10*(space+time+propagated initial+independent interpolation+domain proxy)+64*machine_epsilon*max(1,abs(observable))','factor_10':'Explicit conservative sensitivity multiplier for this supplementary audit; not a proven error constant.','signal_gate':'abs(residual)>5*indicator','leading_gate':'indicator<0.1*abs(leading correction)','prior_status_defect':'Original top-level acceptance is a literal NOT_ESTABLISHED; original scaling groups required disjoint finest-task identities.'},'observables':rows,'refinement_diagnostics':refinements,'scalings':scalings,'switches':switches,'controls':controls,'all_residuals_resolved':all(x['residual_resolved'] for x in rows),'all_leading_corrections_resolved':all(x.get('prediction_correction_resolved',True) for x in rows),'all_sampled_switches_resolved':all(x['before_status']=='EXIT' and x['after_status']=='EXIT' and x['switched'] for x in switches)}
(B/'scientific_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda v:v.item()))
print('PASS_GATES',result['all_residuals_resolved'],result['all_leading_corrections_resolved'],result['all_sampled_switches_resolved'])
print('ROWS',len(rows),'SWITCHES',len(switches),'CONTROLS',len(controls))
for key in ['space_difference','time_difference','initial_data_propagated_difference','interpolation_difference','domain_difference_proxy','sensitivity_indicator']:print('MAX',key,max(r[key] for r in rows))
print('MIN_RESIDUAL_TO_INDICATOR',min(r['residual_to_indicator'] for r in rows))
print('MIN_SWITCH_GAP_GUARD',min(r['minimum_gap_over_guard'] for r in switches))
for r in rows:
 if r['c']==.12 and r['observable']=='s_event':print('EVENT_C12',r)
print('SLOPES',scalings)
print('REFINEMENT',collections.Counter(x['diagnostic']['status'] for x in refinements))
print('HMC_EXTRA',[(x['case'],x['H'],x['M'],x['C']) for x in controls if 'N129_dt2' in x['case']])
print('MIN_LAPSE',min(x['min_lapse'] for x in controls),'MAX_V',max(x['max_V_norm'] for x in controls if x['max_V_norm'] is not None),'MIN_METRIC_EIGEN',min(x['min_spatial_eigenvalue'] for x in controls))
