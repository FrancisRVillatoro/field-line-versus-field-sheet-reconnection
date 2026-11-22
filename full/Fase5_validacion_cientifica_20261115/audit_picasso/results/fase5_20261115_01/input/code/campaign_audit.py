#!/usr/bin/env python3
"""Summarize independent diagnostics; empirical comparisons are not certified bounds."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_campaign import analyze_campaign,write_analysis
from gh_core import rhs

def audit(run):
 run=Path(run);a=analyze_campaign(run);write_analysis(run,a);records=[];data={}
 for task in a['tasks']:
  p=run/task['task']
  try:
   s=json.loads((p/'summary.json').read_text());c=s['config']
   if s.get('status')!='SUCCESS':continue
   z=np.load(p/'evolution.npz');d=s['diagnostics'];B=z['final_boundary'];x=z['x'];Y=z['final_state'];xx,yy=np.meshgrid(x,x,indexing='ij');mask=xx*xx+yy*yy<=1.1**2
   k,kb,aux=rhs(Y,x[1]-x[0],c['order'],B if c['epsilon'] else None,x,x)
   rec={'task':task['task'],'config':c,'H_max':max(v['hamiltonian_linf'] for v in d),'M_max':max(v['momentum_linf'] for v in d),'C_max':max(v['gauge_linf'] for v in d),
    'min_lapse':min(v['min_lapse'] for v in d),'final_min_s_T_in_disk':float(np.min(k[14][mask])),
    'max_midpoint_eikonal_residual':s['max_midpoint_eikonal_residual'],'max_material_label_error':s['max_material_label_error'],
    'max_V_norm':s.get('boundary',{}).get('max_V_norm'),'runtime_seconds':s['runtime_seconds'],'slice_inverse_residual':s['slice_inverse_residual']}
   if (p/'slice_audit.json').is_file():
    sa=json.loads((p/'slice_audit.json').read_text());sl=sa['slices']
    rec.update(traced_exits=sum(v['traced_exits'] for v in sl),mismatched_pairs=sum(v['mismatched_pairs'] for v in sl),unresolved_eligible=sum(v['unresolved_eligible_cases'] for v in sl),unresolved_critical=sum(v['unresolved']-v['unresolved_eligible_cases'] for v in sl),max_label_angle_error=max(v['max_label_error'] or 0 for v in sl),max_level_drift=max(v['max_level_drift'] or 0 for v in sl))
   records.append(rec);data[task['task']]={'trace':z['center_trace'],'state':Y,'x':x,'config':c,'analysis':task.get('trace_analysis',{})}
  except (OSError,KeyError,ValueError) as e:records.append({'task':task['task'],'error':str(e)})
 # Comparisons use fixed physical parameters; tags never determine an analytic coefficient.
 comparisons=[];empirical=[]
 def event(id,c,key):
  for e in data.get(id,{}).get('analysis',{}).get('events',[]):
   if abs(e.get('c',float('inf'))-c)<1e-12 and e.get('status')=='FOUND':
    value=e.get(key)
    if isinstance(value,(int,float)) and np.isfinite(value):return float(value)
  return None
 valid=[r for r in records if 'config' in r]
 for eps in [.04,.08,.12]:
  base=next((r for r in valid if r['config']['epsilon']==eps and r['config']['N']==97 and r['config']['L']==2 and r['config']['order']==6 and r['config']['cfl']==.15 and r['config']['nb']==256 and r['config'].get('direction',1)==1),None)
  if base is None:continue
  bi=base['task']
  for r in valid:
   c=r['config'];ti=r['task']
   if ti==bi or c['epsilon']!=eps or c.get('direction',1)<0:continue
   if c['order']==6:
    vals={};missing=[]
    for key in ['s_event','paired_tau_event','fixed_s_a_over_eps']:
     differences=[]
     for lab in [.04,.12,.18]:
      other,reference=event(ti,lab,key),event(bi,lab,key)
      if other is None or reference is None:missing.append({'observable':key,'c':lab})
      else:differences.append(abs(other-reference))
     vals[key]=max(differences) if len(differences)==3 else None
    comparisons.append({'reference':bi,'other':ti,'epsilon':eps,'config':c,'status':'AVAILABLE' if not missing else 'UNAVAILABLE_EVENTS','missing':missing,'maximum_event_differences':vals})
 for label in [.04,.12,.18]:
  for key in ['s_error','paired_tau_error','fixed_s_corrected_residual']:
   ids=[]
   for eps in [.04,.08,.12]:
    match=[r['task'] for r in valid if r['config']['epsilon']==eps and r['config']['N']==129 and r['config']['L']==2 and r['config']['order']==6]
    if match:ids.append(match[0])
   if len(ids)==3:
    values=[event(i,label,key) for i in ids]
    if any(v is None or v==0 for v in values):
     empirical.append({'c':label,'residual':key,'epsilon':[.04,.08,.12],'values':values,'status':'UNAVAILABLE_OR_ZERO_RESIDUAL','adjacent_slopes':None})
    else:
     residuals=np.array(values);slopes=np.log(abs(residuals[1:]/residuals[:-1]))/np.log([2.,1.5])
     empirical.append({'c':label,'residual':key,'epsilon':[.04,.08,.12],'values':residuals.tolist(),'status':'EMPIRICAL_ONLY','adjacent_slopes':slopes.tolist(),'scope':'Empirical ratios; initial-data and continuum errors are not certified bounds.'})
 parity={}
 neg=next((r for r in valid if r['config'].get('direction',1)<0),None)
 if neg:
  cn=neg['config'];pos=next((r for r in valid if all(r['config'].get(k)==cn.get(k) for k in ['epsilon','N','L','cfl','order','nb']) and r['config'].get('direction',1)>0),None)
  if pos:
   tp=data[pos['task']]['trace'];tn=data[neg['task']]['trace']
   if tp.shape==tn.shape:
    parity={'positive_task':pos['task'],'negative_task':neg['task'],'max_a_oddness_error':float(np.max(abs(tp[:,2]+tn[:,2]))),'max_s_oddness_error':float(np.max(abs(tp[:,1]+tn[:,1]))),'paired_clock_direct_discrepancy':float(np.max(abs(tp[:,3]-tn[:,3]-tp[:,4])))}
 result={'scope':'Local numerical evidence for selected events and traces; no global topology theorem or certified discretization bound. Cluster execution must be identified by Slurm records.',
 'expected_tasks':a['task_count'],'successful_numerical_tasks':len(valid),'traced_exits':sum(r.get('traced_exits',0) for r in valid),'mismatched_pairs':sum(r.get('mismatched_pairs',0) for r in valid),
 'unresolved_eligible':sum(r.get('unresolved_eligible',0) for r in valid),'unresolved_critical':sum(r.get('unresolved_critical',0) for r in valid),
 'tasks':records,'controlled_comparisons':comparisons,'empirical_residual_scaling':empirical,'time_reversal_control':parity}
 (run/'numerical_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ['expected_tasks','successful_numerical_tasks','traced_exits','mismatched_pairs','unresolved_eligible','unresolved_critical']}))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args();audit(args.run_dir)
