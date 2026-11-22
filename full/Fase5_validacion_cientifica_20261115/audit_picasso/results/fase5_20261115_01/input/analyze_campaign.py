#!/usr/bin/env python3
"""NumPy-only, non-fitting analysis of the reserved Phase 5 predictions.

Usage: python analyze_campaign.py --run-dir RUN [--thresholds settings.json]
Writes RUN/analysis.json and RUN/event_errors.csv. Missing/failed tasks are
retained in the report. No execution result is labeled a scientific PASS.
Reusable functions: predictions, hermite_event, analyze_trace,
observed_order, analyze_campaign. The trace columns are
[T,s,a,tau,paired_tau,a_T,s_T], where s is central quotient proper time.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import zipfile
from datetime import datetime, timezone
import numpy as np

DEFAULTS={'labels':[.04,.12,.18], 'root_tolerance':1e-10,
          'derivative_floor':1e-12, 'difference_floor':1e-12,
          'minimum_order':.05, 'maximum_order':12.,
          'leading_delay_error_fraction':.1, 'scaling_noise_factor':5.,
          'include_interpolation_indicator':False, 'strict_tag_grouping':False}

def settings(overrides=None):
    out=dict(DEFAULTS)
    if overrides:
        unknown=set(overrides)-set(out)
        if unknown: raise ValueError('Unknown analysis settings: '+', '.join(sorted(unknown)))
        out.update(overrides)
    return out

def q(s): return np.asarray(s)-2*np.asarray(s)**3/3
def qprime(s): return 1-2*np.asarray(s)**2
def b(s):
    s=np.asarray(s)
    return -32*s/9+155*s**3/27-7*s**5/5+188*s**7/315-64*s**9/567
def Q2(s):
    s=np.asarray(s)
    return s**3/3-4*s**5/15+4*s**7/63

def predictions(eps,c):
    """Frozen coefficients from Phase 4; no data-dependent fitting."""
    eps=float(eps); c=float(c)
    if not math.isfinite(eps) or not math.isfinite(c) or eps<=0 or not 0<c<math.sqrt(2)/3:
        raise ValueError('Require eps>0 and 0<c<sqrt(2)/3 on increasing q branch.')
    s0=math.sqrt(2)*math.sin(math.asin(3*c/math.sqrt(2))/3)
    D=-float(b(s0))/float(qprime(s0))
    return {'c':c,'s0':s0,'D':D,'Q2':float(Q2(s0)),
            's_pred':s0+eps*eps*D,
            'paired_tau_pred':2*s0+eps*eps*(2*D+float(Q2(s0))),
            'leading_s_delay':eps*eps*D,
            'leading_paired_tau_correction':eps*eps*(2*D+float(Q2(s0)))}

def hermite_coeff(y0,y1,m0,m1,h):
    return np.array([y0,h*m0,3*(y1-y0)-h*(2*m0+m1),2*(y0-y1)+h*(m0+m1)],float)

def hermite_value(y0,y1,m0,m1,h,z):
    return float(np.polynomial.polynomial.polyval(z,hermite_coeff(y0,y1,m0,m1,h)))

def hermite_event(trace,target,column=2,derivative_column=5,root_tolerance=1e-10):
    """Find every cubic-Hermite crossing, accepting only one transverse root.

    Returns a status dict. No extrapolation or silent first-root selection.
    Roots at shared interval boundaries are deduplicated.
    """
    z=np.asarray(trace,float)
    if z.ndim!=2 or z.shape[0]<2 or z.shape[1]<7:
        return {'status':'INVALID_TRACE','reason':'Need at least two seven-column rows.'}
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z[:,0])>0):
        return {'status':'INVALID_TRACE','reason':'Trace must be finite and T strictly increasing.'}
    candidates=[]
    for i in range(len(z)-1):
        h=z[i+1,0]-z[i,0]
        co=hermite_coeff(z[i,column],z[i+1,column],z[i,derivative_column],z[i+1,derivative_column],h)
        co[0]-=target
        if np.all(co==0):
            return {'status':'NONTRANSVERSE','reason':'An interval is identically at the target.'}
        for root in np.polynomial.polynomial.polyroots(co):
            if abs(root.imag)<=root_tolerance and -root_tolerance<=root.real<=1+root_tolerance:
                r=float(np.clip(root.real,0,1)); T=float(z[i,0]+h*r)
                slope=float(np.polynomial.polynomial.polyval(r,[co[1],2*co[2],3*co[3]])/h)
                candidates.append({'T':T,'interval':i,'fraction':r,'slope':slope})
    candidates.sort(key=lambda r:r['T'])
    unique=[]
    for r in candidates:
        if not unique or abs(r['T']-unique[-1]['T'])>root_tolerance*max(1.,abs(r['T'])):
            unique.append(r)
    if not unique:
        return {'status':'NOT_REACHED','target':float(target),
                'sampled_range':[float(np.min(z[:,column])),float(np.max(z[:,column]))]}
    if len(unique)!=1: return {'status':'AMBIGUOUS','roots':unique,'target':float(target)}
    return dict(unique[0],status='FOUND',target=float(target))

def interpolate_column(trace,event,column,derivatives):
    i=event['interval']; r=event['fraction']; h=trace[i+1,0]-trace[i,0]
    return hermite_value(trace[i,column],trace[i+1,column],derivatives[i],derivatives[i+1],h,r)

def _initial_error_info(trace,eps,supplied):
    supplied=supplied if isinstance(supplied,dict) else {}
    observed={'a_over_eps':abs(float(trace[0,2]/eps)),
              's':abs(float(trace[0,1])), 'paired_tau':abs(float(trace[0,4]))}
    usable={}
    for k in observed:
        v=supplied.get(k)
        if isinstance(v,(float,int)) and np.isfinite(v) and v>=0: usable[k]=float(v)
    values={k:max(v,usable.get(k,0.)) for k,v in observed.items()}
    return {'values':values,'observed_initial_offsets':observed,'provided':usable,
            'all_three_provided':all(k in usable for k in observed),
            'scope':'Initial observable-error contributions, not a propagated bound on the full Cauchy error.'}

def analyze_trace(trace,eps,c_values=(.04,.12,.18),initial_errors=None,thresholds=None):
    th=settings(thresholds); trace=np.asarray(trace,float); eps=float(eps)
    if eps==0: return {'status':'SKIPPED_DEGENERATE_ZERO_FIELD','reason':'Zero amplitude has no amplitude-normalized scalar or nonzero-label event.'}
    if not math.isfinite(eps) or eps<=0: return {'status':'INVALID_CONFIG','reason':'Require finite positive epsilon.'}
    if trace.ndim!=2 or trace.shape[1]<7 or len(trace)<2:
        return {'status':'INVALID_TRACE','reason':'center_trace needs at least two rows and seven columns.'}
    trace=trace[:,:7]
    if not np.all(np.isfinite(trace)):
        return {'status':'INVALID_TRACE','reason':'Nonfinite values retained as a failure; no rows discarded.'}
    if not np.all(np.diff(trace[:,0])>0):
        return {'status':'INVALID_TRACE','reason':'T is not strictly increasing; negative-time runs are unsupported.'}
    if abs(trace[0,0])>th['root_tolerance']:
        return {'status':'INVALID_TRACE','reason':'Trace must include the initial event T=0.'}
    init=_initial_error_info(trace,eps,initial_errors)
    residual=trace[:,2]/eps-q(trace[:,1])-eps*eps*b(trace[:,1])
    result={'status':'ANALYZED','row_count':len(trace),'initial_errors':init,
            'T_range':[float(trace[0,0]),float(trace[-1,0])],
            's_range':[float(trace[0,1]),float(trace[-1,1])],
            'max_abs_corrected_scalar_residual':float(np.max(np.abs(residual))),
            'rms_corrected_scalar_residual':float(np.sqrt(np.mean(residual*residual))),
            's_monotone_samples':bool(np.all(np.diff(trace[:,1])>0)),
            'a_monotone_samples':bool(np.all(np.diff(trace[:,2])>0)),
            'positive_a_T_samples':bool(np.all(trace[:,5]>0)),
            'positive_s_T_samples':bool(np.all(trace[:,6]>0)), 'events':[]}
    if np.max(np.abs(trace[:,2]))>500:
        return dict(result,status='INVALID_TRACE',reason='Scalar too large for stable clock interpolation.')
    tau_T=np.exp(-trace[:,2])*trace[:,6]
    pair_T=2*np.cosh(trace[:,2])*trace[:,6]
    for c in c_values:
        pred=predictions(eps,c)
        ev=hermite_event(trace,eps*c,root_tolerance=th['root_tolerance'])
        row=dict(pred,status=ev['status'])
        if ev['status']=='FOUND' and ev['slope']>th['derivative_floor']:
            ss=interpolate_column(trace,ev,1,trace[:,6])
            aa=interpolate_column(trace,ev,2,trace[:,5])
            tau=interpolate_column(trace,ev,3,tau_T)
            paired=interpolate_column(trace,ev,4,pair_T)
            i=ev['interval']; r=ev['fraction']
            st=(1-r)*trace[i,6]+r*trace[i+1,6]
            slope_s=ev['slope']/(eps*st) if st>0 else float('nan')
            initial_s=init['values']['s']+init['values']['a_over_eps']/abs(slope_s) if slope_s>0 else float('inf')
            initial_pair=init['values']['paired_tau']+2*math.cosh(aa)*initial_s
            row.update(status='FOUND',T_event=ev['T'],s_event=ss,tau_event=tau,
                       paired_tau_event=paired,a_event=aa,a_target_residual=aa-eps*c,
                       corrected_scalar_residual=aa/eps-float(q(ss))-eps*eps*float(b(ss)),
                       s_error=ss-pred['s_pred'],paired_tau_error=paired-pred['paired_tau_pred'],
                       initial_error_s_indicator=initial_s,initial_error_paired_tau_indicator=initial_pair,
                       normalized_event_slope=slope_s,
                       s_interpolation_indicator=abs(ss-((1-r)*trace[i,1]+r*trace[i+1,1])),
                       paired_tau_interpolation_indicator=abs(paired-((1-r)*trace[i,4]+r*trace[i+1,4])))
        elif ev['status']=='FOUND':
            row.update(status='NONTRANSVERSE_OR_DECREASING',event=ev)
        else: row['event_detail']=ev
        fixed=hermite_event(trace,pred['s0'],column=1,derivative_column=6,root_tolerance=th['root_tolerance'])
        row['fixed_s_status']=fixed['status']
        if fixed['status']=='FOUND' and fixed['slope']>th['derivative_floor']:
            aa=interpolate_column(trace,fixed,2,trace[:,5])/eps
            row.update(fixed_s_a_over_eps=aa,
                       fixed_s_corrected_residual=aa-float(q(pred['s0']))-eps*eps*float(b(pred['s0'])),
                       initial_error_fixed_s_indicator=init['values']['a_over_eps']+abs(float(qprime(pred['s0'])))*init['values']['s'])
        result['events'].append(row)
    return result

def observed_order(h,values,difference_floor=1e-12,minimum_order=.05,maximum_order=12.):
    """Three-level empirical order for unequal refinement ratios, without fits."""
    h=np.asarray(h,float); v=np.asarray(values,float)
    if len(h)!=3 or len(v)!=3 or not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)) or not np.all(h>0) or not np.all(np.diff(h)<0):
        return {'status':'NEED_THREE_DISTINCT_ORDERED_LEVELS'}
    d0=v[0]-v[1]; d1=v[1]-v[2]
    out={'coarse_middle_difference':float(d0),'middle_fine_difference':float(d1)}
    if min(abs(d0),abs(d1))<=difference_floor:
        return dict(out,status='DIFFERENCES_UNRESOLVED')
    if d0*d1<=0: return dict(out,status='NONMONOTONE_DIFFERENCES')
    ratio=abs(d0/d1); r0=h[0]/h[1]; r1=h[1]/h[2]
    def model(p): return np.expm1(p*np.log(r0))/(-np.expm1(-p*np.log(r1)))
    lo=minimum_order; hi=maximum_order
    if not model(lo)<=ratio<=model(hi): return dict(out,status='ORDER_OUTSIDE_CONFIGURED_RANGE',ratio=float(ratio))
    for _ in range(80):
        mid=(lo+hi)/2
        if model(mid)<ratio: lo=mid
        else: hi=mid
    return dict(out,status='EMPIRICAL_ORDER',order=float((lo+hi)/2),ratio=float(ratio))

def _read_json(p):
    try:
        obj=json.loads(p.read_text())
        if not isinstance(obj,dict): raise ValueError('Expected JSON object.')
        return obj,None
    except (OSError,ValueError) as e: return None,str(e)

def _discover(run):
    expected={}
    manifest=run/'input/configs/tasks.list'
    if manifest.is_file():
        for i,line in enumerate(manifest.read_text().splitlines()):
            if line.strip(): expected[run/f'task_{i:03d}']=run/'input'/line.strip()
        # The frozen production manifest is authoritative. In particular,
        # smoke/{flat,nonlinear} outputs are never production tasks, and a
        # missing expected directory must remain visible in the report.
        return sorted(expected),expected
    # Standalone campaigns use only immediate task_* children. Do not
    # recursively search arbitrary result/config files: that mixes smoke
    # tests, inputs, nested archives, and previous analyses with production.
    dirs=sorted(p for p in run.glob('task_*') if p.is_dir())
    if dirs: return dirs,expected
    # A single-case invocation may point directly at its result directory.
    if any((run/name).is_file() for name in ('evolution.npz','summary.json','config.json')):
        return [run],expected
    return [],expected

def _configuration(task,summary,fallback):
    local,err=_read_json(task/'config.json')
    embedded=summary.get('config') if summary and isinstance(summary.get('config'),dict) else {}
    if local is None and fallback: local,err=_read_json(fallback)
    alias_conflicts={}
    def canonical(source,name):
        source=dict(source or {})
        if 'eps' in source and 'epsilon' in source and source['eps']!=source['epsilon']:
            alias_conflicts[name+'_epsilon_alias']=[source['eps'],source['epsilon']]
        if 'eps' not in source and 'epsilon' in source: source['eps']=source['epsilon']
        return source
    embedded=canonical(embedded,'embedded'); local=canonical(local,'local')
    cfg=dict(embedded); cfg.update(local)
    conflicts={k:[embedded[k],cfg[k]] for k in ('eps','N','L','cfl','order','nb','tag') if k in embedded and k in cfg and embedded[k]!=cfg[k]}
    conflicts.update(alias_conflicts)
    return cfg,conflicts,err

def _grid_info(z,cfg):
    out={}
    for key in ('x','y'):
        if key in z:
            arr=np.asarray(z[key],float)
            if arr.ndim==1 and len(arr)>1 and np.all(np.diff(arr)>0):
                delta=np.diff(arr); out[key+'_count']=len(arr)
                out[key+'_spacing']=float(np.mean(delta))
                out[key+'_uniform']=bool(np.allclose(delta,np.mean(delta),rtol=1e-7,atol=1e-12))
    spacings=[out[k+'_spacing'] for k in ('x','y') if k+'_spacing' in out and out.get(k+'_uniform')]
    if len(spacings)==2: out['h']=max(spacings); out['source']='npz_coordinates'
    elif isinstance(cfg.get('N'),(float,int)) and cfg['N']>1 and 'L' in cfg:
        out['h']=2*float(cfg['L'])/(int(cfg['N'])-1); out['source']='config_assumes_inclusive_uniform_grid'
    return out

def _metadata(cfg): return {k:cfg.get(k) for k in ('eps','N','L','cfl','order','nb','tag')}
def _key_value(v): return json.dumps(v,sort_keys=True,separators=(',',':'))

def _refinement_groups(tasks,th):
    available=[t for t in tasks if t.get('status')=='ANALYZED' and t.get('trace_analysis',{}).get('status')=='ANALYZED' and t.get('grid',{}).get('h')]
    groups=[]
    for kind in ('coupled_space_time','temporal'):
        buckets={}
        fixed=['eps','L','order','nb']+(['cfl'] if kind=='coupled_space_time' else ['N'])
        if th['strict_tag_grouping']: fixed.append('tag')
        for task in available:
            cfg=task['config']
            if any(cfg.get(k) is None for k in fixed): continue
            resolution=task['grid']['h'] if kind=='coupled_space_time' else cfg.get('cfl')
            if not isinstance(resolution,(float,int)) or resolution<=0: continue
            key=tuple(_key_value(cfg.get(k)) for k in fixed)
            buckets.setdefault(key,[]).append((float(resolution),task))
        for key,entries in buckets.items():
            entries.sort(key=lambda z:z[0],reverse=True)
            levels={}
            for h,task in entries: levels.setdefault(h,[]).append(task)
            if len(levels)<2: continue
            group={'kind':kind,'resolution_parameter':'h' if kind=='coupled_space_time' else 'cfl (time-step scaling parameter at fixed grid)',
                   'fixed_config':dict(zip(fixed,[json.loads(v) for v in key])),
                   'task_ids':[t['task'] for _,t in entries],'levels':len(levels),
                   'duplicate_resolutions':{str(k):[t['task'] for t in v] for k,v in levels.items() if len(v)>1},'comparisons':[]}
            if group['duplicate_resolutions']:
                group['status']='AMBIGUOUS_REPLICATES_NOT_SILENTLY_SELECTED'; groups.append(group); continue
            ordered=[(h,ts[0]) for h,ts in sorted(levels.items(),reverse=True)]
            group['status']='REFINEMENT_DIAGNOSTICS'
            for c in th['labels']:
                for observable,error_key,init_key,interp_key in (
                    ('s_event','s_error','initial_error_s_indicator','s_interpolation_indicator'),
                    ('paired_tau_event','paired_tau_error','initial_error_paired_tau_indicator','paired_tau_interpolation_indicator'),
                    ('fixed_s_a_over_eps','fixed_s_corrected_residual','initial_error_fixed_s_indicator',None)):
                    selected=[]
                    for h,task in ordered:
                        ev=next((r for r in task['trace_analysis']['events'] if abs(r['c']-c)<1e-12),None)
                        if ev and observable in ev: selected.append((h,task,ev))
                    if len(selected)<2: continue
                    hmid,tmid,emid=selected[-2]; hfine,tfine,efine=selected[-1]
                    diff=abs(efine[observable]-emid[observable])
                    init=efine.get(init_key,0.)+emid.get(init_key,0.)
                    interp=(efine.get(interp_key,0.)+emid.get(interp_key,0.)) if interp_key and th['include_interpolation_indicator'] else 0.
                    indicator=diff+init+interp
                    entry={'c':c,'observable':observable,'corrected_residual_key':error_key,
                           'finest_task':tfine['task'],'next_task':tmid['task'],
                           'finest_corrected_residual':efine.get(error_key),
                           'last_refinement_difference':diff,'initial_error_contribution':init,
                           'interpolation_contribution':interp,'conservative_difference_indicator':indicator,
                           'initial_error_information_complete':tfine['trace_analysis']['initial_errors']['all_three_provided'] and tmid['trace_analysis']['initial_errors']['all_three_provided'],
                           'uncertainty_scope':'Difference-plus-initial-error indicator; not a certified error bound.'}
                    if len(selected)>=3:
                        triple=selected[-3:]
                        entry['observed_order']=observed_order([v[0] for v in triple],[v[2][observable] for v in triple],
                                   th['difference_floor'],th['minimum_order'],th['maximum_order'])
                    else: entry['observed_order']={'status':'NEED_THREE_LEVELS'}
                    if observable in ('s_event','paired_tau_event'):
                        leading=abs(efine['leading_s_delay' if observable=='s_event' else 'leading_paired_tau_correction'])
                        entry['target_error_indicator']=th['leading_delay_error_fraction']*leading
                        entry['indicator_below_target']=indicator<entry['target_error_indicator']
                    group['comparisons'].append(entry)
            groups.append(group)
    return groups

def _scaling(tasks,groups,th):
    # Require both a spatial/coupled and an independent temporal refinement
    # diagnostic for a task before interpreting amplitude scaling.
    uncertainty={}
    for g in groups:
        for row in g.get('comparisons',[]):
            key=(row['finest_task'],row['c'],row['corrected_residual_key'])
            uncertainty.setdefault(key,{})[g['kind']]=row
    buckets={}
    fixed=['N','L','cfl','order','nb']+(['tag'] if th['strict_tag_grouping'] else [])
    for task in tasks:
        if task.get('status')!='ANALYZED' or task.get('trace_analysis',{}).get('status')!='ANALYZED': continue
        cfg=task['config']
        if any(cfg.get(k) is None for k in fixed): continue
        key=tuple(_key_value(cfg.get(k)) for k in fixed)
        buckets.setdefault(key,[]).append(task)
    out=[]
    for key,entries in buckets.items():
        eps_values={float(t['config']['eps']) for t in entries}
        if len(eps_values)<3: continue
        for c in th['labels']:
            for residual_key in ('s_error','paired_tau_error','fixed_s_corrected_residual'):
                row={'fixed_config':dict(zip(fixed,[json.loads(v) for v in key])),'c':c,'residual':residual_key,'status':'NOT_ESTABLISHED','samples':[]}
                reasons=[]
                for task in entries:
                    ev=next((r for r in task['trace_analysis']['events'] if abs(r['c']-c)<1e-12),{})
                    if residual_key not in ev: reasons.append('missing event/residual in '+task['task']); continue
                    uu=uncertainty.get((task['task'],c,residual_key),{})
                    good=len(uu)==2 and all(v['observed_order']['status']=='EMPIRICAL_ORDER' and v['initial_error_information_complete'] for v in uu.values())
                    noise=sum(v['conservative_difference_indicator'] for v in uu.values()) if len(uu)==2 else None
                    rr=ev[residual_key]
                    if not good: reasons.append('insufficient independent refinements/initial errors for '+task['task'])
                    if noise is None or abs(rr)<=max(th['difference_floor'],th['scaling_noise_factor']*noise):
                        reasons.append('corrected residual not separated from numerical indicators for '+task['task'])
                    row['samples'].append({'task':task['task'],'eps':float(task['config']['eps']),'residual':rr,'numerical_indicator':noise})
                samples=sorted(row['samples'],key=lambda r:r['eps'],reverse=True)
                if len({v['eps'] for v in samples})!=len(samples): reasons.append('duplicate amplitudes at the same resolution')
                if len(samples)<3: reasons.append('fewer than three usable amplitudes')
                if samples and not(all(v['residual']>0 for v in samples) or all(v['residual']<0 for v in samples)):
                    reasons.append('residual signs change')
                if not reasons:
                    row['status']='EMPIRICAL_SCALING_ONLY'
                    row['adjacent_log_slopes']=[math.log(abs(a['residual']/bb['residual']))/math.log(a['eps']/bb['eps']) for a,bb in zip(samples[:-1],samples[1:])]
                    row['expected_power']=4
                else: row['reasons']=sorted(set(reasons))
                row['samples']=samples; out.append(row)
    return out

def analyze_campaign(run_dir,thresholds=None):
    run=Path(run_dir).resolve(); th=settings(thresholds)
    tasks=[]; directories,expected=_discover(run)
    for task in directories:
        summary,summary_error=_read_json(task/'summary.json')
        cfg,conflicts,config_error=_configuration(task,summary,expected.get(task))
        rec={'task':str(task.relative_to(run)) or '.', 'config':_metadata(cfg),
             'config_conflicts':conflicts,'summary_read_error':summary_error,'config_read_error':config_error}
        wrapper,_=_read_json(task/'wrapper_status.json')
        rec['process_status']=wrapper.get('status') if wrapper else None
        rec['summary_status']=summary.get('status') if summary else None
        try:
            direction=float(cfg.get('direction',1))
            if not np.isfinite(direction): raise ValueError('direction must be finite.')
            if direction<0:
                rec['status']='SKIPPED_NEGATIVE_TIME_NOT_IMPLEMENTED'; tasks.append(rec); continue
            eps=float(cfg['eps'])
            if eps==0:
                rec['status']='SKIPPED_DEGENERATE_ZERO_FIELD'
                rec['reason']='Flat zero-field control is audited separately; normalized scalar/event predictions are degenerate.'
                tasks.append(rec); continue
            if not np.isfinite(eps) or eps<=0: raise ValueError('Require finite positive epsilon.')
            cfg['eps']=eps; rec['config']['eps']=eps
            with (task/'evolution.npz').open('rb') as raw:
                with np.load(raw,allow_pickle=False) as z:
                    if 'center_trace' not in z: raise ValueError('Missing center_trace key.')
                    trace=np.asarray(z['center_trace'],float)
                    rec['grid']=_grid_info(z,cfg)
                    for amplitude_key in ('eps','epsilon'):
                        if amplitude_key in z:
                            stored=np.asarray(z[amplitude_key])
                            if stored.size==1:
                                stored=float(stored.reshape(-1)[0])
                                if not np.isfinite(stored) or not np.isclose(stored,eps,rtol=1e-12,atol=0):
                                    conflicts['npz_'+amplitude_key]=[eps,stored]
            initial=(summary or {}).get('initial_errors')
            numerics=(summary or {}).get('numerics',{})
            if not isinstance(numerics,dict):
                rec['metadata_warning']='summary.numerics is not an object; ignored.'
                numerics={}
            if initial is None: initial=numerics.get('initial_errors')
            rec['trace_analysis']=analyze_trace(trace,eps,th['labels'],initial,th)
            rec['status']=rec['trace_analysis']['status']
            if conflicts: rec['status']='CONFIG_CONFLICT_NOT_REFINEMENT_ELIGIBLE'
            elif summary is None: rec['status']='MISSING_SUMMARY_NOT_REFINEMENT_ELIGIBLE'
            elif rec['process_status']=='FAILED' or str(rec['summary_status']).upper() in ('FAILED','ERROR','INVALID','CRASHED'):
                rec['status']='FAILED_PROCESS_NOT_REFINEMENT_ELIGIBLE'
        except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile,np.linalg.LinAlgError) as exc:
            rec['status']='UNAVAILABLE_OR_INVALID'; rec['error']=str(exc)
        tasks.append(rec)
    groups=_refinement_groups(tasks,th)
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),'run_dir':str(run),
            'scientific_status':'DIAGNOSTIC_ONLY','acceptance':'NOT_ESTABLISHED',
            'analytic_coefficients_fitted':False,'thresholds':th,
            'protocol':'Frozen Phase 4 q,b,D,Q2; positive-time branch only.',
            'limitations':['Refinement differences plus initial errors are indicators, not rigorous error bounds.',
                          'Same-CFL grid refinement couples spatial and temporal error; no pure spatial order is claimed.',
                          'Topology, boundary causality, full Einstein constraints, and common-domain validity require separate audits.',
                          'Missing initial-error contributions prevent amplitude-scaling conclusions.',
                          'tag is retained as metadata; set strict_tag_grouping=true if tags distinguish physical/numerical methods.'],
            'task_count':len(tasks),'tasks':tasks,'refinement_groups':groups,
            'corrected_residual_scaling':_scaling(tasks,groups,th)}
    return result

def _clean(value):
    if isinstance(value,dict): return {str(k):_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_clean(v) for v in value]
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def write_analysis(run_dir,result):
    run=Path(run_dir); run.mkdir(parents=True,exist_ok=True)
    result=_clean(result)
    (run/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[]
    for task in result['tasks']:
        for event in task.get('trace_analysis',{}).get('events',[]):
            flat={'task':task['task'],**task['config']}
            flat.update({k:v for k,v in event.items() if not isinstance(v,(dict,list))})
            rows.append(flat)
    fields=['task','eps','N','L','cfl','order','nb','tag','c','status']
    fields+=sorted({k for r in rows for k in r}-set(fields))
    with (run/'event_errors.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=fields); out.writeheader(); out.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--thresholds',help='JSON object overriding DEFAULTS; unknown keys rejected.')
    args=p.parse_args()
    overrides=json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    result=analyze_campaign(args.run_dir,overrides); write_analysis(args.run_dir,result)
    print(json.dumps({'scientific_status':result['scientific_status'],'tasks':result['task_count'],
                      'analyzed':sum(t['status']=='ANALYZED' for t in result['tasks']),
                      'refinement_groups':len(result['refinement_groups']),
                      'analysis_file':str(Path(args.run_dir)/'analysis.json')},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
