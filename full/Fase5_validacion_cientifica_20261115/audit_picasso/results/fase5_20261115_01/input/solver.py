#!/usr/bin/env python3
"""Bounded 2+1 vacuum-reduction experiment. Production dependency: NumPy."""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from gh_core import rhs,constraints,hermite,crossing_fraction,interpolate

def initial(config):
    N=int(config['N']);L=float(config['L']);eps=float(config['epsilon'])
    assert N%2==1 and N>=17
    x=np.linspace(-L,L,N);X,Y=np.meshgrid(x,x,indexing='ij');r=np.hypot(X,Y)
    data=np.load(ROOT/'data/radial_profiles.npz');ids=np.where(abs(data['epsilon']-eps)<1e-12)[0]
    if not len(ids):raise ValueError('epsilon not present in precomputed initial data')
    i=int(ids[0]);rr=data['r'];dr=rr[1]-rr[0];k=np.floor(r/dr).astype(int)
    if np.max(k)>=len(rr)-1:raise ValueError('domain outside radial initial-data extension')
    lam=hermite(data['lam'][i,k],data['lam'][i,k+1],data['lam_r'][i,k],data['lam_r'][i,k+1],(r-rr[k])/dr,dr)
    state=np.zeros((15,N,N));state[0]=-1.;state[3]=state[5]=np.exp(2*lam)
    P=1-r*r;state[10]=-eps*eps*P*P;state[12]=eps*X*Y;state[13]=eps*np.exp(-2*lam)*P
    nb=int(config.get('nb',256));theta=np.arange(nb)*2*np.pi/nb
    boundary=np.column_stack((np.cos(theta),np.sin(theta),np.zeros(nb))) if eps else None
    return x,state,boundary,theta

def evolve(config,out):
    start=time.perf_counter();x,Y,B,theta=initial(config);h=x[1]-x[0];eps=float(config['epsilon'])
    order=int(config.get('order',6));Tend=float(config.get('Tmax',.26));direction=int(config.get('direction',1))
    assert direction in (-1,1)
    nsteps=int(np.ceil(Tend/(float(config.get('cfl',.15))*h)));dt=direction*Tend/nsteps
    targets=np.array(config.get('slice_targets',[0,.05,.1,.15,.2,.24]),float)*direction
    ns=len(targets);N=len(x);nb=len(theta);ci=N//2
    phis=np.full((ns,N,N),np.nan);Ts=phis.copy();bxy=np.full((ns,nb,2),np.nan);bphi=np.full((ns,nb),np.nan)
    btimes=bphi.copy();brecon=bphi.copy();slice_residual=0.;bslice_residual=0.
    xx,yy=np.meshgrid(x,x,indexing='ij');mask=(xx*xx+yy*yy)<=float(config.get('diagnostic_radius',1.1))**2
    clocks=np.zeros(3);T=0.;trace=[];diagnostics=[];eikonal_errors=[];boundary_extrema={'max_V_norm':-np.inf,'min_spatial_gradient2':np.inf,'max_eikonal_material_defect':0.}
    stride=max(1,int(config.get('diagnostic_stride',max(1,nsteps//8))))
    def call(Z,BB):
        k,kb,aux=rhs(Z,h,order,BB,x,x)
        if not np.all(np.isfinite(k)) or np.any(aux['inv'][...,0,0]>=0) or np.any(Z[3]<=0) or np.any(Z[3]*Z[5]-Z[4]**2<=0):raise RuntimeError('nonfinite RHS or loss of admissible signature')
        if np.any(k[14]<=0):raise RuntimeError('Gaussian slicing ceased to be a graph')
        return k,kb,aux
    k,kb,aux=call(Y,B)
    if eps:
        for key,val in aux['boundary_checks'].items():boundary_extrema[key]=val
    def append_trace():trace.append([T,*[clocks[0],Y[12,ci,ci],clocks[1],clocks[2],Y[13,ci,ci],aux['clocks'][0]]])
    append_trace()
    diagnostics.append({'T':T,**constraints(Y,aux,h,order,mask)})
    for j,targ in enumerate(targets):
        if targ==0:
            phis[j]=Y[12];Ts[j]=0.
            bxy[j]=np.column_stack((np.cos(theta),np.sin(theta)));bphi[j]=eps*np.cos(theta)*np.sin(theta);btimes[j]=0.;brecon[j]=0.
    for step in range(nsteps):
        prev=Y;oldB=B;oldk=k;oldkb=kb;oldaux=aux;oldT=T;oldclocks=clocks.copy()
        def addB(B0,b,d):return None if B0 is None else B0+d*b
        k2,b2,a2=call(Y+.5*dt*k,addB(B,kb,.5*dt))
        k3,b3,a3=call(Y+.5*dt*k2,addB(B,b2,.5*dt))
        k4,b4,a4=call(Y+dt*k3,addB(B,b3,dt))
        Y=Y+dt/6*(k+2*k2+2*k3+k4)
        if B is not None:B=B+dt/6*(kb+2*b2+2*b3+b4)
        clocks=clocks+dt/6*(aux['clocks']+2*a2['clocks']+2*a3['clocks']+a4['clocks'])
        T=direction*Tend*(step+1)/nsteps;k,kb,aux=call(Y,B);append_trace()
        # Independent midpoint time difference: not substitution of eikonal RHS.
        smid=.5*(prev[14]+Y[14]);mid=.5*(prev+Y);mid[14]=smid
        from gh_core import diff,inverse3,full
        ds=np.stack(((Y[14]-prev[14])/dt,diff(smid,0,h,1,order),diff(smid,1,h,1,order)),axis=-1)
        invmid=inverse3(full(mid[:6]));er=np.einsum('...ab,...a,...b->...',invmid,ds,ds)+1
        eikonal_errors.append(float(np.max(abs(er[mask]))))
        if eps:
            bc=aux['boundary_checks'];boundary_extrema['max_V_norm']=max(boundary_extrema['max_V_norm'],bc['max_V_norm'])
            boundary_extrema['min_spatial_gradient2']=min(boundary_extrema['min_spatial_gradient2'],bc['min_spatial_gradient2'])
            boundary_extrema['max_eikonal_material_defect']=max(boundary_extrema['max_eikonal_material_defect'],bc['max_eikonal_material_defect'])
        for j,targ in enumerate(targets):
            cross=(direction*(prev[14]-targ)<0)&(direction*(Y[14]-targ)>=0)&~np.isfinite(phis[j])
            if np.any(cross):
                z=crossing_fraction(prev[14][cross],Y[14][cross],oldk[14][cross],k[14][cross],targ,dt)
                phis[j][cross]=hermite(prev[12][cross],Y[12][cross],prev[13][cross],Y[13][cross],z,dt)
                Ts[j][cross]=oldT+z*dt
                slice_residual=max(slice_residual,float(np.max(abs(hermite(prev[14][cross],Y[14][cross],oldk[14][cross],k[14][cross],z,dt)-targ))))
            if B is not None:
                crossb=(direction*(oldB[:,2]-targ)<0)&(direction*(B[:,2]-targ)>=0)&~np.isfinite(bphi[j])
                if np.any(crossb):
                    z=crossing_fraction(oldB[crossb,2],B[crossb,2],oldkb[crossb,2],kb[crossb,2],targ,dt)
                    pts=hermite(oldB[crossb,:2],B[crossb,:2],oldkb[crossb,:2],kb[crossb,:2],z[:,None],dt)
                    bxy[j,crossb]=pts;btimes[j,crossb]=oldT+z*dt
                    # Interpolate the spacetime scalar independently at the recorded material slice.
                    vals=[interpolate(arr,x,x,pts) for arr in (prev[12],Y[12],prev[13],Y[13])]
                    bphi[j,crossb]=hermite(*vals,z,dt)
                    bs=hermite(oldB[crossb,2],B[crossb,2],oldkb[crossb,2],kb[crossb,2],z,dt)
                    brecon[j,crossb]=bs-targ;bslice_residual=max(bslice_residual,float(np.max(abs(bs-targ))))
        if (step+1)%stride==0 or step+1==nsteps:
            diagnostics.append({'T':T,**constraints(Y,aux,h,order,mask)})
    trace=np.array(trace)
    np.savez_compressed(out/'evolution.npz',x=x,y=x,epsilon=eps,center_trace=trace,
        center_trace_columns=np.array(['T','s','a','tau','paired_tau','a_T','s_T']),gaussian_slices_targets=targets,
        phi_slices=phis,T_slices=Ts,boundary_xy=bxy,boundary_theta=theta,boundary_labels=eps*np.cos(theta)*np.sin(theta),
        boundary_phi=bphi,boundary_T=btimes,boundary_s_reconstruction_error=brecon,
        final_state=Y,final_boundary=np.empty((0,3)) if B is None else B)
    report={'status':'SUCCESS','config':config,'nsteps':nsteps,'dt':dt,'h':h,'runtime_seconds':time.perf_counter()-start,
      'diagnostics':diagnostics,'max_midpoint_eikonal_residual':max(eikonal_errors),'slice_inverse_residual':slice_residual,
      'boundary_slice_inverse_residual':bslice_residual,'boundary':boundary_extrema if eps else {},
      'center_gaussian_eikonal_discrepancy':float(abs(Y[14,ci,ci]-clocks[0])),
      'slices_complete_in_diagnostic_disk':[bool(np.all(np.isfinite(p[mask]))) for p in phis],
      'boundary_slices_complete':[bool(np.all(np.isfinite(p))) for p in bphi] if eps else [],
      'max_material_label_error':float(np.nanmax(abs(bphi-eps*np.cos(theta)*np.sin(theta)))) if eps else 0.}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','nsteps','runtime_seconds','max_material_label_error')}),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--output',required=True);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    try:
        if args.self_test:
            import subprocess
            subprocess.run([sys.executable,str(ROOT/'code/self_test.py'),'--output',str(out)],check=True)
        else:
            config=json.loads(Path(args.config).read_text());(out/'config.json').write_text(json.dumps(config,indent=2)+'\n');report=evolve(config,out)
            if config.get('trace_slices',False) and config['epsilon']>0:
                from audit_slices import audit
                audit_report=audit(out/'evolution.npz')
                (out/'slice_audit.json').write_text(json.dumps(audit_report,indent=2,allow_nan=False)+'\n')
                report['connectivity_audit']={'exits':sum(v['traced_exits'] for v in audit_report['slices']),
                    'mismatched_pairs':sum(v['mismatched_pairs'] for v in audit_report['slices']),
                    'unresolved_eligible_cases':sum(v['unresolved_eligible_cases'] for v in audit_report['slices'])}
                (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    except Exception as exc:
        (out/'summary.json').write_text(json.dumps({'status':'FAILED','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
