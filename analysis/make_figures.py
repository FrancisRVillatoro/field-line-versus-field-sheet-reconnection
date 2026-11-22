"""Regenerate manuscript plots and tables from the preserved validation data."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.ticker import NullFormatter
from scipy.interpolate import RectBivariateSpline
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'data'
F=ROOT/'figures';F.mkdir(parents=True,exist_ok=True)
D=ROOT/'data';D.mkdir(exist_ok=True)
main=json.loads((V/'scientific_validation.json').read_text())
small=json.loads((V/'small_amplitude_validation.json').read_text())
obs=sorted(main['observables']+small['observables'],key=lambda v:(v['observable'],v['c'],v['epsilon']))
keys=['epsilon','c','observable','observed','residual','sensitivity_indicator']
with (D/'observables.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(obs)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11,'legend.fontsize':8,'mathtext.fontset':'dejavuserif','savefig.bbox':'tight'})
colors=['#2166ac','#b2182b','#1b7837']
names=['s_event','paired_tau_event','fixed_s_a_over_eps']
labels=[r'$s_*$',r'$\Delta\tau_*$',r'$a(s_0)/\varepsilon$']
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
for name,label,color,mark in zip(names,labels,colors,['o','s','^']):
    rows=[r for r in obs if r['observable']==name and r['c']==.12]
    e=np.array([r['epsilon'] for r in rows]);res=np.abs([r['residual'] for r in rows]);U=np.array([r['sensitivity_indicator'] for r in rows])
    axs[0].loglog(e,res,'-'+mark,color=color,label=label,ms=4)
    axs[0].loglog(e,U,':',color=color,lw=.8,alpha=.65)
    axs[1].semilogx(np.sqrt(e[:-1]*e[1:]),np.diff(np.log(res))/np.diff(np.log(e)),'-'+mark,color=color,label=label,ms=4)
axs[0].set(xlabel=r'Amplitude $\varepsilon$',ylabel=r'Absolute residual $|E_O|$',title=r'(a) Fixed material label $c=0.12$')
axs[0].text(.011,8e-12,'Dotted: sensitivity indicators',fontsize=8)
axs[0].legend(loc='upper left')
axs[1].axhline(4,color='.4',ls='--',lw=1)
axs[1].set(xlabel=r'Geometric mean of adjacent amplitudes',ylabel=r'Adjacent-amplitude exponent',title='(b) Approach to fourth power')
axs[1].set_ylim(3.945,4.025)
axs[1].set_xticks([.015,.03,.06,.10],labels=['0.015','0.03','0.06','0.10'])
axs[1].xaxis.set_minor_formatter(NullFormatter())
axs[1].set_xlabel('Geometric mean amplitude')
axs[1].set_ylabel(r'Amplitude exponent $p_{\rm eff}$')
for ax in axs:ax.grid(alpha=.2,which='both')
fig.savefig(F/'amplitude_scaling.pdf');fig.savefig(F/'amplitude_scaling.png',dpi=180);plt.close(fig)

data=np.load(D/'connectivity_figure_data.npz');eps=float(data['epsilon']);c=float(data['c'])
fig,axs=plt.subplots(2,2,figsize=(7,6.5),layout='constrained')
for col,part in enumerate(['before','after']):
    xx=np.linspace(-1.1,1.1,801);X,Y=np.meshgrid(xx,xx,indexing='ij')
    spline=RectBivariateSpline(data['x'],data['y'],data['phi_'+part]/eps)
    z=spline(xx,xx);b=data['boundary_'+part];bc=np.vstack([b,b[:1]])
    mask=MPath(bc).contains_points(np.stack([X.ravel(),Y.ravel()],axis=1)).reshape(X.shape)
    z=np.ma.array(z,mask=~mask);ac=float(spline(0,0)[0,0]);s0=float(data['s'][col])
    for row in range(2):
        ax=axs[row,col]
        ax.contour(X,Y,z,levels=[c],colors=[colors[0]],linewidths=2)
        ax.contour(X,Y,z,levels=[ac],colors=['.45'],linestyles=['--'],linewidths=.9)
        ax.plot(bc[:,0],bc[:,1],color='k',lw=1)
        ax.plot(0,0,'k+',ms=6)
        ax.set_aspect('equal');lim=1.1 if row==0 else .22;ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel=r'$X$',ylabel=r'$Y$')
        ax.grid(alpha=.12)
        if row==0:
            theta0=np.arcsin(2*c)/2
            for name,theta in [('E',theta0),('N',np.pi/2-theta0),('W',np.pi+theta0),('S',3*np.pi/2-theta0)]:
                at=np.array([np.interp(theta,np.linspace(0,2*np.pi,len(b),endpoint=False),b[:,j]) for j in range(2)])
                ax.plot(*at,'o',color=colors[0],ms=4);ax.text(*(1.07*at),name,ha='center',va='center',fontsize=9)
            pair='N–E; S–W' if col==0 else 'N–W; S–E'
            ax.set_title(f'({"a" if col==0 else "b"}) {"Before" if col==0 else "After"}: $s={s0:.6f}$\n{pair}',fontsize=10)
        else:ax.set_title(f'({"c" if col==0 else "d"}) Central enlargement',fontsize=10)
fig.savefig(F/'connectivity_switch.pdf');fig.savefig(F/'connectivity_switch.png',dpi=180);plt.close(fig)

sel=[r for r in obs if r['c']==.12 and r['observable']=='s_event']
tau=[r for r in obs if r['c']==.12 and r['observable']=='paired_tau_event']
lines=[]
for row,tr in zip(sel,tau):
    lines.append(f"{row['epsilon']:.2f} & {row['observed']:.13f} & {tr['observed']:.13f} & ${row['residual']:.4e}$ & ${row['sensitivity_indicator']:.2e}$ \\\\")
(D/'event_table_rows.tex').write_text('\n'.join(lines)+'\n')
out={'original_observables':len(main['observables']),'small_observables':len(small['observables']),'sampled_switches':len(main['switches']),'symbolic_checks':json.loads((ROOT/'analysis'/'polynomial_checks.json').read_text())['count'],'max_original_sensitivity_indicator':max(r['sensitivity_indicator'] for r in main['observables']),'minimum_original_signal_ratio':min(r['residual_to_indicator'] for r in main['observables']),'minimum_small_signal_ratio':small['minimum_residual_to_indicator'],'max_boundary_V_norm':max(r['max_V_norm'] for r in main['controls'] if r['max_V_norm'] is not None)}
(D/'paper_numbers.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
