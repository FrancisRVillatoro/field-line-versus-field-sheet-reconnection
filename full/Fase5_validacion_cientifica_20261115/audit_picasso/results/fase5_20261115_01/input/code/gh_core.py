"""2+1 Einstein-scalar in covariant harmonic reduction, NumPy only.
R_ab - nabla_(a C_b)=2 phi_a phi_b; physical Box_g phi=0.
Coordinates (T,x,y). Gaussian clock s is a passive eikonal field.
State: six g_ab, six d_T g_ab, phi, d_T phi, s.
"""
import numpy as np
PAIRS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))

def full(v):
    a=np.empty(v.shape[1:]+(3,3),dtype=v.dtype)
    for k,(i,j) in enumerate(PAIRS):a[...,i,j]=a[...,j,i]=v[k]
    return a

def pack(a):return np.stack([a[...,i,j] for i,j in PAIRS])

def diff(a,axis,h,degree=1,order=6):
    if degree==1:
        offsets=(-3,-2,-1,1,2,3) if order==6 else (-2,-1,1,2)
        weights=(-1,9,-45,45,-9,1) if order==6 else (1,-8,8,-1)
        denom=60 if order==6 else 12
    else:
        offsets=(-3,-2,-1,0,1,2,3) if order==6 else (-2,-1,0,1,2)
        weights=(2,-27,270,-490,270,-27,2) if order==6 else (-1,16,-30,16,-1)
        denom=180 if order==6 else 12
    out=np.zeros_like(a)
    for k,w in zip(offsets,weights):out+=w*np.roll(a,-k,axis=axis)
    return out/(denom*h**degree)

def inverse3(g):
    a,b,c=g[...,0,0],g[...,0,1],g[...,0,2]
    d,e,f=g[...,1,1],g[...,1,2],g[...,2,2]
    A=d*f-e*e;B=c*e-b*f;C=b*e-c*d;D=a*f-c*c;E=b*c-a*e;F=a*d-b*b
    det=a*A+b*B+c*C
    inv=np.empty_like(g)
    for i,j,u in [(0,0,A),(0,1,B),(0,2,C),(1,1,D),(1,2,E),(2,2,F)]:inv[...,i,j]=inv[...,j,i]=u/det
    return inv

def metric_geometry(g,dg):
    inv=inverse3(g)
    L=dg.swapaxes(-3,-2)+np.moveaxis(dg,-3,-1)-dg
    Gamma=.5*np.einsum('...km,...mab->...kab',inv,L,optimize=True)
    dinv=-np.einsum('...am,...dmn,...nb->...dab',inv,dg,inv,optimize=True)
    divinv=np.einsum('...ccm->...m',dinv)
    RicQ=.5*np.einsum('...m,...mab->...ab',divinv,L,optimize=True)
    RicQ-=.5*np.einsum('...bcm,...mac->...ab',dinv,L,optimize=True)
    trG=np.einsum('...ccd->...d',Gamma)
    RicQ+=np.einsum('...d,...dab->...ab',trG,Gamma,optimize=True)
    RicQ-=np.einsum('...cbd,...dac->...ab',Gamma,Gamma,optimize=True)
    Cup=np.einsum('...ab,...kab->...k',inv,Gamma,optimize=True)
    Cdown=np.einsum('...ab,...b->...a',g,Cup,optimize=True)
    H=dg.swapaxes(-3,-2)-.5*dg
    dCQ=np.einsum('...acd,...bcd->...ab',dinv,H,optimize=True)
    Q=RicQ-.5*(dCQ+dCQ.swapaxes(-1,-2))+np.einsum('...cab,...c->...ab',Gamma,Cdown,optimize=True)
    Q=.5*(Q+Q.swapaxes(-1,-2))
    return inv,Gamma,Cup,Q

def interpolate(field,x,y,points):
    """Tensor product cubic Lagrange; field shape (nx,ny,...)."""
    hx=x[1]-x[0];hy=y[1]-y[0]
    ux=(points[:,0]-x[0])/hx;uy=(points[:,1]-y[0])/hy
    ix=np.floor(ux).astype(int);iy=np.floor(uy).astype(int)
    if np.any(ix<1)|np.any(ix>len(x)-3)|np.any(iy<1)|np.any(iy>len(y)-3):raise RuntimeError('Material boundary left interpolation interior')
    def w(t):return np.array([-t*(t-1)*(t-2)/6,(t+1)*(t-1)*(t-2)/2,-(t+1)*t*(t-2)/2,(t+1)*t*(t-1)/6])
    wx=w(ux-ix);wy=w(uy-iy)
    out=np.zeros((len(points),)+field.shape[2:],dtype=field.dtype)
    ex=(None,)*(field.ndim-2)
    for a in range(4):
        for b in range(4):out+=(wx[a]*wy[b])[(slice(None),)+ex]*field[ix+a-1,iy+b-1]
    return out

def rhs(Y,h,order=6,boundary=None,x=None,y=None,freeze_width=4):
    dx=diff(Y,1,h,1,order);dy=diff(Y,2,h,1,order)
    g=full(Y[:6]);P=full(Y[6:12])
    dg=np.stack((P,full(dx[:6]),full(dy[:6])),axis=-3)
    inv,Gamma,Cup,Q=metric_geometry(g,dg)
    dphi=np.stack((Y[13],dx[12],dy[12]),axis=-1)
    indices=[0,1,2,3,4,5,12]
    zz=Y[indices];xx=diff(zz,1,h,2,order);yy=diff(zz,2,h,2,order);xy=diff(dy[indices],1,h,1,order)
    wave=2*inv[...,0,1]*dx[6:12]+2*inv[...,0,2]*dy[6:12]+inv[...,1,1]*xx[:6]+2*inv[...,1,2]*xy[:6]+inv[...,2,2]*yy[:6]
    matter=np.einsum('...a,...b->...ab',dphi,dphi)
    out=np.empty_like(Y);out[:6]=Y[6:12]
    out[6:12]=(-wave+2*pack(Q)-4*pack(matter))/inv[...,0,0]
    out[12]=Y[13]
    out[13]=(-2*inv[...,0,1]*dx[13]-2*inv[...,0,2]*dy[13]-inv[...,1,1]*xx[6]-2*inv[...,1,2]*xy[6]-inv[...,2,2]*yy[6]+np.einsum('...a,...a->...',Cup,dphi))/inv[...,0,0]
    alpha=1/np.sqrt(-inv[...,0,0]);beta=-inv[...,0,1:]/inv[...,0,0,None]
    gaminv=inv[...,1:,1:]+np.einsum('...i,...j->...ij',beta,beta)/alpha[...,None,None]**2
    ds_space=np.stack((dx[14],dy[14]),axis=-1)
    out[14]=np.einsum('...i,...i->...',beta,ds_space)+alpha*np.sqrt(1+np.einsum('...ij,...i,...j->...',gaminv,ds_space,ds_space))
    if freeze_width:
        fw=freeze_width
        # Buffer follows its initial tangent: metric/scalar velocities fixed.
        for sl in [(slice(None,fw),slice(None)),(slice(-fw,None),slice(None)),(slice(None),slice(None,fw)),(slice(None),slice(-fw,None))]:
            out[(slice(6,12),)+sl]=0.;out[(13,)+sl]=0.;out[(14,)+sl]=1.
    ci=Y.shape[1]//2;cj=Y.shape[2]//2
    central_lapse=np.sqrt(-g[ci,cj,0,0]);a=Y[12,ci,cj]
    clocks=np.array([central_lapse,np.exp(-a)*central_lapse,2*np.cosh(a)*central_lapse])
    bdot=None;boundary_checks={}
    if boundary is not None:
        pts=boundary[:,:2]
        gp=interpolate(g,x,y,pts);ip=inverse3(gp)
        fp=interpolate(dphi,x,y,pts)
        ds=np.concatenate((out[14,...,None],ds_space),axis=-1)
        sp=interpolate(ds,x,y,pts)
        n=-np.einsum('...ab,...b->...a',ip,sp)
        npf=np.einsum('...a,...a->...',n,fp)
        D=np.einsum('...ab,...b->...a',ip,fp)+n*npf[:,None]
        D2=np.einsum('...a,...a->...',fp,D)
        if np.any(D2<=1e-14):raise RuntimeError('Boundary scalar gradient degenerate')
        V=n-D*(npf/D2)[:,None]
        if np.any(V[:,0]<=0):raise RuntimeError('Boundary not transverse to harmonic time')
        bdot=np.column_stack((V[:,1]/V[:,0],V[:,2]/V[:,0],1/V[:,0]))
        vnorm=np.einsum('...ab,...a,...b->...',gp,V,V)
        if np.any(vnorm>=0):raise RuntimeError('Material boundary ceased to be timelike')
        boundary_checks={'max_V_norm':float(np.max(vnorm)),'min_spatial_gradient2':float(np.min(D2)),
                         'max_eikonal_material_defect':float(np.max(np.abs(interpolate(Y[14],x,y,pts)-boundary[:,2])))}
    aux={'g':g,'dg':dg,'inv':inv,'Cup':Cup,'Gamma':Gamma,'alpha':alpha,'beta':beta,'gaminv':gaminv,'dphi':dphi,'dx':dx,'dy':dy,'clocks':clocks,'boundary_checks':boundary_checks}
    return out,bdot,aux

def constraints(Y,aux,h,order,mask):
    g=aux['g'];gam=g[...,1:,1:];gi=aux['gaminv'];dg=aux['dg'];alpha=aux['alpha'];beta=aux['beta']
    dgam=dg[...,1:,1:,1:]
    L=dgam.swapaxes(-3,-2)+np.moveaxis(dgam,-3,-1)-dgam
    G=.5*np.einsum('...km,...mab->...kab',gi,L,optimize=True)
    dG=np.stack((diff(G,0,h,1,order),diff(G,1,h,1,order)),axis=-4)
    Ric=np.einsum('...kkij->...ij',dG)-np.einsum('...jkik->...ij',dG)
    trace=np.einsum('...kkl->...l',G)
    Ric+=np.einsum('...l,...lij->...ij',trace,G)-np.einsum('...kjl,...lik->...ij',G,G)
    R=np.einsum('...ij,...ij->...',gi,Ric)
    bcov=g[...,0,1:]
    db=np.stack((aux['dx'][1:3],aux['dy'][1:3]),axis=-1)
    # db after stack has (component,nx,ny,derivative); convert to (...,derivative,component).
    db=np.moveaxis(db,0,-1)
    Db=db-np.einsum('...kij,...k->...ij',G,bcov)
    K=(-full(Y[6:12])[...,1:,1:]+Db+Db.swapaxes(-1,-2))/(2*alpha[...,None,None])
    Km=np.einsum('...ij,...jk->...ik',gi,K)
    tr=np.trace(Km,axis1=-2,axis2=-1);norm=np.einsum('...ij,...ji->...',Km,Km)
    grad=aux['dphi'][...,1:];Pi=(Y[13]-np.einsum('...i,...i->...',beta,grad))/alpha
    grad2=np.einsum('...ij,...i,...j->...',gi,grad,grad)
    H=R+tr*tr-norm-2*(Pi*Pi+grad2)
    dKm=np.stack((diff(Km,0,h,1,order),diff(Km,1,h,1,order)),axis=-3)
    M=np.einsum('...jji->...i',dKm)+np.einsum('...jjk,...ki->...i',G,Km)-np.einsum('...kji,...jk->...i',G,Km)
    M-=np.stack((diff(tr,0,h,1,order),diff(tr,1,h,1,order)),axis=-1)
    M+=2*Pi[...,None]*grad
    return {'hamiltonian_linf':float(np.max(np.abs(H[mask]))),'momentum_linf':float(np.max(np.abs(M[mask]))),'gauge_linf':float(np.max(np.abs(aux['Cup'][mask]))),
            'min_lapse':float(np.min(alpha[mask])),'min_spatial_metric_eigenvalue':float(np.min(np.linalg.eigvalsh(gam[mask])))}

def hermite(a,b,da,db,z,dt):return (2*z**3-3*z*z+1)*a+(-2*z**3+3*z*z)*b+dt*((z**3-2*z*z+z)*da+(z**3-z*z)*db)

def crossing_fraction(a,b,da,db,target,dt):
    z=np.clip((target-a)/(b-a),0,1)
    for _ in range(5):
        val=hermite(a,b,da,db,z,dt)-target
        der=(6*z*z-6*z)*a+(-6*z*z+6*z)*b+dt*((3*z*z-4*z+1)*da+(3*z*z-2*z)*db)
        z=np.clip(z-val/der,0,1)
    return z
