#!/usr/bin/env python3
"""Independent NumPy pointwise reference for zero-source generalized harmonic GR.

This module is not a grid solver. Arrays may have arbitrary common leading
batch axes. Last axes are g[a,b], dg[d,a,b], ddg[d,e,a,b]. All metric and
derivative-pair symmetries are expected in the inputs. Indices run 0,1,2.
The scalar uses dphi[a], ddphi[a,b]. Signature is (-,+,+).
"""
import numpy as np

def geometry(g, dg, ddg):
    """Return Ricci, reduced Ricci, contracted Gamma, inverse metric.

    Reduced Ricci is Ric_ab - nabla_(a C_b), C^a=g^bc Gamma^a_bc.
    C is used in the chosen coordinate chart; it is not a tensor under
    arbitrary coordinate transformations unless a background is supplied.
    """
    g=np.asarray(g,dtype=float); dg=np.asarray(dg,dtype=float); ddg=np.asarray(ddg,dtype=float)
    gi=np.linalg.inv(g)
    dgi=-np.einsum('...ap,...epq,...qb->...eab',gi,dg,gi,optimize=True)
    # S[d,b,c] = g_dc,b + g_db,c - g_bc,d.
    S=np.swapaxes(dg,-3,-2)+np.moveaxis(dg,-3,-1)-dg
    dS=np.swapaxes(ddg,-3,-2)+np.moveaxis(ddg,-3,-1)-ddg
    Gamma=.5*np.einsum('...ad,...dbc->...abc',gi,S,optimize=True)
    dGamma=.5*(np.einsum('...ead,...dbc->...eabc',dgi,S,optimize=True)
                  +np.einsum('...ad,...edbc->...eabc',gi,dS,optimize=True))
    Ric=(np.einsum('...ccab->...ab',dGamma)-np.einsum('...bcac->...ab',dGamma)
         +np.einsum('...ccd,...dab->...ab',Gamma,Gamma,optimize=True)
         -np.einsum('...cbd,...dac->...ab',Gamma,Gamma,optimize=True))
    C=np.einsum('...bc,...abc->...a',gi,Gamma,optimize=True)
    dC=(np.einsum('...ebc,...abc->...ea',dgi,Gamma,optimize=True)
         +np.einsum('...bc,...eabc->...ea',gi,dGamma,optimize=True))
    Clower=np.einsum('...ba,...a->...b',g,C,optimize=True)
    dClower=(np.einsum('...eba,...a->...eb',dg,C,optimize=True)
              +np.einsum('...ba,...ea->...eb',g,dC,optimize=True))
    nablaC=dClower-np.einsum('...cab,...c->...ab',Gamma,Clower,optimize=True)
    reduced=Ric-.5*(nablaC+np.swapaxes(nablaC,-1,-2))
    return {'ricci':Ric,'reduced_ricci':reduced,'contracted_gamma':C,
            'inverse_metric':gi,'christoffel':Gamma}

def metric_tt(g, dg, ddg_without_tt, dphi):
    """Solve reduced Ricci=2*dphi*dphi for all metric tt derivatives.

    ddg_without_tt includes spatial and mixed second derivatives. Its tt
    block is ignored and replaced with zero. No damping is included.
    For a method of lines, mixed terms come from spatial derivatives of
    the evolved first-time-derivative variable, not from an extra solve.
    """
    d2=np.array(ddg_without_tt,dtype=float,copy=True)
    d2[...,0,0,:,:]=0.
    result=geometry(g,dg,d2)
    gi=result['inverse_metric']
    if np.any(gi[...,0,0]>=0):
        raise ValueError('The chosen time coordinate must have g^{00}<0.')
    source=2*np.einsum('...a,...b->...ab',dphi,dphi)
    return 2*(result['reduced_ricci']-source)/gi[...,0,0,None,None]

def scalar_tt(g,dg,ddphi_without_tt,dphi,use_full_covariant_wave=True):
    """Solve Box_g phi=0 (default) or its C=0 reduced variant for phi_tt.

    The full covariant wave retains the numerically measured contracted
    Gamma. On the harmonic constraint surface both choices coincide.
    Keeping the full wave makes the scalar stress conservation identity
    hold directly, useful for the gauge-constraint propagation argument.
    """
    gi=np.linalg.inv(g)
    if np.any(gi[...,0,0]>=0):
        raise ValueError('The chosen time coordinate must have g^{00}<0.')
    d2=np.array(ddphi_without_tt,dtype=float,copy=True)
    d2[...,0,0]=0.
    contraction=np.einsum('...ab,...ab->...',gi,d2)
    if use_full_covariant_wave:
        S=np.swapaxes(dg,-3,-2)+np.moveaxis(dg,-3,-1)-dg
        Gamma=.5*np.einsum('...ad,...dbc->...abc',gi,S,optimize=True)
        C=np.einsum('...bc,...abc->...a',gi,Gamma,optimize=True)
        contraction-=np.einsum('...a,...a->...',C,dphi)
    return -contraction/gi[...,0,0]
