"""Exact algebra for the regular-flow comparison; no evolution is rerun."""
from pathlib import Path
import json
import sympy as s

# Use normalized auxiliary covector p, |p|_E=1, p_s != -1.
p0,p1,p2=s.symbols('p0 p1 p2', real=True)
constraint=p1**2+p2**2-(1-p0**2)
w=s.Matrix([-p1,1-p1**2/(1+p0),-p1*p2/(1+p0)])
p=s.Matrix([p0,p1,p2])
def constrained_zero(expr):
    num=s.together(expr).as_numer_denom()[0]
    return s.rem(s.Poly(num,p2),s.Poly(constraint,p2)).as_expr()==0
checks={
 'ideal_contraction': constrained_zero((p.T*w)[0]),
 'nowhere_zero_auxiliary_norm': constrained_zero((w.T*w)[0]-1),
 'central_value': w.subs({p0:1,p1:0,p2:0})==s.Matrix([0,1,0]),
}
a,ap,eps,c,dz=s.symbols('a ap eps c dz',real=True)
flux=dz*(s.exp(a)-s.exp(eps*c))
checks['coordinate_length_flux_rate']=s.simplify(s.exp(a)*s.diff(flux,a)*ap/dz-s.exp(2*a)*ap)==0
checks['proper_length_line_density']=s.simplify(s.exp(2*a)*ap/s.exp(a)-s.exp(a)*ap)==0
# Implicit-sheet Hessian: phi_s sigma_ij + phi_ij = 0 at the critical point.
H11,H12,H22=s.symbols('H11 H12 H22')
H=s.Matrix([[H11,H12],[H12,H22]])
checks['sheet_hessian_determinant']=s.simplify((-H/ap).det()-H.det()/ap**2)==0
assert all(checks.values()),checks
result={'status':'PASS','checks':checks,'scope':'Exact algebra under the explicitly stated nonvanishing and unit-covector constraints; smooth development and boundary exchange are established by the manuscript proofs.'}
out=Path(__file__).resolve().parents[1]/'data/sheet_transport_checks.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
