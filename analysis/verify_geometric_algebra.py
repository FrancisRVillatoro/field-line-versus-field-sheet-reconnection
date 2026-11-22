"""Exact checks of selected-form normalization and initial constraints."""
import sympy as s
import json
from pathlib import Path
q,px,py,phi,A,B=s.symbols('q px py phi A B', real=True, nonzero=True)
hinv=s.diag(-1,1/A,1/B)
g=s.diag(-s.exp(-2*phi),A*s.exp(-2*phi),B*s.exp(-2*phi),s.exp(2*phi))
F=s.zeros(4)
for i,p in enumerate([q,px,py]):F[i,3]=s.exp(phi)*p;F[3,i]=-s.exp(phi)*p
checks={}
def check(name,r):
    if isinstance(r,s.MatrixBase):ok=all(s.simplify(v)==0 for v in r)
    else:ok=s.simplify(r)==0
    assert ok,name
    checks[name]=True
Fup=g.inv()*F*g.inv()
invariant=sum(F[i,j]*Fup[i,j] for i in range(4) for j in range(4))/2
Dp2=px*px/A+py*py/B
check('two_form_invariant_conformal_factor',invariant-s.exp(2*phi)*(Dp2-q*q))
V=s.Matrix([1,-q*px/(A*Dp2),-q*py/(B*Dp2),0])
check('boundary_ideal_contraction',(V.T*F))
check('boundary_norm',(V.T*g*V)[0]-s.exp(-2*phi)*(-1+q*q/Dp2))
check('critical_pullback',F.subs({px:0,py:0})[1:4,1:4])
x,y,eps=s.symbols('x y eps',real=True); lam=s.Function('lam')(x,y);P=1-x*x-y*y
gamma=s.exp(2*lam)*s.eye(2);gi=gamma.inv();coords=[x,y];K=s.Matrix([[0,eps**2*P**2/2],[eps**2*P**2/2,0]])
Gamma=[[[s.simplify(sum(gi[k,l]*(s.diff(gamma[l,j],coords[i])+s.diff(gamma[l,i],coords[j])-s.diff(gamma[i,j],coords[l])) for l in range(2))/2) for j in range(2)] for i in range(2)] for k in range(2)]
mix=gi*K;Pi=eps*s.exp(-2*lam)*P;grad=[eps*y,eps*x]
for i in range(2):
    div=sum(s.diff(mix[j,i],coords[j])+sum(Gamma[j][j][k]*mix[k,i]-Gamma[k][j][i]*mix[j,k] for k in range(2)) for j in range(2))
    check(f'exact_momentum_{i}',div+2*Pi*grad[i])
Kn=s.trace(gi*K*gi*K);grad2=sum(gi[i,j]*grad[i]*grad[j] for i in range(2) for j in range(2))
lap=-eps**2*(x*x+y*y)-s.exp(-2*lam)*(eps**2*P**2+eps**4*P**4/4)
check('exact_Hamiltonian_reduction',-2*s.exp(-2*lam)*lap-Kn-2*(Pi*Pi+grad2))
result={'status':'PASS','count':len(checks),'checks':checks,'scope':'Exact local algebra; no numerical or global existence assertion.'}
Path(__file__).with_name('geometric_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
