"""Exact independent check of the Gaussian perturbation hierarchy (SymPy)."""
from polynomial_data import t,x,y,rho,P,L,f,G,w,s
from pathlib import Path
import json
coords=(t,x,y); eta=s.diag(-1,1,1); H=s.zeros(3); H[1:3,1:3]=G
box=lambda v:-s.diff(v,t,2)+s.diff(v,x,2)+s.diff(v,y,2)
tr=s.trace(eta*H)
checks={}
def check(name,v):
    r=s.expand(v); checks[name]=r==0
    if r != 0:
        raise AssertionError(f"{name}: nonzero residual {r}")
check('leading_scalar_wave',box(f))
check('radial_initial_equation',s.diff(L,x,2)+s.diff(L,y,2)+rho+P**2)
check('radial_normalization',L.subs({x:2,y:0}))
for a in range(3):
    for b in range(a,3):
        Ric=(sum(eta[c,c]*(s.diff(H[b,c],coords[c],coords[a])+s.diff(H[a,c],coords[c],coords[b])) for c in range(3))-box(H[a,b])-s.diff(tr,coords[a],coords[b]))/2
        check('Ricci_'+str(a)+str(b),Ric-2*s.diff(f,coords[a])*s.diff(f,coords[b]))
source=-s.diff(s.trace(G),t)*s.diff(f,t)/2
source-=sum(G[i,j]*s.diff(f,coords[i+1],coords[j+1]) for i in range(2) for j in range(2))
source+=sum((s.diff(s.trace(G),coords[j+1])/2-sum(s.diff(G[i,j],coords[i+1]) for i in range(2)))*s.diff(f,coords[j+1]) for j in range(2))
check('cubic_scalar_wave',s.diff(w,t,2)-s.diff(w,x,2)-s.diff(w,y,2)-source)
check('cubic_initial_value',w.subs(t,0))
check('cubic_initial_velocity',s.diff(w,t).subs(t,0)+2*L*P)
for i in range(2):
    for j in range(i,2):
        check('metric_initial_'+str(i)+str(j),G[i,j].subs(t,0)-(2*L if i==j else 0))
        check('metric_velocity_'+str(i)+str(j),s.diff(G[i,j],t).subs(t,0)-(-P**2 if i!=j else 0))
q=t-s.Rational(2,3)*t**3
b=-s.Rational(32,9)*t+s.Rational(155,27)*t**3-s.Rational(7,5)*t**5+s.Rational(188,315)*t**7-s.Rational(64,567)*t**9
check('central_b',w.subs({x:0,y:0})-b)
Q=t**3/3-s.Rational(4,15)*t**5+s.Rational(4,63)*t**7
check('paired_proper_clock',s.diff(Q,t)-q*q)
check('time_reflection_scalar',w.subs({t:-t,y:-y},simultaneous=True)+w)
check('center_gradient_x',s.diff(w,x).subs({x:0,y:0}))
check('center_gradient_y',s.diff(w,y).subs({x:0,y:0}))

def homogeneous_central_value(initial_velocity):
    """Return the axial flat-wave solution for zero displacement and polynomial velocity."""
    coefficient = s.expand(initial_velocity)
    central_value = 0
    order = 1
    while coefficient != 0:
        central_value += coefficient.subs({x: 0, y: 0}) * t**order / s.factorial(order)
        coefficient = s.expand(s.diff(coefficient, x, 2) + s.diff(coefficient, y, 2))
        order += 2
    return central_value


b_ini = homogeneous_central_value(-2 * L * P)
published_b_ini = (-s.Rational(32, 9)*t + s.Rational(73, 27)*t**3 - t**5/3
                   + s.Rational(26, 315)*t**7 - s.Rational(64, 2835)*t**9)
check('central_b_ini_from_initial_velocity', b_ini - published_b_ini)
b_src = s.expand(b - b_ini)


def sign_on_interval(expression, lo, hi):
    """Certify a constant nonzero sign by exact polynomial root counting."""
    poly = s.Poly(expression, t, domain=s.QQ)
    if poly.count_roots(lo, hi) != 0:
        raise AssertionError(f'Unresolved sign on [{lo}, {hi}] for {poly}')
    return s.sign(poly.eval((lo + hi) / 2))


def certify_percentage(label_text, printed_text):
    """Verify rounding of 100*d2_src/d2 on the increasing branch of q."""
    label, printed = map(s.Rational, (label_text, printed_text))
    equation = s.Poly(q - label, t, domain=s.QQ)
    # On [0, 1/4], q' >= 7/8. All three material labels cross in this interval.
    roots = equation.intervals(inf=0, sup=s.Rational(1, 4), eps=s.Rational(1, 10**20))
    if len(roots) != 1 or roots[0][1] != 1:
        raise AssertionError(f'Expected one simple crossing for c={label}')
    (lo, hi), _ = roots[0]
    # The common factor -1/q'(s0) in both timing shifts cancels in the ratio.
    # A 0.01 percentage-point bin has half-width 1/200 in percent units.
    lower, upper = printed - s.Rational(1, 200), printed + s.Rational(1, 200)
    denominator_sign = sign_on_interval(b, lo, hi)
    for limit, expected_sign in [(lower, 1), (upper, -1)]:
        difference_sign = sign_on_interval(100*b_src - limit*b, lo, hi) * denominator_sign
        if difference_sign != expected_sign:
            raise AssertionError(f'Source share does not round to {printed_text}% for c={label_text}')
    midpoint = (lo + hi) / 2
    share = (100*b_src/b).subs(t, midpoint)
    return {'c': label_text, 'percent_approx': str(share.evalf(15)),
            'rounded_percent': printed_text, 'rounding_certified': True,
            's0_rational_bracket': [str(lo), str(hi)],
            'open_percent_bin': [str(lower), str(upper)]}


percentages = [certify_percentage(label, target) for label, target in
               [('0.04', '-0.14'), ('0.12', '-1.28'), ('0.18', '-3.03')]]
result = {'status': 'PASS', 'count': len(checks), 'checks': checks,
          'b_ini': str(b_ini), 'b_src': str(b_src),
          'source_percentages': percentages,
          'scope': 'The count includes only exact polynomial identities. Three separate checks certify the printed percentages by rational root isolation and exact sign tests.'}
output = json.dumps(result, indent=2) + '\n'
Path(__file__).with_name('polynomial_checks.json').write_text(output, encoding='utf-8')
print(output, end='')
