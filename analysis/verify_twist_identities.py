"""Checks for the twisting Killing vector of Section 6; no evolution is rerun.

Metric g = exp(-2 phi) h + exp(2 phi) (dz + A)^2 with K = d/dz, phi = log|K|,
h a Lorentzian metric on the quotient and A a one-form on the quotient.
Selected leg e_K = K^flat/|K| = exp(phi) (dz + A), F_K = d e_K, Psi = exp(phi).
Twist potential omega: exp(4 phi) *_h dA = d omega.

Part 1 (exact, pointwise). The statements are algebraic in the one-jet of
(phi, A) at a point, so fields and first derivatives are independent symbols.
For the unit normal u of a slice s = const containing the Killing orbits:
  F^F = 2 exp(2 phi) dphi ^ dz ^ dA,   E.B = (F^F)/(2 vol_g),
  E.B = +-<dphi, domega>_h,   B^s = 0,   B(Psi) = 0,
  (B^x, B^y) sqrt(det gamma_hat) = -(Psi_y, -Psi_x), independent of A,
  F(v, d/dz) = v(Psi),   K^flat ^ dK^flat = |K|^2 e_K ^ F_K.
Part 2 (high precision at random points for random polynomial fields).
Four-dimensional Ricci components against the reduction identities, which give
the vacuum equations
  Ric[h]_ab = 2 phi_a phi_b + (1/2) exp(-4 phi) omega_a omega_b,
  Box_h phi + (1/2) exp(-4 phi) h^ab omega_a omega_b = 0,
  div_h (exp(-4 phi) grad_h omega) = 0.
Part 3 (exact). Hamiltonian and momentum constraint terms of the twist, and
first-order flatness of the uniform-twist core used in Section 6.
Part 4 (exact). Proposition 11: in the core the eps^4 coefficient of phi is
  f4 = -kappa^2 s^2/4, so a = eps q + eps^3 b - kappa^2 eps^4 s^2/4 + O(eps^5),
the central events at levels +-eps c are
  s_pm = +-(s0 + eps^2 d2) + eps^3 d3 + O(eps^4),  d3 = kappa^2 s0^2/(4 q'(s0)),
and the paired proper-time interval has no eps^3 term.
"""
import itertools
import json
import random
from pathlib import Path

import mpmath as mp
import sympy as sp

checks = {}


def record(name, ok):
    assert ok, name
    checks[name] = True


eps4 = sp.LeviCivita
perms4 = list(itertools.permutations(range(4)))

# ---------------------------------------------------------------- Part 1
P, N, q, g11, g12 = sp.symbols('P N q g11 g12', positive=True)   # P = exp(phi), q = sqrt(det gamma)
g22 = (q**2 + g12**2)/g11
p = list(sp.symbols('p0 p1 p2', real=True)) + [0]                # phi_a, phi_z = 0
a = list(sp.symbols('a0 a1 a2', real=True))                      # A_a
dA = [[sp.Symbol(f'd{b}{c}', real=True) for c in range(3)] + [0] for b in range(3)] + [[0]*4]
theta = a + [1]
h = sp.Matrix([[-N**2, 0, 0], [0, g11, g12], [0, g12, g22]])
hinv = h.inv()
G = sp.zeros(4, 4)
for i in range(3):
    for j in range(3):
        G[i, j] = h[i, j]/P**2 + P**2*a[i]*a[j]
    G[i, 3] = G[3, i] = P**2*a[i]
G[3, 3] = P**2
Ginv = sp.zeros(4, 4)
for i in range(3):
    for j in range(3):
        Ginv[i, j] = P**2*hinv[i, j]
    Ginv[i, 3] = Ginv[3, i] = -P**2*sum(hinv[i, k]*a[k] for k in range(3))
Ginv[3, 3] = 1/P**2 + P**2*sum(hinv[i, k]*a[i]*a[k] for i in range(3) for k in range(3))
record('kaluza_klein_inverse', (G*Ginv - sp.eye(4)).applyfunc(sp.cancel) == sp.zeros(4, 4))
record('volume_factor', sp.cancel(G.det() + (N*q)**2/P**4) == 0)
sqrtg = N*q/P**2
F = sp.zeros(4, 4)
for m in range(4):
    for n in range(4):
        F[m, n] = P*(p[m]*theta[n] - p[n]*theta[m]) + P*(dA[m][n] - dA[n][m])
FF = sp.expand(sp.Rational(1, 4)*sum(eps4(*w)*F[w[0], w[1]]*F[w[2], w[3]] for w in perms4))
dz = [0, 0, 0, 1]
W = sp.expand(sp.Rational(1, 2)*sum(eps4(*w)*p[w[0]]*dz[w[1]]*(dA[w[2]][w[3]] - dA[w[3]][w[2]])
                                   for w in perms4))
record('FF_equals_2exp2phi_dphi_dz_dA', sp.cancel(FF - 2*P**2*W) == 0)
alpha = N/P
record('lapse_of_invariant_slices', sp.cancel(Ginv[0, 0] + P**2/N**2) == 0)
ulow = [-alpha, 0, 0, 0]
uup = [sp.cancel(sum(Ginv[m, n]*ulow[n] for n in range(4))) for m in range(4)]
record('unit_normal', sp.cancel(sum(ulow[m]*uup[m] for m in range(4)) + 1) == 0)
Elow = [sp.expand(sum(F[m, n]*uup[n] for n in range(4))) for m in range(4)]
Bup = [sp.cancel(-sp.Rational(1, 2)*sum(eps4(*w)*ulow[w[1]]*F[w[2], w[3]] for w in perms4 if w[0] == m)/sqrtg)
       for m in range(4)]
EB = sp.cancel(sum(Elow[m]*Bup[m] for m in range(4)))
record('EB_equals_half_FF_over_vol', sp.cancel(EB - FF/(2*sqrtg)) == 0)
# twist potential gradient: omega_a = exp(4 phi) (*_h dA)_a, (*beta)_a = (1/2) beta^{bc} eps_{bca}
eps3 = sp.LeviCivita
dAh = sp.Matrix(3, 3, lambda b, c: dA[b][c] - dA[c][b])
dAup = hinv*dAh*hinv
sqrth = N*q
omega = [P**4*sp.Rational(1, 2)*sum(dAup[b, c]*eps3(b, c, k)*sqrth for b in range(3) for c in range(3))
         for k in range(3)]
inner = sp.cancel(sum(hinv[i, j]*p[i]*omega[j] for i in range(3) for j in range(3)))
ratio = sp.cancel(EB/inner)
record('EB_equals_pm_inner_dphi_domega', ratio in (1, -1))
record('B_tangent_to_slice', Bup[0] == 0)
Psi_d = [P*pm for pm in p]
record('B_annihilates_Psi', sp.cancel(sum(Bup[m]*Psi_d[m] for m in range(4))) == 0)
sqrt_gh = q/P
gh = sp.Matrix([[G[i, j] for j in (1, 2, 3)] for i in (1, 2, 3)])
record('induced_volume_independent_of_A', sp.cancel(gh.det() - sqrt_gh**2) == 0)
record('inplane_B_x', sp.cancel(Bup[1]*sqrt_gh + Psi_d[2]) == 0)
record('inplane_B_y', sp.cancel(Bup[2]*sqrt_gh - Psi_d[1]) == 0)
record('inplane_B_free_of_A', all(sp.diff(sp.cancel(Bup[k]*sqrt_gh), v) == 0
                                  for k in (1, 2) for v in a + [dA[b][c] for b in range(3) for c in range(3)]))
v = sp.symbols('v0 v1 v2', real=True)
record('flux_contraction', sp.cancel(sum(F[m, 3]*v[m] for m in range(3)) - sum(v[m]*Psi_d[m] for m in range(3))) == 0)
Kflat = [P**2*t for t in theta]
dK = sp.zeros(4, 4)
for m in range(4):
    for n in range(4):
        dK[m, n] = 2*P**2*(p[m]*theta[n] - p[n]*theta[m]) + P**2*(dA[m][n] - dA[n][m])
e = [P*t for t in theta]


def wedge13(one, two):
    """Components (lmn) of one ^ two for a one-form and a two-form."""
    return {c: sp.expand(one[c[0]]*two[c[1], c[2]] + one[c[1]]*two[c[2], c[0]] + one[c[2]]*two[c[0], c[1]])
            for c in itertools.combinations(range(4), 3)}


lhs = wedge13(Kflat, dK)
rhs = wedge13(e, F)
record('twist_three_form', all(sp.cancel(lhs[c] - P**2*rhs[c]) == 0 for c in lhs))

# ---------------------------------------------------------------- Part 2
mp.mp.dps = 60
s, x, y, z = sp.symbols('s x y z', real=True)
X = [s, x, y, z]
rng = random.Random(20261122)
mons = [s**i*x**j*y**k for i in range(4) for j in range(4) for k in range(4) if i + j + k <= 3]


def rpoly(scale):
    return sum(sp.Rational(rng.randint(-9, 9), 10)*scale*m for m in mons)


phi_f = rpoly(sp.Rational(1, 3))
A_f = [rpoly(sp.Rational(1, 3)) for _ in range(3)] + [0]
eta = sp.diag(-1, 1, 1)
h_f = sp.zeros(3, 3)
for i in range(3):
    for j in range(i, 3):
        h_f[i, j] = h_f[j, i] = eta[i, j] + rpoly(sp.Rational(1, 30))
g_f = sp.zeros(4, 4)
for i in range(3):
    for j in range(3):
        g_f[i, j] = sp.exp(-2*phi_f)*h_f[i, j] + sp.exp(2*phi_f)*A_f[i]*A_f[j]
    g_f[i, 3] = g_f[3, i] = sp.exp(2*phi_f)*A_f[i]
g_f[3, 3] = sp.exp(2*phi_f)


def jets(M, dim, point):
    """Values, first and second coordinate derivatives of a matrix field at a point."""
    n = M.shape[0]
    val = mp.matrix(n, n)
    d1 = [[[mp.mpf(0)]*n for _ in range(n)] for _ in range(dim)]
    d2 = [[[[mp.mpf(0)]*n for _ in range(n)] for _ in range(dim)] for _ in range(dim)]
    for i in range(n):
        for j in range(i, n):
            expr = M[i, j]
            val[i, j] = val[j, i] = mp.mpf(sp.N(expr.subs(point), 70))
            for k in range(min(dim, 3)):
                dk = sp.diff(expr, X[k])
                d1[k][i][j] = d1[k][j][i] = mp.mpf(sp.N(dk.subs(point), 70))
                for l in range(k, min(dim, 3)):
                    dkl = sp.diff(dk, X[l])
                    v2 = mp.mpf(sp.N(dkl.subs(point), 70))
                    d2[k][l][i][j] = d2[k][l][j][i] = d2[l][k][i][j] = d2[l][k][j][i] = v2
    return val, d1, d2


def christoffel_ricci(val, d1, d2, dim):
    inv = mp.inverse(val)
    dinv = [[[-sum(inv[m, a1]*d1[l][a1][b1]*inv[b1, k] for a1 in range(dim) for b1 in range(dim))
              for k in range(dim)] for m in range(dim)] for l in range(dim)]
    Gam = [[[sum(inv[m, k]*(d1[i][k][j] + d1[j][k][i] - d1[k][i][j]) for k in range(dim))/2
             for j in range(dim)] for i in range(dim)] for m in range(dim)]
    dGam = [[[[sum(dinv[l][m][k]*(d1[i][k][j] + d1[j][k][i] - d1[k][i][j]) for k in range(dim))/2
               + sum(inv[m, k]*(d2[l][i][k][j] + d2[l][j][k][i] - d2[l][k][i][j]) for k in range(dim))/2
               for j in range(dim)] for i in range(dim)] for m in range(dim)] for l in range(dim)]
    Ric = mp.matrix(dim, dim)
    for i in range(dim):
        for j in range(dim):
            Ric[i, j] = (sum(dGam[m][m][i][j] for m in range(dim)) - sum(dGam[j][m][i][m] for m in range(dim))
                         + sum(Gam[m][m][k]*Gam[k][i][j] for m in range(dim) for k in range(dim))
                         - sum(Gam[m][j][k]*Gam[k][i][m] for m in range(dim) for k in range(dim)))
    return inv, Gam, Ric


worst = mp.mpf(0)
for trial in range(3):
    point = {s: sp.Rational(rng.randint(-3, 3), 10), x: sp.Rational(rng.randint(-3, 3), 10),
             y: sp.Rational(rng.randint(-3, 3), 10)}
    gv, gd1, gd2 = jets(g_f, 4, point)
    _, _, R4 = christoffel_ricci(gv, gd1, gd2, 4)
    hv, hd1, hd2 = jets(h_f, 3, point)
    hi, Gh, Rh = christoffel_ricci(hv, hd1, hd2, 3)
    ph = mp.mpf(sp.N(phi_f.subs(point), 70))
    dph = [mp.mpf(sp.N(sp.diff(phi_f, X[k]).subs(point), 70)) for k in range(3)]
    ddph = [[mp.mpf(sp.N(sp.diff(phi_f, X[k], X[l]).subs(point), 70)) for l in range(3)] for k in range(3)]
    Av = [mp.mpf(sp.N(A_f[k].subs(point), 70)) for k in range(3)]
    Fexpr = [[sp.diff(A_f[j], X[i]) - sp.diff(A_f[i], X[j]) for j in range(3)] for i in range(3)]
    Fv = [[mp.mpf(sp.N(Fexpr[i][j].subs(point), 70)) for j in range(3)] for i in range(3)]
    dFv = [[[mp.mpf(sp.N(sp.diff(Fexpr[i][j], X[k]).subs(point), 70)) for j in range(3)] for i in range(3)]
           for k in range(3)]
    box = sum(hi[i, j]*(ddph[i][j] - sum(Gh[k][i][j]*dph[k] for k in range(3))) for i in range(3) for j in range(3))
    Fup = [[sum(hi[i, k]*hi[j, l]*Fv[k][l] for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]
    F2 = sum(Fv[i][j]*Fup[i][j] for i in range(3) for j in range(3))
    FFmix = [[sum(Fv[i][k]*Fv[j][l]*hi[k, l] for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]
    e4 = mp.e**(4*ph)
    hor = [[1 if m == i else 0 for m in range(3)] + [-Av[i]] for i in range(3)]
    Kz = [0, 0, 0, 1]

    def R4c(U, V):
        return sum(U[m]*V[n]*R4[m, n] for m in range(4) for n in range(4))

    for i in range(3):
        for j in range(3):
            pred = Rh[i, j] - 2*dph[i]*dph[j] + box*hv[i, j] - e4*FFmix[i][j]/2
            worst = max(worst, abs(R4c(hor[i], hor[j]) - pred))
    worst = max(worst, abs(R4c(Kz, Kz) - e4*(-box + e4*F2/4)))
    # Maxwell-type component: (1/2) div_h (exp(4 phi) F)_a
    for i in range(3):
        div = mp.mpf(0)
        for b in range(3):
            for c in range(3):
                cov = (4*dph[c]*e4*Fv[i][b] + e4*dFv[c][i][b]
                       - sum(Gh[d][c][i]*e4*Fv[d][b] + Gh[d][c][b]*e4*Fv[i][d] for d in range(3)))
                div += hi[b, c]*cov
        worst = max(worst, abs(R4c(hor[i], Kz) - div/2))
    # duality with omega_a = exp(4 phi) (*_h F)_a
    sq = mp.sqrt(-mp.det(hv))
    om = [e4*sum(Fup[b][c]*int(sp.LeviCivita(b, c, k)) for b in range(3) for c in range(3))*sq/2 for k in range(3)]
    om2 = sum(hi[i, j]*om[i]*om[j] for i in range(3) for j in range(3))
    worst = max(worst, abs(F2 + 2*om2/e4**2))
    for i in range(3):
        for j in range(3):
            worst = max(worst, abs(FFmix[i][j] - hv[i, j]*F2/2 - om[i]*om[j]/e4**2))
record('reduction_identities_random_points', worst < mp.mpf(10)**(-45))

# ---------------------------------------------------------------- Part 3
Pw, Dw2, Pp, Dp2, ph3 = sp.symbols('Pi_omega Domega2 Pi Dphi2 phi', real=True)
# S_ab = 2 phi_a phi_b + (1/2) e^{-4phi} omega_a omega_b in Gaussian form, h_nn = -1:
Snn = 2*Pp**2 + sp.exp(-4*ph3)*Pw**2/2
trS = 2*(-Pp**2 + Dp2) + sp.exp(-4*ph3)*(-Pw**2 + Dw2)/2
record('hamiltonian_source', sp.simplify(2*Snn + trS - 2*(Pp**2 + Dp2) - sp.exp(-4*ph3)*(Pw**2 + Dw2)/2) == 0)
lam, eps_, kap, mu, r = sp.symbols('lambda epsilon kappa mu r', real=True)
Pr = 1 - r**2
# Radial reduction with Pi_omega = kappa eps^2 e^{2phi} e^{-2lambda} mu and omega = 0 initially
Hres = (-2*sp.exp(-2*lam)*sp.Symbol('Lap') - sp.exp(-4*lam)*eps_**4*Pr**4/2 - 2*eps_**2*sp.exp(-4*lam)*Pr**2
        - 2*eps_**2*sp.exp(-2*lam)*r**2 - sp.exp(-4*ph3)*(kap*eps_**2*sp.exp(2*ph3)*sp.exp(-2*lam)*mu)**2/2)
Lap_sol = sp.solve(sp.Eq(Hres, 0), sp.Symbol('Lap'))[0]
target = -eps_**2*r**2 - sp.exp(-2*lam)*(eps_**2*Pr**2 + eps_**4*Pr**4/4 + kap**2*eps_**4*mu**2/4)
record('radial_constraint_with_twist', sp.simplify(Lap_sol - target) == 0)
# Uniform-twist core: omega = delta s, phi = 0, h = eta gives A = (delta/2)(x dy - y dx) up to sign,
# and g = eta + (dz + A)^2 has vanishing Riemann tensor at first order in delta.
dl = sp.symbols('delta', real=True)
Ac = [0, -dl*y/2, dl*x/2, 0]
gc = sp.zeros(4, 4)
for i in range(3):
    for j in range(3):
        gc[i, j] = eta[i, j] + Ac[i]*Ac[j]
    gc[i, 3] = gc[3, i] = Ac[i]
gc[3, 3] = 1
dAc = sp.Matrix(3, 3, lambda b, c: sp.diff(Ac[c], X[b]) - sp.diff(Ac[b], X[c]))
omc = [sp.Rational(1, 2)*sum((eta*dAc*eta)[b, c]*eps3(b, c, k) for b in range(3) for c in range(3)) for k in range(3)]
record('core_twist_potential', omc in ([dl, 0, 0], [-dl, 0, 0]))
gci = gc.inv()
Gc = [[[sp.simplify(sum(gci[m, k]*(sp.diff(gc[k, j], X[i]) + sp.diff(gc[k, i], X[j]) - sp.diff(gc[i, j], X[k]))
                        for k in range(4))/2) for j in range(4)] for i in range(4)] for m in range(4)]
first_order_riemann = []
for rho, sig, mm, nn in itertools.product(range(4), repeat=4):
    Rc = (sp.diff(Gc[rho][nn][sig], X[mm]) - sp.diff(Gc[rho][mm][sig], X[nn])
          + sum(Gc[rho][mm][l]*Gc[l][nn][sig] - Gc[rho][nn][l]*Gc[l][mm][sig] for l in range(4)))
    first_order_riemann.append(sp.diff(sp.simplify(Rc), dl).subs(dl, 0))
record('core_flat_to_first_order', all(sp.simplify(t) == 0 for t in first_order_riemann))

# ---------------------------------------------------------------- Part 4
# Proposition 11. In the domain of dependence r + |s| <= 3/8 of the disk where mu = 1 the leading
# twist is omega = kappa eps^2 s. The eps^4 coefficient f4 of phi solves
# Box_eta f4 = -(1/2) eta^ab omega2_a omega2_b with zero data. q and b are the twist-free
# central coefficients of Appendix A (the symbol t denotes s).
from polynomial_data import t as tq, x as xq, y as yq, f as fq, w as wq
kq, eq, s0 = sp.symbols('kappa epsilon s0', positive=True)
etaq = [-1, 1, 1]
Xq = [tq, xq, yq]


def boxq(F):
    return sum(etaq[i]*sp.diff(F, Xq[i], 2) for i in range(3))


om2 = kq*tq
record('core_twist_wave_and_data',
       boxq(om2) == 0 and om2.subs(tq, 0) == 0 and sp.diff(om2, tq).subs(tq, 0) == kq)
f4 = -kq**2*tq**2/4
src4 = -sp.Rational(1, 2)*sum(etaq[i]*sp.diff(om2, Xq[i])**2 for i in range(3))
record('fourth_order_central_scalar',
       sp.expand(boxq(f4) - src4) == 0 and f4.subs(tq, 0) == 0 and sp.diff(f4, tq).subs(tq, 0) == 0)
qq = fq.subs({xq: 0, yq: 0})
bq = wq.subs({xq: 0, yq: 0})
aq = eq*qq + eq**3*bq + eq**4*f4          # central value through order eps^4
qp0 = sp.diff(qq, tq).subs(tq, s0)
d2 = -bq.subs(tq, s0)/qp0
d3 = kq**2*s0**2/(4*qp0)
splus = s0 + eq**2*d2 + eq**3*d3
sminus = -(s0 + eq**2*d2) + eq**3*d3
c0 = qq.subs(tq, s0)


def taylor_coefficients(expr, order):
    return [sp.simplify(sp.diff(expr, eq, k).subs(eq, 0)/sp.factorial(k)) for k in range(order)]


# a(s_pm)/eps -+ c vanishes through eps^3 (division by eps is exact for the polynomial aq)
root_plus = sp.expand(aq/eq).subs(tq, splus) - c0
root_minus = sp.expand(aq/eq).subs(tq, sminus) + c0
record('crossing_times_with_twist',
       all(cf == 0 for cf in taylor_coefficients(root_plus, 4) + taylor_coefficients(root_minus, 4)))
# e^{-a} through eps^3, integrated between the shifted events, against 2 s0 + eps^2 [2 d2 + Q2(s0)]
expa = sp.series(sp.exp(-aq), eq, 0, 4).removeO()
prim = sp.integrate(sp.expand(expa), tq)
dtau = prim.subs(tq, splus) - prim.subs(tq, sminus)
Q2 = sp.integrate(qq**2, (tq, 0, s0))
target = 2*s0 + eq**2*(2*d2 + Q2)
record('paired_interval_without_cubic_term', all(cf == 0 for cf in taylor_coefficients(dtau - target, 4)))

result = {'status': 'PASS', 'count': len(checks), 'checks': checks,
          'scope': 'Exact pointwise algebra (part 1), reduction identities at random points with random '
                   'polynomial fields to 60 digits (part 2), exact constraint and core algebra (part 3) and '
                   'exact perturbative coefficients of Proposition 11 (part 4); existence and the boundary '
                   'exchange are established by the manuscript proofs.'}
Path(__file__).with_name('twist_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
