Exact arithmetic and boundary checks
====================================

Run in this directory with standard Python (without -O):

    python verify_polynomials.py
    python verify_boundary_bounds.py
    python verify_twist_identities.py

The first script uses SymPy and imports polynomial_data.py. It writes
polynomial_checks.json. Its 24 exact polynomial residuals include the central
coefficient b_ini, reconstructed independently by repeated application of the
spatial Laplacian to the initial velocity -2LP. Three additional rational
root-isolation and sign checks certify the reported rounding of the metric
source fractions at material labels 0.04, 0.12 and 0.18.

The second script uses SymPy, NumPy and SciPy. It writes
boundary_bounds_checks.json with the 11 exact comparison identities of the
boundary lemma and integrates 256 trajectories at 101 output times.
The trajectories check the implementation. The uniform boundary bound follows
from the proof in the article.

The third script uses SymPy and its mpmath library. It concerns the twisting
Killing vector of Section 6 and writes twist_checks.json with 24 statements.
Fifteen are exact pointwise identities of the selected two-form, checked on
independent symbols for the fields and their first derivatives: the square of
the two-form, the relation of E.B to the gradients of the norm and of the twist
potential, the direction and in-plane components of the magnetic field, the
flux contraction and the twist three-form. One compares the four-dimensional
Ricci tensor with the reduction identities behind the reduced vacuum equations
at three random points of random polynomial fields, with 60-digit arithmetic.
Four concern the twisted Hamiltonian constraint, its radial form for the data
of Section 6, the twist potential of the central core and the first-order
flatness of that core. The last four check Proposition 11 in exact arithmetic:
the leading core twist and its initial data, the fourth-order central scalar
coefficient, the expansions of both central events through third order, and
the absence of a third-order term in the paired proper-time interval.

The archived reference outputs were regenerated with Python 3.12.14, SymPy
1.14.0, NumPy 2.3.5 and SciPy 1.17.0. Last-place differences in the numerical
trajectory diagnostics can occur across versions.
