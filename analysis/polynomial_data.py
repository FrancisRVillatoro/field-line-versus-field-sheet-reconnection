#!/usr/bin/env python3
"""Gaussian perturbation coefficients in Appendix A; the symbol t denotes s.
These are exact polynomials, not a finite-amplitude numerical solution.
"""
import sympy as s
t,x,y=s.symbols('t x y',real=True); coords=[t,x,y]
rho=x*x+y*y; P=1-rho
L=s.Rational(16,9)-rho/4+rho**2/16-rho**3/36
f=x*y+t*P-s.Rational(2,3)*t**3
G=s.Matrix([[2*L+t*t*(-1+x*x+3*y*y-rho*rho)+s.Rational(2,3)*t**4*(1-2*y*y)-s.Rational(4,15)*t**6,
             -t*P*P+2*t*t*x*y-s.Rational(4,3)*t**3*rho+s.Rational(4,3)*t**4*x*y],
            [-t*P*P+2*t*t*x*y-s.Rational(4,3)*t**3*rho+s.Rational(4,3)*t**4*x*y,
             2*L+t*t*(-1+3*x*x+y*y-rho*rho)+s.Rational(2,3)*t**4*(1-2*x*x)-s.Rational(4,15)*t**6]])
w=(-2*L*P*t+t**3*(s.Rational(155,27)-s.Rational(13,3)*rho+s.Rational(19,6)*rho**2-s.Rational(26,27)*rho**3)
   +t**4*x*y*(1-s.Rational(4,3)*rho)
   +t**5*(-s.Rational(7,5)+s.Rational(58,15)*rho-s.Rational(34,15)*rho**2)
   -s.Rational(52,45)*t**6*x*y+t**7*(s.Rational(188,315)-s.Rational(304,315)*rho)-s.Rational(64,567)*t**9)
