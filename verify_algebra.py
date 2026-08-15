"""
verify_algebra.py

Symbolically verifies the algebraic elimination steps used in Section 4
("Stationarity conditions") of:

    Shoichiro Kamaguchi, "Explicit Stationarity Regions for AR(4), AR(5),
    and AR(6) via Unit-Circle Analysis."

The paper repeatedly states that a given elimination or substitution step
"was verified symbolically" or "reproduces exactly" a target equation.
This script performs those verifications explicitly and prints PASS/FAIL
for each one, so that every such claim in the paper can be checked
independently of the manuscript's prose derivation.

Two kinds of steps are checked:

  1. Degree-reduction steps that use one Chebyshev relation (e.g.
     T3(x) = x*U2(x) - U1(x)) together with the quadratic equation (I) or
     (imag) to eliminate the U2(x) term from a cubic, leaving a quadratic.
     These are verified by performing the substitution symbolically in the
     T_n/U_n (Chebyshev) basis and checking the result is *exactly* equal
     to the paper's stated result (not just proportional).

  2. Substitution steps of the form "solve AX=B for X and substitute into
     another quadratic, clearing denominators" -- verified by direct
     substitution and simplification with sympy.

Requirements: sympy only. Run with:

    python3 verify_algebra.py

All checks should print "PASS". Runtime is a few seconds.
"""

import sympy as sp

x, X = sp.symbols('x X')
phi1, phi2, phi3, phi4, phi5, phi6 = sp.symbols('phi1 phi2 phi3 phi4 phi5 phi6')

# Chebyshev polynomials of the first and second kind (standard low-order forms)
T1_, T2_, T3_ = x, 2*x**2 - 1, 4*x**3 - 3*x
U0_, U1_, U2_ = 1, 2*x, 4*x**2 - 1

results = []

def check(label, expr_should_be_zero):
    val = sp.expand(expr_should_be_zero)
    ok = (val == 0)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        print("        residual:", val)
    results.append(ok)
    return ok

print("="*70)
print("AR(4): final equivalence-chain identity (Section 4.4)")
print("="*70)
# Paper's chain (with X = 2x):
#   X^2 + (-phi1)*X + (-1-phi2+phi4) = 0                        ... (i)
#   (1+phi4)*X = phi1 - phi3                                    ... (ii)
# Eliminating X between (i) and (ii) gives:
#   (phi1-phi3)^2 + (1+phi4)*(phi1-phi3)*(-phi1) + (1+phi4)^2*(-1-phi2+phi4) = 0   ... (iii)
# which the paper rewrites as:
#   (phi3+phi1*phi4)*(phi1-phi3) + (1+phi4)^2*(1+phi2-phi4) = 0                    ... (iv)
lhs_iii = (phi1-phi3)**2 + (1+phi4)*(phi1-phi3)*(-phi1) + (1+phi4)**2*(-1-phi2+phi4)
lhs_iv  = (phi3+phi1*phi4)*(phi1-phi3) + (1+phi4)**2*(1+phi2-phi4)
check("(iii), the direct elimination result, equals exactly -(iv) as stated in the paper", lhs_iii + lhs_iv)

i_expr = X**2 + (-phi1)*X + (-1-phi2+phi4)
X_sub = (phi1-phi3)/(1+phi4)
cleared = sp.expand(sp.together(i_expr.subs(X, X_sub)) * (1+phi4)**2)
check("(i) with X=(phi1-phi3)/(1+phi4) substituted in, cleared of denominators, equals (iii)", cleared - lhs_iii)

print()
print("="*70)
print("AR(5): (R),(I) -> (R') -> (L) -> (E)  (Section 4.5)")
print("="*70)

# (R) = T3 - (phi1+phi5)*T2 - (phi2+phi4)*T1 - phi3, as stated explicitly in the paper
R = T3_ - (phi1+phi5)*T2_ - (phi2+phi4)*T1_ - phi3
check("(R) as defined via Chebyshev T_n matches the paper's explicit cubic 4x^3-3x-(phi1+phi5)(2x^2-1)-(phi2+phi4)x-phi3",
      R - (4*x**3 - 3*x - (phi1+phi5)*(2*x**2-1) - (phi2+phi4)*x - phi3))

# (I), rewritten in the U-basis, gives U2 = (phi1-phi5)*U1 + (phi2-phi4); this is exactly
# what the paper says it substitutes ("substituting U2(x)=(phi1-phi5)U1(x)+(phi2-phi4) from (I)")
I_x = 4*x**2 + 2*(phi5-phi1)*x + (-1-phi2+phi4)   # (I) with X=2x expanded in x
U2_from_I = (phi1-phi5)*U1_ + (phi2-phi4)
check("(I) rewritten in the U-basis (U2=4x^2-1, U1=2x) is equivalent to U2 = (phi1-phi5)U1+(phi2-phi4)",
      I_x - (U2_ - U2_from_I))

# Eliminate the cubic term of (R) via T3 = x*U2 - U1, then substitute the (I)-derived U2:
R_after_elim = sp.expand(x*U2_from_I - U1_ - (phi1+phi5)*T2_ - (phi2+phi4)*T1_ - phi3)
Rp_x = -phi5*(4*x**2) - (1+phi4)*(2*x) + (phi1-phi3+phi5)   # (R') with X=2x expanded in x
check("(R), with T3=x*U2-U1 and U2 eliminated via (I), equals exactly (R')", R_after_elim - Rp_x)

# (R') paired with (I): use (I) to substitute X^2=(phi1-phi5)X+(1+phi2-phi4) into (R'), giving (L)
Rp_X = -phi5*X**2 - (1+phi4)*X + (phi1-phi3+phi5)
X2_from_I = (phi1-phi5)*X + (1+phi2-phi4)
Rp_after_elim = sp.expand(Rp_X.subs(X**2, X2_from_I))
L_claimed = (1+phi4+phi1*phi5-phi5**2)*X - (phi1-phi3-phi2*phi5+phi4*phi5)
# Rp' = 0 combined with the X^2-substitution yields -L = 0, the same equation as L = 0
# (both sides of an "= 0" equation may be freely rescaled by a nonzero constant), so the
# correct check is that the two expressions are negatives of one another.
check("(R') with X^2 eliminated via (I) gives the same equation as (L) (equal up to an overall sign, since both sides are set to 0)",
      Rp_after_elim + L_claimed)

# (L): AX=B => X=B/A; substitute into (I), clear denominators, compare with (E)
A5 = 1 + phi4 + phi1*phi5 - phi5**2
B5 = phi1 - phi3 - phi2*phi5 + phi4*phi5
check("(L) matches A*X=B with the paper's stated A, B", L_claimed - (A5*X - B5))

E_claimed = B5**2 + A5*B5*(phi5-phi1) + A5**2*(-1-phi2+phi4)
I_X = 4*X**2/4 + 2*(phi5-phi1)*X/2 + (-1-phi2+phi4)  # (I) directly in X=2x: X^2+(phi5-phi1)X+(-1-phi2+phi4)
I_at_X = I_X.subs(X, B5/A5)
cleared_E = sp.expand(sp.together(I_at_X) * A5**2)
check("(I) with X=B/A substituted in, cleared of denominators, equals (E)", cleared_E - E_claimed)

print()
print("="*70)
print("AR(6): (real),(imag) -> (R6') -> A6*X=B6 -> raw quartic -> BI6 identity  (Section 4.6)")
print("="*70)

check("Chebyshev identity T3(x) = x*U2(x) - U1(x)", T3_ - (x*U2_ - U1_))

real_eq = (1-phi6)*T3_ - (phi1+phi5)*T2_ - (phi2+phi4)*T1_ - phi3     # (real)
imag_eq_U = (1+phi6)*U2_ - (phi1-phi5)*U1_ - (phi2-phi4)*U0_          # (imag), U-basis
I6_claimed_x = 4*(1+phi6)*x**2 + 2*(phi5-phi1)*x + (-1-phi2+phi4-phi6)  # (I6) with X=2x expanded in x
check("(imag) rewritten in x (via U2=4x^2-1 etc.) equals (I6) as stated in the paper",
      imag_eq_U - I6_claimed_x)

U2_from_imag = ((phi1-phi5)*U1_ + (phi2-phi4)*U0_) / (1+phi6)
real_after_elim = sp.expand(sp.together((1-phi6)*(x*U2_from_imag - U1_) - (phi1+phi5)*T2_ - (phi2+phi4)*T1_ - phi3) * (1+phi6))
Rp6_x = -(phi5+phi1*phi6)*(4*x**2) + (phi6**2-phi2*phi6-phi4-1)*(2*x) + (1+phi6)*(phi1-phi3+phi5)
check("(real), with T3=x*U2-U1 and U2 eliminated via (imag), *(1+phi6), equals exactly (R6')",
      real_after_elim - Rp6_x)

# (R6') paired with (I6): eliminate X^2 using I6 (X^2 = [(phi1-phi5)X+(1+phi2-phi4+phi6)]/(1+phi6))
Rp6_X = -(phi5+phi1*phi6)*X**2 + (phi6**2-phi2*phi6-phi4-1)*X + (1+phi6)*(phi1-phi3+phi5)
I6_X = (1+phi6)*X**2 + (phi5-phi1)*X + (-1-phi2+phi4-phi6)
X2_from_I6 = sp.solve(sp.Eq(I6_X, 0), X**2)[0]
Rp6_after_elim = sp.expand(sp.together(Rp6_X.subs(X**2, 0) + (-(phi5+phi1*phi6))*X2_from_I6) * (1+phi6))
A6 = (1+phi6)*(phi6**2-phi2*phi6-phi4-1) + (phi5-phi1)*(phi5+phi1*phi6)
B6 = (phi5+phi1*phi6)*(phi6-phi4+phi2+1) - (1+phi6)**2*(phi1-phi3+phi5)
AXmB_claimed = A6*X - B6
check("(R6') with X^2 eliminated via (I6), cleared of the (1+phi6) denominator, equals exactly A6*X - B6",
      Rp6_after_elim - AXmB_claimed)

raw_claimed = (1+phi6)*B6**2 + A6*B6*(phi5-phi1) + A6**2*(-1-phi2+phi4-phi6)
I6_at_X = I6_X.subs(X, B6/A6)
cleared_raw = sp.expand(sp.together(I6_at_X) * A6**2)
check("(I6) with X=B6/A6 substituted in, cleared of denominators, equals the raw (pre-BI6) quartic-degree-7 equation",
      cleared_raw - raw_claimed)

T_ = -1-phi2+phi4-phi6
U_ = phi6**2-phi6*phi2-phi4-1
V_ = phi5+phi1*phi6
W_ = phi1-phi3+phi5
BI6 = (-phi1*V_*T_**2
       - T_*(phi6-phi2-2)*(phi5-phi1)*V_
       - U_**2*T_
       + (phi5-phi1)**2*V_*W_
       + (1+phi6)*( 2*V_*(phi6-phi4+phi2+1)*W_ + U_*W_*(phi5-phi1) )
       - (1+phi6)**3*W_**2)
check("raw (pre-BI6) quartic-degree-7 equation equals exactly -(1+phi6)^2 * BI6, as claimed in Section 4.6",
      raw_claimed + (1+phi6)**2*BI6)

# Sanity check: BI6 (degree 5) is genuinely lower total degree than the raw equation (degree 7)
deg_BI6 = sp.total_degree(sp.Poly(sp.expand(BI6), phi1, phi2, phi3, phi4, phi5, phi6))
deg_raw = sp.total_degree(sp.Poly(sp.expand(raw_claimed), phi1, phi2, phi3, phi4, phi5, phi6))
print(f"[{'PASS' if (deg_BI6==5 and deg_raw==7) else 'FAIL'}] total degree of BI6 is {deg_BI6} (paper claims 5); "
      f"total degree of the raw equation is {deg_raw} (paper claims 7)")
results.append(deg_BI6 == 5 and deg_raw == 7)

# Also report the degree of the AR4 and AR5 compact conditions, referenced in Section 4.7's
# discussion of complexity growth (degree 3, 4, 5 for AR4, AR5, AR6 respectively).
deg_AR4 = sp.total_degree(sp.Poly(sp.expand(lhs_iv), phi1, phi2, phi3, phi4))
deg_AR5 = sp.total_degree(sp.Poly(sp.expand(E_claimed), phi1, phi2, phi3, phi4, phi5))
print(f"[INFO] total degree of the AR(4) compact condition: {deg_AR4} (paper claims 3)")
print(f"[INFO] total degree of the AR(5) compact condition: {deg_AR5} (paper claims 4)")

print()
print("="*70)
n_pass = sum(1 for r in results if r)
print(f"TOTAL: {n_pass}/{len(results)} checks passed")
print("="*70)
if n_pass != len(results):
    raise SystemExit(1)
