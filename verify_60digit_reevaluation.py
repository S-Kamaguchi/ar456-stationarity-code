"""
verify_60digit_reevaluation.py

Reproduces the 60-digit arbitrary-precision re-evaluation described in
Section 5.3 ("Results") of:

    Shoichiro Kamaguchi, "Explicit Stationarity Regions for AR(4), AR(5),
    and AR(6) via Unit-Circle Analysis."

Section 5.3 reports that, of the 50,000 near-boundary AR(6) points
(seed=7, the same batch used for results table), 7 show a disagreement between
the derived criterion and the Schur-Cohn test when both are evaluated in
IEEE 754 double precision, and that re-evaluating these 7 points in
60-digit arbitrary-precision arithmetic (Python's mpmath library)
resolves every one of them in favor of agreement.

This script performs that re-evaluation explicitly:

  1. Reproduces the exact near-boundary AR(6) batch from
     numerical_verification_ar456.py (make_near_boundary_batch, seed=7,
     N=50000), and identifies the double-precision disagreements between
     cond_AR6 (the derived criterion) and schur_cohn_stationary (the
     Schur-Cohn / Levinson-Durbin step-down test) -- this reproduces the
     "7" cited in results table's footnote.
  2. Re-implements both cond_AR6 and schur_cohn_stationary using mpmath
     at 60 decimal digits of precision (mp.mp.dps = 60), and re-evaluates
     each of the 7 disagreeing points.
  3. Reports, for each point, the double-precision verdict from both
     methods alongside the 60-digit verdict from both methods, and
     confirms that all 7 resolve to agreement at 60 digits.

Requirements: numpy, mpmath. Run with:

    python3 verify_60digit_reevaluation.py
"""

import numpy as np
import mpmath as mp

# ---- double-precision functions, identical to numerical_verification_ar456.py ----

def BI6(P):
    p1, p2, p3, p4, p5, p6 = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5]
    T = -1 - p2 + p4 - p6
    U = p6 ** 2 - p6 * p2 - p4 - 1
    V = p5 + p1 * p6
    W = p1 - p3 + p5
    term1 = -p1 * V * T ** 2
    term2 = -T * (p6 - p2 - 2) * (p5 - p1) * V
    term3 = -U ** 2 * T
    term4 = (p5 - p1) ** 2 * V * W
    term5 = (1 + p6) * (2 * V * (p6 - p4 + p2 + 1) * W + U * W * (p5 - p1))
    term6 = -(1 + p6) ** 3 * W ** 2
    return term1 + term2 + term3 + term4 + term5 + term6


def cond_AR6(P):
    p1, p2, p3, p4, p5, p6 = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5]
    c1 = (1 - p2 - p4 - p6) > np.abs(p1 + p3 + p5)
    c2 = (1 - p6 ** 2) > np.abs(p5 + p1 * p6)
    c3 = BI6(P) > 0
    U = p6 ** 2 - p6 * p2 - p4 - 1
    V = p5 + p1 * p6
    W = p1 - p3 + p5
    lhs = -2 * ((1 + p6) * U + (-p1 + p5) * V)
    rhs = np.abs(V * (p6 - p4 + p2 + 1) - (1 + p6) ** 2 * W)
    c4 = lhs > rhs
    return c1 & c2 & c3 & c4


def schur_cohn_stationary(phi):
    phi = np.array(phi, dtype=float)
    p = len(phi)
    cur = phi.copy()
    for m in range(p, 0, -1):
        km = cur[m - 1]
        if abs(km) >= 1.0:
            return False
        if m == 1:
            break
        new = np.zeros(m - 1)
        denom = 1 - km ** 2
        for i in range(1, m):
            new[i - 1] = (cur[i - 1] + km * cur[m - i - 1]) / denom
        cur = new
    return True


def make_near_boundary_batch(p, N, rng, eps_range=(1e-4, 0.05)):
    k = rng.uniform(-1, 1, size=(N, p))
    close = rng.uniform(*eps_range, size=(N, p))
    sign = np.sign(k)
    mask = rng.random((N, p)) < 0.5
    k = np.where(mask, sign * (1 - close), k)
    phi = np.zeros((N, p + 1))
    for m in range(1, p + 1):
        km = k[:, m - 1]
        new = phi.copy()
        new[:, m] = km
        for i in range(1, m):
            new[:, i] = phi[:, i] - km * phi[:, m - i]
        phi = new
    return phi[:, 1:p + 1]


# ---- 60-digit mpmath re-implementations ----

mp.mp.dps = 60


def BI6_mp(phi):
    p1, p2, p3, p4, p5, p6 = [mp.mpf(x) for x in phi]
    T = -1 - p2 + p4 - p6
    U = p6 ** 2 - p6 * p2 - p4 - 1
    V = p5 + p1 * p6
    W = p1 - p3 + p5
    term1 = -p1 * V * T ** 2
    term2 = -T * (p6 - p2 - 2) * (p5 - p1) * V
    term3 = -U ** 2 * T
    term4 = (p5 - p1) ** 2 * V * W
    term5 = (1 + p6) * (2 * V * (p6 - p4 + p2 + 1) * W + U * W * (p5 - p1))
    term6 = -(1 + p6) ** 3 * W ** 2
    return term1 + term2 + term3 + term4 + term5 + term6


def cond_AR6_mp(phi):
    p1, p2, p3, p4, p5, p6 = [mp.mpf(x) for x in phi]
    c1 = (1 - p2 - p4 - p6) > abs(p1 + p3 + p5)
    c2 = (1 - p6 ** 2) > abs(p5 + p1 * p6)
    c3 = BI6_mp(phi) > 0
    U = p6 ** 2 - p6 * p2 - p4 - 1
    V = p5 + p1 * p6
    W = p1 - p3 + p5
    lhs = -2 * ((1 + p6) * U + (-p1 + p5) * V)
    rhs = abs(V * (p6 - p4 + p2 + 1) - (1 + p6) ** 2 * W)
    c4 = lhs > rhs
    return c1 and c2 and c3 and c4


def schur_cohn_stationary_mp(phi):
    cur = [mp.mpf(x) for x in phi]
    p = len(cur)
    for m in range(p, 0, -1):
        km = cur[m - 1]
        if abs(km) >= 1:
            return False
        if m == 1:
            break
        denom = 1 - km ** 2
        new = [(cur[i - 1] + km * cur[m - i - 1]) / denom for i in range(1, m)]
        cur = new
    return True


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    Pb = make_near_boundary_batch(6, 50000, rng)
    derived = cond_AR6(Pb)
    sc = np.array([schur_cohn_stationary(Pb[i]) for i in range(50000)])
    mismatch_idx = np.where(derived != sc)[0]

    print(f"Double-precision disagreements (derived vs Schur-Cohn), "
          f"seed=7, N=50000: {len(mismatch_idx)}")
    print(f"(results table footnote reports 7 such disagreements.)\n")

    print("Re-evaluating each double-precision-mismatched point at 60-digit precision:")
    n_resolved = 0
    for idx in mismatch_idx:
        phi = Pb[idx].tolist()
        d64, s64 = bool(derived[idx]), bool(sc[idx])
        d_mp, s_mp = cond_AR6_mp(phi), schur_cohn_stationary_mp(phi)
        resolved = (d_mp == s_mp)
        n_resolved += resolved
        print(f"  idx={idx}: double-precision (derived={d64}, Schur-Cohn={s64})  "
              f"60-digit (derived={d_mp}, Schur-Cohn={s_mp})  resolved={resolved}")

    print(f"\n{n_resolved}/{len(mismatch_idx)} resolved to agreement at 60-digit precision.")
