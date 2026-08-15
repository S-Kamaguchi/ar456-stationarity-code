import numpy as np

def ground_truth_stationary(phi):
    # phi = [phi1,...,phip], inverse char eq: x^p - (phi1 x^{p-1} + ... + phip) = 0
    p = len(phi)
    coeffs = np.concatenate(([1.0], -np.array(phi)))  # x^p - phi1 x^{p-1} - ... - phip
    roots = np.roots(coeffs)
    return np.all(np.abs(roots) < 1.0 - 1e-9)

def make_stationary_batch(p, N, rng):
    # reflection coefficient (Levinson-Durbin) parametrization -> guaranteed true stationary points
    k = rng.uniform(-1, 1, size=(N, p))
    phi = np.zeros((N, p+1))
    for m in range(1, p+1):
        km = k[:, m-1]
        new = phi.copy()
        new[:, m] = km
        for i in range(1, m):
            new[:, i] = phi[:, i] - km*phi[:, m-i]
        phi = new
    return phi[:, 1:p+1]

# ---------------- AR(4) ----------------
def cond_AR4(P):
    p1,p2,p3,p4 = P[:,0],P[:,1],P[:,2],P[:,3]
    c1 = (1-p2-p4) > np.abs(p1+p3)
    c2 = (p3+p1*p4)*(p1-p3) + (1+p4)**2*(1+p2-p4) > 0
    c3 = 2*(1+p4) > np.abs(p1-p3)
    return c1 & c2 & c3

# ---------------- AR(5) ----------------
def cond_AR5(P):
    p1,p2,p3,p4,p5 = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4]
    c1 = (1-p2-p4) > np.abs(p1+p3+p5)
    c2 = np.abs(p5) < 1
    A = 1+p4+p1*p5-p5**2
    B = p1-p3-p2*p5+p4*p5
    c3 = B**2 + A*B*(p5-p1) + A**2*(-1-p2+p4) < 0
    c4 = 2*A > np.abs(B)
    return c1 & c2 & c3 & c4

# ---------------- AR(6) (using BI_6 compact form) ----------------
def BI6(P):
    p1,p2,p3,p4,p5,p6 = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4],P[:,5]
    T = -1-p2+p4-p6
    U = p6**2-p6*p2-p4-1
    V = p5+p1*p6
    W = p1-p3+p5
    term1 = -p1*V*T**2
    term2 = -T*(p6-p2-2)*(p5-p1)*V
    term3 = -U**2*T
    term4 = (p5-p1)**2*V*W
    term5 = (1+p6)*( 2*V*(p6-p4+p2+1)*W + U*W*(p5-p1) )
    term6 = -(1+p6)**3*W**2
    return term1+term2+term3+term4+term5+term6

def cond_AR6(P):
    p1,p2,p3,p4,p5,p6 = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4],P[:,5]
    c1 = (1-p2-p4-p6) > np.abs(p1+p3+p5)
    c2 = (1-p6**2) > np.abs(p5+p1*p6)
    c3 = BI6(P) > 0
    U = p6**2-p6*p2-p4-1
    V = p5+p1*p6
    W = p1-p3+p5
    lhs = -2*( (1+p6)*U + (-p1+p5)*V )
    rhs = np.abs( V*(p6-p4+p2+1) - (1+p6)**2*W )
    c4 = lhs > rhs
    return c1 & c2 & c3 & c4

def run_test(p, cond_fn, N_random, N_stationary, box, seed):
    rng = np.random.default_rng(seed)
    results = {}

    # Test A: uniform random box (mostly non-stationary; tests necessity broadly)
    P = rng.uniform(-box, box, size=(N_random, p))
    gt = np.array([ground_truth_stationary(P[i]) for i in range(N_random)])
    pred = cond_fn(P)
    mismatch = gt != pred
    results['random_box'] = (N_random, mismatch.sum(), gt.sum())

    # Test B: guaranteed-stationary points via reflection coefficients
    Ps = make_stationary_batch(p, N_stationary, rng)
    gts = np.array([ground_truth_stationary(Ps[i]) for i in range(N_stationary)])
    preds = cond_fn(Ps)
    assert gts.all(), "reflection param failed to produce stationary points!"
    mismatch_s = gts != preds
    results['guaranteed_stationary'] = (N_stationary, mismatch_s.sum())

    return results

for name, p, cond_fn, box in [('AR(4)',4,cond_AR4,3.0), ('AR(5)',5,cond_AR5,2.0), ('AR(6)',6,cond_AR6,1.5)]:
    print(f"=== {name} ===")
    res = run_test(p, cond_fn, N_random=50000, N_stationary=50000, box=box, seed=42)
    print("  random-box test: N=%d, mismatches=%d, #ground-truth-stationary-found=%d" % res['random_box'])
    print("  guaranteed-stationary test: N=%d, mismatches=%d" % res['guaranteed_stationary'])

print()
print("################ Direct Schur-Cohn (Levinson-Durbin step-down) cross-check ################")

def schur_cohn_stationary(phi):
    # step the Levinson-Durbin recursion DOWN from order p to order 1,
    # extracting reflection coefficients; stationary iff all |k_m|<1.
    phi = np.array(phi, dtype=float)
    p = len(phi)
    cur = phi.copy()
    for m in range(p, 0, -1):
        km = cur[m-1]
        if abs(km) >= 1.0:
            return False
        if m == 1:
            break
        new = np.zeros(m-1)
        denom = 1 - km**2
        for i in range(1, m):
            new[i-1] = (cur[i-1] + km*cur[m-i-1]) / denom
        cur = new
    return True

def run_schur_cohn_crosscheck(p, cond_fn, N_random, N_stationary, box, seed):
    rng = np.random.default_rng(seed+1)
    P = rng.uniform(-box, box, size=(N_random, p))
    sc = np.array([schur_cohn_stationary(P[i]) for i in range(N_random)])
    gt = np.array([ground_truth_stationary(P[i]) for i in range(N_random)])
    pred = cond_fn(P)
    print(f"  [random box, N={N_random}] Schur-Cohn vs root-finding mismatches: {(sc!=gt).sum()}")
    print(f"  [random box, N={N_random}] proposed-criterion vs Schur-Cohn mismatches: {(pred!=sc).sum()}")

    Ps = make_stationary_batch(p, N_stationary, rng)
    scs = np.array([schur_cohn_stationary(Ps[i]) for i in range(N_stationary)])
    preds = cond_fn(Ps)
    print(f"  [guaranteed-stationary, N={N_stationary}] Schur-Cohn says non-stationary count (should be 0): {(~scs).sum()}")
    print(f"  [guaranteed-stationary, N={N_stationary}] proposed-criterion vs Schur-Cohn mismatches: {(preds!=scs).sum()}")

for name, p, cond_fn, box in [('AR(4)',4,cond_AR4,3.0), ('AR(5)',5,cond_AR5,2.0), ('AR(6)',6,cond_AR6,1.5)]:
    print(f"=== {name} ===")
    run_schur_cohn_crosscheck(p, cond_fn, N_random=100000, N_stationary=100000, box=box, seed=123)

print()
print("################ Boundary stress test (points near the stationarity boundary) ################")

def make_near_boundary_batch(p, N, rng, eps_range=(1e-4, 0.05)):
    # reflection coefficients pushed close to +-1 (near boundary) and close to 0 (near center),
    # to stress-test near the true boundary of St(p)
    k = rng.uniform(-1, 1, size=(N, p))
    # push a random subset of coordinates close to +-1
    close = rng.uniform(*eps_range, size=(N,p))
    sign = np.sign(k)
    mask = rng.random((N,p)) < 0.5
    k = np.where(mask, sign*(1-close), k)
    phi = np.zeros((N, p+1))
    for m in range(1, p+1):
        km = k[:, m-1]
        new = phi.copy()
        new[:, m] = km
        for i in range(1, m):
            new[:, i] = phi[:, i] - km*phi[:, m-i]
        phi = new
    return phi[:, 1:p+1]

for name, p, cond_fn, box in [('AR(4)',4,cond_AR4,3.0), ('AR(5)',5,cond_AR5,2.0), ('AR(6)',6,cond_AR6,1.5)]:
    rng = np.random.default_rng(7)
    Pb = make_near_boundary_batch(p, 50000, rng)
    gtb = np.array([ground_truth_stationary(Pb[i]) for i in range(50000)])
    predb = cond_fn(Pb)
    mism = (gtb != predb).sum()
    print(f"{name}: near-boundary N=50000, mismatches={mism} (all should be stationary by construction: {gtb.sum()}/50000)")

print()
print("################ Checking whether boundary mismatches are float-precision artifacts ################")
rng = np.random.default_rng(7)
Pb = make_near_boundary_batch(4, 50000, rng)
gtb = np.array([ground_truth_stationary(Pb[i]) for i in range(50000)])
predb = cond_AR4(Pb)
mismask = gtb != predb
idxs = np.where(mismask)[0][:5]
for idx in idxs:
    phi = Pb[idx]
    coeffs = np.concatenate(([1.0], -phi))
    roots = np.roots(coeffs)
    maxabs = np.max(np.abs(roots))
    p1,p2,p3,p4 = phi
    margin1 = (1-p2-p4) - abs(p1+p3)
    margin3 = 2*(1+p4) - abs(p1-p3)
    print(f"phi={phi}, max|root|={maxabs:.10f} (dist from 1: {abs(maxabs-1):.2e}), margins: c1={margin1:.2e}, c3={margin3:.2e}")
