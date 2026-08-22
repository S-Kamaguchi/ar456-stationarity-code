# Reproducibility code

Supporting code for:

Shōichirō Kamaguchi, "Explicit Stationarity Regions for AR(4), AR(5), and AR(6) via Unit-Circle Analysis."

Four standalone scripts, each reproducing a different set of computations or checks reported in the paper. All require only `numpy`, `sympy`, `matplotlib`, and `mpmath` (standard scientific Python; no internet access needed).

## `numerical_verification_ar456.py`

Reproduces the numerical verification of Section 5: for each of AR(4), AR(5), and AR(6), samples coefficient vectors via (i) uniform random sampling, (ii) the reflection-coefficient (Levinson–Durbin) parametrization (guaranteed stationary), and (iii) a near-boundary stress test, then compares the derived stationarity criterion against direct root computation and against an independent Schur–Cohn step-down implementation.

```
python3 numerical_verification_ar456.py
```

Note on the direct-root reference (Section 5.3): the direct-root reference used throughout this script (ground_truth_stationary) classifies a point as stationary only when every root's modulus is below 1 - 1e-9, not a bare < 1. This is a deliberate safety margin, not merely floating-point rounding noise: it is applied identically everywhere the direct-root reference is used, and it is the reason a small fraction of points that are genuinely stationary (arbitrarily close to, but strictly inside, the unit circle) are reported as "mismatches" against the derived criterion and against the Schur–Cohn test in the near-boundary stress test only. The derived criterion and the Schur–Cohn test, neither of which relies on root-finding, do not use this margin and agree with each other (see the paper's Section 5.3 for the full discussion, including a small number of AR(6) double-precision edge cases that resolve to agreement under higher-precision arithmetic, reproduced separately by verify_60digit_reevaluation.py below).

## `verify_algebra.py`

Symbolically verifies, using `sympy`, the principal algebraic elimination steps used to derive the AR(4)/AR(5)/AR(6) closed-form criteria in Section 4 (the degree-reduction steps via Chebyshev identities, the linear-elimination steps, and the BI6 compaction identity for AR(6)). Each step is checked for *exact* symbolic equality against the equation stated in the paper, not mere numerical agreement.

```
python3 verify_algebra.py
```

Expected output: 15/15 checks PASS.

## `verify_60digit_reevaluation.py`

Reproduces the 60-digit arbitrary-precision re-evaluation cited in Section 5.3's results-table footnote: identifies the 7 near-boundary AR(6) points (seed=7) where the derived criterion and the Schur–Cohn test disagree in double precision, then re-evaluates both using `mpmath` at 60 decimal digits and confirms all 7 resolve to agreement at 60-digit precision.

```
python3 verify_60digit_reevaluation.py
```

Expected output: 7/7 resolved.

## `generate_figures.py`

Regenerates all five figures used in the paper. The AR(2) triangle (Figure 3) is drawn directly from the known closed-form region, with no validation points at all. The AR(3) triangular cross-sections (Figure 4) and the AR(4) 2D cross-section (Figure 1) each shade or scatter their validation points by direct root computation (is_stationary, via numpy.roots), with the derived boundary curve overlaid independently in Figure 1. The two full 3D regions (Figure 2, AR(4); Figure 5, AR(3)) are rendered differently: their surfaces are drawn directly from the closed-form/derived-inequality description of the region (cond_AR4 for Figure 2; the AR(3) closed form for Figure 5), and their red interior points are generated independently via the reflection-coefficient (Levinson–Durbin forward) parametrization, which produces guaranteed-stationary points by construction rather than by classifying arbitrary points via root-finding or the Schur–Cohn test.

```
python3 generate_figures.py [output_directory]
```

## Provenance

`numerical_verification_ar456.py` is the exact script referenced throughout the manuscript's numerical verification section. `verify_algebra.py`, `generate_figures.py`, and `verify_60digit_reevaluation.py` were written afterward, specifically to make the paper's "verified symbolically", figure-generation, and 60-digit re-evaluation claims independently checkable and re-runnable rather than asserted.
