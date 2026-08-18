# Reproducibility code

Supporting code for:

Shōichirō Kamaguchi, "Explicit Stationarity Regions for AR(4), AR(5), and AR(6) via Unit-Circle Analysis."

Four independent scripts, each reproducing a different set of claims made in the paper. All require only `numpy`, `sympy`, `matplotlib`, and `mpmath` (standard scientific Python; no internet access needed).

## `numerical_verification_ar456.py`

Reproduces the numerical verification of Section 5: for each of AR(4), AR(5), and AR(6), samples coefficient vectors via (i) uniform random sampling, (ii) the reflection-coefficient (Levinson–Durbin) parametrization (guaranteed stationary), and (iii) a near-boundary stress test, then compares the derived stationarity criterion against direct root computation and against an independent Schur–Cohn step-down implementation.

```
python3 numerical_verification_ar456.py
```

Note on the near-boundary stress test (Section 5.3): direct root computation (`numpy.roots`) disagrees with the other two methods when a sampled root lies extremely close to the unit circle, which occurs for a small fraction of the near-boundary sample by construction. This shows up as apparent "mismatches" against root-finding in that section only; the derived criterion and the Schur–Cohn test, which do not rely on root-finding, do not exhibit this and agree with each other (see the paper's Section 5.3 for the full discussion, including a small number of AR(6) double-precision edge cases that resolve to agreement under higher-precision arithmetic).

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

Regenerates all five figures used in the paper (the AR(2) triangle, the AR(3) cross-sections, the AR(4) 2D cross-section with derived boundary curve, and the full 3D regions for AR(3) and AR(4)). All stationarity classification used for the scatter/validation points is via direct root computation, independent of the derived inequalities.

```
python3 generate_figures.py [output_directory]
```

## Provenance

`numerical_verification_ar456.py` is the exact script referenced throughout the manuscript's numerical verification section. `verify_algebra.py`, `generate_figures.py`, and `verify_60digit_reevaluation.py` were written afterward, specifically to make the paper's "verified symbolically", figure-generation, and 60-digit re-evaluation claims independently checkable and re-runnable rather than asserted.
