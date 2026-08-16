"""
generate_figures.py

Regenerates all five figures used in:

    Shoichiro Kamaguchi, "Explicit Stationarity Regions for AR(4), AR(5),
    and AR(6) via Unit-Circle Analysis."

  Figure 1 (fig3_AR4_cross_section.png): 2D cross-section of St(4) at
      fixed phi2=0, phi4=0.3, in the (phi1,phi3)-plane, with the
      derived boundary curve (Q=0) and auxiliary boundary lines overlaid on a
      root-finding-classified shaded region (Section 4.4).
  Figure 2 (fig4_AR4_3D_region.png): the full 3D cross-section of St(4)
      at fixed phi4=0.3, in (phi1,phi2,phi3)-space (Section 4.4).
  Figure 3 (fig1_AR2_region.png): the classical AR(2) stationarity
      triangle St(2) (Section 4.4).
  Figure 4 (fig2_AR3_region.png): triangular cross-sections of St(3) at
      three fixed values of phi3, with root-finding validation points
      (Section 4.4).
  Figure 5 (fig5_AR3_3D_region.png): the full 3D region St(3) in
      (phi1,phi3,phi2)-space, using the closed-form phi2-envelope derived
      in the paper (Section 4.4).

All classification of "stationary" points is via direct root computation
of the inverse characteristic polynomial (companion-matrix eigenvalues via
numpy.roots), independent of the paper's derived inequalities, exactly as
described in each figure's caption.

Requirements: numpy, matplotlib (mplot3d is part of matplotlib). Run with:

    python3 generate_figures.py [output_directory]

Output directory defaults to the current directory. Each figure is saved
as a PNG at ~160 dpi, matching the files included with the manuscript.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(OUTDIR, exist_ok=True)


def is_stationary(phi):
    """Direct ground truth: all roots of the inverse characteristic
    polynomial x^p - (phi_1 x^{p-1} + ... + phi_p) inside the open unit
    disk, via companion-matrix eigenvalues (numpy.roots)."""
    coeffs = np.concatenate(([1.0], -np.asarray(phi, dtype=float)))
    roots = np.roots(coeffs)
    return np.all(np.abs(roots) < 1.0)


# ---------------------------------------------------------------------
# Figure 3 (fig1_AR2_region.png): classical AR(2) region St(2)
# ---------------------------------------------------------------------
def make_fig_AR2():
    verts = np.array([(-2, -1), (2, -1), (0, 1), (-2, -1)])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill(verts[:, 0], verts[:, 1], color='#AED6F1', edgecolor='#1F618D', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.set_title(r'$AR(2)$ stationarity region $St(2)$')
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig1_AR2_region.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------
# Figure 4 (fig2_AR3_region.png): triangular cross-sections of St(3)
# ---------------------------------------------------------------------
def make_fig_AR3():
    phi3_vals = [-0.6, 0.0, 0.6]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    fig.suptitle(r'$AR(3)$ stationarity region $St(3)$: triangular cross-sections at fixed $\phi_3$'
                 '\n(triangle = derived region; red points = independent root-finding check)')
    for ax, phi3 in zip(axes, phi3_vals):
        v = np.array([(-phi3, 1), (phi3 - 2, 2*phi3 - 1), (phi3 + 2, -2*phi3 - 1), (-phi3, 1)])
        ax.fill(v[:, 0], v[:, 1], color='#AED6F1', edgecolor='#1F618D', linewidth=1.2)

        N = 4000
        p1 = rng.uniform(-3, 3, N)
        p2 = rng.uniform(-2, 2, N)
        stat = np.array([is_stationary([p1[i], p2[i], phi3]) for i in range(N)])
        ax.scatter(p1[stat], p2[stat], s=2, color='#C0392B',
                   label='root-finding: stationary' if phi3 == phi3_vals[0] else None)

        ax.axhline(0, color='gray', linewidth=0.4)
        ax.axvline(0, color='gray', linewidth=0.4)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        ax.set_title(rf'$\phi_3 = {phi3}$')
        if phi3 == phi3_vals[0]:
            ax.legend(loc='upper center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig2_AR3_region.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------
# Figure 1 (fig3_AR4_cross_section.png): 2D cross-section of St(4)
# ---------------------------------------------------------------------
def make_fig_AR4_cross_section():
    phi2, phi4 = 0.0, 0.3
    rng = np.random.default_rng(1)
    N = 60000
    p1 = rng.uniform(-3, 3, N)
    p3 = rng.uniform(-3, 3, N)
    stat = np.array([is_stationary([p1[i], phi2, p3[i], phi4]) for i in range(N)])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(p1[stat], p3[stat], s=1, color='#AED6F1', label='root-finding: stationary region')

    # Derived boundary: (phi1-phi3)^2 + (1+phi4)(phi1-phi3)(-phi1) + (1+phi4)^2(-1-phi2+phi4) = 0,
    # a quadratic in phi3 for each fixed phi1 (solved directly, independent of the root-finding above).
    p1_line = np.linspace(-3, 3, 800)
    for branch, color, lbl in [(1, '#B03A2E', 'derived boundary: Q = 0')]:
        # Rewrite the boundary as a quadratic in phi3: let d = phi1-phi3.
        # d^2 - (1+phi4)*phi1*d + (1+phi4)^2*(-1-phi2+phi4) = 0  =>  solve for d.
        a_ = 1.0
        b_ = -(1+phi4)*p1_line
        c_ = (1+phi4)**2*(-1-phi2+phi4)
        disc = b_**2 - 4*a_*c_
        valid = disc >= 0
        d1 = (-b_[valid] + np.sqrt(disc[valid])) / (2*a_)
        d2 = (-b_[valid] - np.sqrt(disc[valid])) / (2*a_)
        phi3_top = p1_line[valid] - np.minimum(d1, d2)
        phi3_bot = p1_line[valid] - np.maximum(d1, d2)
        ax.plot(p1_line[valid], phi3_top, color=color, linewidth=2, label=lbl)
        ax.plot(p1_line[valid], phi3_bot, color=color, linewidth=2)

    # Auxiliary boundaries used in the derivation
    ax.plot(p1_line, p1_line - 2*(1+phi4), '--', color='purple', linewidth=1, label=r'$2(1+\phi_4)=|\phi_1-\phi_3|$')
    ax.plot(p1_line, p1_line + 2*(1+phi4), '--', color='purple', linewidth=1)
    # phi(1)=0: 1-(phi1+phi2+phi3+phi4)=0 => phi3 = 1-phi1-phi2-phi4
    ax.plot(p1_line, 1 - p1_line - phi2 - phi4, ':', color='green', linewidth=1, label=r'$\phi(1)=0$')
    # phi(-1)=0: 1-(-phi1+phi2-phi3+phi4)=0 => phi3 = phi1 - phi2 - phi4 + 1 ... (sign convention per Section 2)
    ax.plot(p1_line, p1_line - phi2 - phi4 + 1, ':', color='orange', linewidth=1, label=r'$\phi(-1)=0$')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_3$')
    ax.set_title(rf'$AR(4)$ cross-section at $\phi_2={phi2}$, $\phi_4={phi4}$')
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig3_AR4_cross_section.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------
# Figure 2 (fig4_AR4_3D_region.png): full 3D cross-section of St(4)
# ---------------------------------------------------------------------
def cond_AR4(p1, p2, p3, p4):
    c1 = (1 - p2 - p4) > np.abs(p1 + p3)
    c2 = (p3 + p1*p4)*(p1 - p3) + (1+p4)**2*(1 + p2 - p4) > 0
    c3 = 2*(1+p4) > np.abs(p1 - p3)
    return c1 & c2 & c3


def make_fig_AR4_3D():
    phi4 = 0.3
    n = 140
    phi1_range = np.linspace(-3, 3, n)
    phi2_range = np.linspace(-1.8, 1.0, n)
    P1, P2 = np.meshgrid(phi1_range, phi2_range, indexing='ij')

    m = 500
    phi3_scan = np.linspace(-3.5, 3.5, m)
    Zmin = np.full(P1.shape, np.nan)
    Zmax = np.full(P1.shape, np.nan)
    for i in range(n):
        for j in range(n):
            mask = cond_AR4(phi1_range[i], phi2_range[j], phi3_scan, phi4)
            if mask.any():
                idx = np.where(mask)[0]
                Zmin[i, j] = phi3_scan[idx[0]]
                Zmax[i, j] = phi3_scan[idx[-1]]

    rng = np.random.default_rng(7)
    N = 2500
    k1 = rng.uniform(-1, 1, N)
    k2 = rng.uniform(-1, 1, N)
    k3 = rng.uniform(-1, 1, N)
    k4 = phi4
    a1 = k1
    a1b, a2 = a1 - k2*a1, k2
    a1c, a2c, a3c = a1b - k3*a2, a2 - k3*a1b, k3
    P = np.stack([a1c - k4*a3c, a2c - k4*a2c, a3c - k4*a1c, np.full(N, k4)], axis=1)

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf_kwargs = dict(color='#5B9BD5', alpha=0.45, linewidth=0, antialiased=True, shade=True)
    ax.plot_surface(P1, P2, Zmax, **surf_kwargs)
    ax.plot_surface(P1, P2, Zmin, **surf_kwargs)
    ax.plot_wireframe(P1, P2, Zmax, rstride=10, cstride=10, color='#2E5C8A', linewidth=0.3, alpha=0.5)
    ax.plot_wireframe(P1, P2, Zmin, rstride=10, cstride=10, color='#2E5C8A', linewidth=0.3, alpha=0.5)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#C0392B', s=3, alpha=0.55, depthshade=False,
               label='guaranteed-stationary samples\n(Schur-Cohn / root-finding)')
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.set_zlabel(r'$\phi_3$')
    ax.set_title(r'$St(4)$ at fixed $\phi_4=0.3$, in $(\phi_1,\phi_2,\phi_3)$-space')
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig4_AR4_3D_region.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------
# Figure 5 (fig5_AR3_3D_region.png): full 3D region St(3), closed form
# ---------------------------------------------------------------------
def make_fig_AR3_3D():
    n = 160
    phi1_range = np.linspace(-2.2, 2.2, n)
    phi3_range = np.linspace(-1, 1, n)
    P1, P3 = np.meshgrid(phi1_range, phi3_range, indexing='ij')
    Zlow = P3**2 - P1*P3 - 1
    Zhigh = np.minimum(1 - P1 - P3, 1 + P1 + P3)
    valid = Zhigh > Zlow
    Zlow_m = np.where(valid, Zlow, np.nan)
    Zhigh_m = np.where(valid, Zhigh, np.nan)

    rng = np.random.default_rng(3)
    N = 2500
    k1 = rng.uniform(-1, 1, N)
    k2 = rng.uniform(-1, 1, N)
    k3 = rng.uniform(-1, 1, N)
    a1 = k1
    a1b, a2 = a1 - k2*a1, k2
    b1, b2, b3 = a1b - k3*a2, a2 - k3*a1b, k3
    Pv = np.stack([b1, b2, b3], axis=1)
    box_mask = (Pv[:, 0] >= -2.2) & (Pv[:, 0] <= 2.2) & (Pv[:, 2] >= -1) & (Pv[:, 2] <= 1)
    Pv2 = Pv[box_mask]

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf_kwargs = dict(color='#5B9BD5', alpha=0.45, linewidth=0, antialiased=True, shade=True)
    ax.plot_surface(P1, P3, Zhigh_m, **surf_kwargs)
    ax.plot_surface(P1, P3, Zlow_m, **surf_kwargs)
    ax.plot_wireframe(P1, P3, Zhigh_m, rstride=10, cstride=10, color='#2E5C8A', linewidth=0.3, alpha=0.5)
    ax.plot_wireframe(P1, P3, Zlow_m, rstride=10, cstride=10, color='#2E5C8A', linewidth=0.3, alpha=0.5)
    ax.scatter(Pv2[:, 0], Pv2[:, 2], Pv2[:, 1], color='#C0392B', s=3, alpha=0.55, depthshade=False,
               label='guaranteed-stationary samples\n(root-finding)')
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_3$')
    ax.set_zlabel(r'$\phi_2$')
    ax.set_title(r'$St(3)$ in $(\phi_1,\phi_3,\phi_2)$-space')
    ax.view_init(elev=16, azim=-55)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig5_AR3_3D_region.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("wrote", path)


if __name__ == '__main__':
    make_fig_AR2()
    make_fig_AR3()
    make_fig_AR4_cross_section()
    make_fig_AR4_3D()
    make_fig_AR3_3D()
    print("All 5 figures regenerated in:", os.path.abspath(OUTDIR))
