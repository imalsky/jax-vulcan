"""Molecular-diffusion benchmark against the Zhang, Shia & Yung (2013) analytic
diffusive-separation solution.

Zhang, X., Shia, R.-L., & Yung, Y. L. 2013, ApJ, 767, 172,
"Jovian Stratosphere as a Chemical Transport System: Benchmark Analytical
Solutions." The paper gives closed-form steady states of the 1D
advection-diffusion(-reaction) equation for validating transport solvers. The
molecular-diffusion case is the classic diffusive-separation profile: a minor
species of molecular mass m_i in a background of mean mass mu, subject to eddy
diffusion K and molecular diffusion D, relaxes to

    f_i(z) = f_i(z0) * exp[ -(D/(D+K)) * (1/H_i - 1/H_atm) * (z - z0) ]        (1)

where f_i is the mole fraction, H_i = R T / (m_i g) is the species scale height,
H_atm = R T / (mu g) is the mean (pressure) scale height, and z increases
upward. Eq. (1) is the zero-total-flux steady state of

    d/dz [ (D+K) df_i/dz + D f_i (1/H_i - 1/H_atm) ] = 0.

WHY THIS BENCHMARK. VULCAN-JAX carries two discretizations of the molecular
drift term D f (1/H_i - 1/H_atm), and each has one failure mode:
  * central "gravity" scheme (use_vm_mol=False) -- 2nd-order ACCURATE (matches
    Eq. 1), but UNSTABLE: for cell Peclet number |vm| dz / D > 2 it develops a
    spurious sign-alternating mode and the steady state goes negative (the TVD
    violation of the standard scheme);
  * 1st-order upwind "vm" scheme (use_vm_mol=True) -- unconditionally STABLE but
    dissipative (numerical diffusion ~ |vm| dz / 2, so it under-separates).
Both target the SAME continuous flux (vm = -D (1/H_i - 1/H_atm), so upwind
advection of vm*n and the central gravity term are the same PDE). VULCAN's
production scheme is the HYBRID: converge under upwind (stable), then finish in
central (accurate). A completed hybrid run ends in the central phase, so its
converged state is the accurate central curve -- with none of the instability.

This script drives the production VULCAN-JAX diffusion kernels
(`jax_step.compute_diff_grav` + `_build_diff_coeffs_jax`) on a synthetic
isothermal, constant-gravity column, solving the discrete steady state directly
(bottom mole fraction pinned, zero-flux top). Two panels:
  A. Accuracy (well-resolved grid): the central/hybrid steady state matches
     Eq. (1); pure upwind is dissipative. Overlaid with the upstream VULCAN 2.0
     operator (op.diffdf / op.diffdf_vm, via the validated transcription in
     tests/diffusion_numpy_ref.py) to show VULCAN-JAX-vs-upstream port fidelity.
  B. Stability (pure molecular, K=0, grid resolution swept): most-negative mole
     fraction vs cell Peclet. Central goes negative for Pe > 2; upwind stays
     positive at every Pe -- which is exactly why the hybrid needs it.

Run (in the `vulcan` conda env):
    python benchmarks/zhang2013_moldiff_benchmark.py
Writes benchmarks/zhang2013_moldiff_benchmark.png and prints an error table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from vulcan_jax.jax_step import (
    AtmStatic,
    _apply_diffusion_jax,
    _build_diff_coeffs_jax,
    compute_diff_grav,
)
from vulcan_jax.phy_const import Navo, kb

# The upstream VULCAN operator ("VULCAN 2.0": op.diffdf central / op.diffdf_vm
# upwind) is reproduced by tests/diffusion_numpy_ref.py, an FP-faithful NumPy
# transcription validated against the actual VULCAN-master operator to <1e-5 in
# tests/test_diffusion_variants.py. Import it here as the upstream reference.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import diffusion_numpy_ref as diffref  # noqa: E402

R_GAS = Navo * kb  # ideal-gas constant, erg mol^-1 K^-1 (= 8.314e7)

# --- Column parameters (Jovian-stratosphere-like, isothermal + constant g so
#     Eq. (1) is an exact exponential; the benchmark tests the discretization,
#     not the atmosphere model). -----------------------------------------------
T_ISO = 160.0  # isothermal temperature, K
G_CGS = 2479.0  # gravity, cm s^-2 (Jupiter)
MU_ATM = 2.3  # background mean molecular weight, g mol^-1 (H2/He)
M_TRACE = 16.0  # tracer molecular weight, g mol^-1 (CH4, heavy -> settles)
F0 = 1.0e-4  # tracer mole fraction pinned at the base (minor species)
NZ = 120  # number of cell centres
P_BOT = 1.0e3  # base pressure, dyn cm^-2 (1 mbar)
N_SCALE_HEIGHTS = 6.0  # column depth in mean scale heights (sets P_TOP)

# Accuracy panel (constant coefficients): K = R_KD * D  ->  D/(D+K) = 1/(1+R_KD).
D_CONST = 1.0e6  # molecular diffusion coefficient, cm^2 s^-1
R_KD = 4.0  # eddy/molecular ratio (D/(D+K) = 0.2)


def _fmt_err(fe: float) -> str:
    """Format a fractional error: a percentage below 100%, a factor above it
    (e.g. an upwind profile 16x off the analytic reads '16x', not '1500%')."""
    return f"{100 * fe:.1f}%" if fe < 1.0 else f"{fe + 1:.0f}x"


def build_column(nz=NZ, n_scale_heights=N_SCALE_HEIGHTS):
    """Isothermal, constant-gravity column geometry on `nz` levels spanning
    `n_scale_heights` mean scale heights.

    Returns a dict of NumPy arrays: centre grids (nz,) and interface grids
    (nz-1,). Heights increase upward with z[0] = 0 at the base.
    """
    H_atm = R_GAS * T_ISO / (MU_ATM * G_CGS)  # mean scale height, cm
    z_top = n_scale_heights * H_atm
    # Uniform height grid -> log-uniform pressure (isothermal, constant g).
    zco = np.linspace(0.0, z_top, nz)  # (nz,) cell centres, cm
    pco = P_BOT * np.exp(-zco / H_atm)  # (nz,) dyn cm^-2
    n_tot = pco / (kb * T_ISO)  # (nz,) total number density, cm^-3
    dzi = np.diff(zco)  # (nz-1,) centre-to-centre spacing
    z_iface = 0.5 * (zco[:-1] + zco[1:])  # (nz-1,)
    return {
        "nz": nz,
        "H_atm": H_atm,
        "zco": zco,
        "pco": pco,
        "n_tot": n_tot,
        "dzi": dzi,
        "z_iface": z_iface,
    }


def make_atm_static(col, Dzz_trace_iface, Kzz_iface, *, use_vm_mol):
    """Assemble an `AtmStatic` for the two-species (background + tracer) column.

    `Dzz_trace_iface` (nz-1,) is the tracer molecular-diffusion coefficient on
    interfaces; the background column has zero molecular diffusion. When
    `use_vm_mol` is True the interface drift `vm = -Dzz (1/H_i - 1/H_atm)` is
    populated (upwind scheme); otherwise the central gravity terms are used.
    """
    nz = col["nz"]
    ni = 2  # 0 = background, 1 = tracer
    zeros_i = np.zeros((nz - 1, ni), dtype=np.float64)

    Dzz = zeros_i.copy()
    Dzz[:, 1] = Dzz_trace_iface  # tracer only

    # Interface inverse scale heights (isothermal -> position-independent here,
    # but computed from the arrays so the code matches recompute_vm_jax).
    inv_H_trace = M_TRACE * G_CGS / (R_GAS * T_ISO)
    inv_H_atm = MU_ATM * G_CGS / (R_GAS * T_ISO)
    drift = inv_H_trace - inv_H_atm  # (1/H_i - 1/H_atm), 1/cm
    vm = zeros_i.copy()
    if use_vm_mol:
        vm[:, 1] = -Dzz_trace_iface * drift  # matches atm_refresh.recompute_vm_jax

    return AtmStatic(
        Kzz=jnp.asarray(Kzz_iface),
        Dzz=jnp.asarray(Dzz),
        dzi=jnp.asarray(col["dzi"]),
        vz=jnp.zeros(nz - 1),
        Hpi=jnp.full(nz - 1, col["H_atm"]),  # mean scale height on interfaces
        Ti=jnp.full(nz - 1, T_ISO),
        Tco=jnp.full(nz, T_ISO),
        g=jnp.full(nz, G_CGS),
        ms=jnp.asarray([MU_ATM, M_TRACE]),
        alpha=jnp.zeros(ni),  # no thermal diffusion
        M=jnp.asarray(col["n_tot"]),
        vm=jnp.asarray(vm),
        vs=jnp.asarray(zeros_i),
        top_flux=jnp.zeros(ni),
        bot_flux=jnp.zeros(ni),
        bot_vdep=jnp.zeros(ni),
        gas_indx_mask=jnp.asarray([True, True]),
        # No diffusion-limited escape in this benchmark: the top boundary is
        # zero-flux (top_flux is zero and use_topflux is False), so no species
        # takes the `diff_esc` branch in `_build_diff_coeffs_jax`.
        diff_esc_mask=jnp.zeros(ni, dtype=bool),
        use_vm_mol=bool(use_vm_mol),
        use_settling=False,
        use_topflux=False,
        use_botflux=False,
    )


def tracer_tridiag(atm, col):
    """Extract the tracer's (A, B, C) tridiagonal operator rows from the JAX
    diffusion kernels: diff[j] = A[j] y[j] + B[j] y[j+1] + C[j] y[j-1].

    Background number density dominates ysum, so the tracer rows are effectively
    linear. Also returns the operator applied to the background-scaled tracer,
    used to self-check the coefficient extraction.
    """
    nz = col["nz"]
    n_tot = col["n_tot"]
    # State used only to define ysum (~ background); tracer value is immaterial.
    y = np.zeros((nz, 2), dtype=np.float64)
    y[:, 0] = n_tot  # background
    y[:, 1] = F0 * n_tot  # tracer (minor)
    y_j = jnp.asarray(y)

    grav = compute_diff_grav(atm)
    A_e, B_e, C_e, A_m, B_m, C_m, _ = _build_diff_coeffs_jax(y_j, atm, grav)
    A_t = np.asarray(A_e) + np.asarray(A_m)[:, 1]
    B_t = np.asarray(B_e) + np.asarray(B_m)[:, 1]
    C_t = np.asarray(C_e) + np.asarray(C_m)[:, 1]

    # Self-check: reconstruct the operator from (A,B,C) and compare to the kernel.
    diff_kernel = np.asarray(
        _apply_diffusion_jax(y_j, A_e, B_e, C_e, A_m, B_m, C_m, atm)
    )[:, 1]
    diff_recon = np.empty(nz)
    diff_recon[0] = A_t[0] * y[0, 1] + B_t[0] * y[1, 1]
    diff_recon[-1] = A_t[-1] * y[-1, 1] + C_t[-1] * y[-2, 1]
    diff_recon[1:-1] = (
        A_t[1:-1] * y[1:-1, 1] + B_t[1:-1] * y[2:, 1] + C_t[1:-1] * y[:-2, 1]
    )
    recon_err = np.max(np.abs(diff_recon - diff_kernel)) / (
        np.max(np.abs(diff_kernel)) + 1e-300
    )
    assert recon_err < 1e-10, f"coefficient extraction mismatch: {recon_err:.2e}"
    return A_t, B_t, C_t


def reference_tridiag(col, Dzz_trace_iface, Kzz_iface, *, use_vm_mol):
    """Tracer (A, B, C) rows from the upstream VULCAN operator ("VULCAN 2.0").

    Uses tests/diffusion_numpy_ref.py, the validated transcription of op.diffdf
    (central) / op.diffdf_vm (upwind), on the same synthetic column as the JAX
    path -- so the two operators should agree to machine precision.
    """
    nz, ni = col["nz"], 2
    n_tot = col["n_tot"]
    zeros_i = np.zeros((nz - 1, ni), dtype=np.float64)
    Dzz = zeros_i.copy()
    Dzz[:, 1] = Dzz_trace_iface
    inv_H_trace = M_TRACE * G_CGS / (R_GAS * T_ISO)
    inv_H_atm = MU_ATM * G_CGS / (R_GAS * T_ISO)
    vm = zeros_i.copy()
    if use_vm_mol:
        vm[:, 1] = -Dzz_trace_iface * (inv_H_trace - inv_H_atm)
    atm = SimpleNamespace(
        dzi=col["dzi"],
        Kzz=Kzz_iface,
        Dzz=Dzz,
        alpha=np.zeros(ni),
        ms=np.array([MU_ATM, M_TRACE]),
        Tco=np.full(nz, T_ISO),
        Hp=np.full(nz, col["H_atm"]),
        Hpi=np.full(nz - 1, col["H_atm"]),
        Ti=np.full(nz - 1, T_ISO),
        g=np.full(nz, G_CGS),
        vz=np.zeros(nz - 1),
        vm=vm,
        vs=zeros_i.copy(),
        top_flux=np.zeros(ni),
        bot_flux=np.zeros(ni),
        bot_vdep=np.zeros(ni),
        gas_indx=np.array([0, 1]),
    )
    cfg = SimpleNamespace(
        use_moldiff=True,
        use_topflux=False,
        use_botflux=False,
        use_vm_mol=bool(use_vm_mol),
        use_settling=False,
        non_gas_sp=False,
    )
    y = np.zeros((nz, ni), dtype=np.float64)
    y[:, 0] = n_tot
    y[:, 1] = F0 * n_tot
    coeffs = diffref.build_diffusion_coeffs(y, atm, cfg, mode="auto")
    A_t = coeffs.A_eddy + coeffs.A_mol[:, 1]
    B_t = coeffs.B_eddy + coeffs.B_mol[:, 1]
    C_t = coeffs.C_eddy + coeffs.C_mol[:, 1]
    return A_t, B_t, C_t


def solve_steady_state(A_t, B_t, C_t, y0_base):
    """Solve the discrete steady state diff[j]=0 (j>=1, zero-flux top built in)
    with the base cell pinned to y0_base (Dirichlet). Tridiagonal, nz x nz."""
    nz = A_t.shape[0]
    M = np.zeros((nz, nz), dtype=np.float64)
    rhs = np.zeros(nz, dtype=np.float64)
    M[0, 0] = 1.0  # pin base mole fraction
    rhs[0] = y0_base
    for j in range(1, nz):
        M[j, j] = A_t[j]
        M[j, j - 1] = C_t[j]
        if j < nz - 1:
            M[j, j + 1] = B_t[j]
    return np.linalg.solve(M, rhs)


def run_case(col, Dzz_trace_iface, Kzz_iface, f_analytic):
    """Solve central and upwind steady states for BOTH VULCAN-JAX and the
    upstream VULCAN 2.0 operator; return profiles + error/agreement metrics."""
    n_tot = col["n_tot"]
    y0 = F0 * n_tot[0]  # base tracer number density

    out = {"analytic": f_analytic}
    for name, use_vm in (("central", False), ("upwind", True)):
        atm = make_atm_static(col, Dzz_trace_iface, Kzz_iface, use_vm_mol=use_vm)
        f_jax = solve_steady_state(*tracer_tridiag(atm, col), y0) / n_tot
        f_ref = (
            solve_steady_state(
                *reference_tridiag(col, Dzz_trace_iface, Kzz_iface, use_vm_mol=use_vm),
                y0,
            )
            / n_tot
        )
        out[name] = f_jax
        out[f"{name}_ref"] = f_ref
        # Fractional (relative) error |f/f_analytic - 1| vs analytic.
        for tag, f in ((name, f_jax), (f"{name}_ref", f_ref)):
            fe = np.abs(f / f_analytic - 1.0)
            out[f"{tag}_max_fe"] = float(np.max(fe))
            out[f"{tag}_rms_fe"] = float(np.sqrt(np.mean(fe**2)))
        # Port fidelity: VULCAN-JAX vs VULCAN 2.0 (should be machine precision).
        out[f"{name}_port_fe"] = float(np.max(np.abs(f_jax / f_ref - 1.0)))
    return out


def stability_sweep(nz_values, n_scale_heights):
    """Pure-molecular-diffusion (K=0) stability scan across grid resolutions.

    The undamped drift gives cell Peclet number Pe = |vm| dz / D = kappa*dz.
    Central differencing develops a spurious sign-alternating mode for Pe > 2,
    so its steady state goes negative (unphysical); upwind stays positive at any
    Pe. Returns (peclet, central_min, upwind_min), the minima normalised by the
    base mole fraction.
    """
    inv_H_trace = M_TRACE * G_CGS / (R_GAS * T_ISO)
    inv_H_atm = MU_ATM * G_CGS / (R_GAS * T_ISO)
    kappa = inv_H_trace - inv_H_atm
    peclet, cmin, umin = [], [], []
    for nz in nz_values:
        col = build_column(nz, n_scale_heights)
        peclet.append(float(kappa * np.mean(np.diff(col["zco"]))))
        f_a = F0 * np.exp(-kappa * col["zco"])  # analytic (K=0 -> D/(D+K)=1)
        res = run_case(col, np.full(nz - 1, D_CONST), np.zeros(nz - 1), f_a)
        cmin.append(float(np.min(res["central"]) / F0))
        umin.append(float(np.min(res["upwind"]) / F0))
    return np.array(peclet), np.array(cmin), np.array(umin)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for the figure and CSV (default: alongside this script).",
    )
    parser.add_argument(
        "--basename",
        default="zhang2013_moldiff_benchmark",
        help="stem for the written files.",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="comma-separated figure formats, e.g. 'png,pdf'.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="also write the plotted data as CSV.",
    )
    args = parser.parse_args(argv)

    inv_H_trace = M_TRACE * G_CGS / (R_GAS * T_ISO)
    inv_H_atm = MU_ATM * G_CGS / (R_GAS * T_ISO)
    kappa = inv_H_trace - inv_H_atm  # 1/H_i - 1/H_atm, 1/cm

    # ---- Panel A: accuracy at a well-resolved grid (constant D, K) ----------
    col = build_column(NZ, N_SCALE_HEIGHTS)
    H_atm = col["H_atm"]
    frac = 1.0 / (1.0 + R_KD)  # D/(D+K)
    f_analytic = F0 * np.exp(-frac * kappa * col["zco"])
    res = run_case(
        col, np.full(NZ - 1, D_CONST), np.full(NZ - 1, R_KD * D_CONST), f_analytic
    )

    # ---- Panel B: stability across grid resolution (pure molecular, K=0) ----
    nz_sweep = [8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60, 90, 120]
    peclet, cmin, umin = stability_sweep(nz_sweep, N_SCALE_HEIGHTS)

    # ---- Report -------------------------------------------------------------
    print("Zhang, Shia & Yung (2013) molecular-diffusion benchmark")
    print(f"  H_atm = {H_atm / 1e5:.1f} km, tracer m={M_TRACE} in mu={MU_ATM}")
    print()
    print(f"ACCURACY (nz={NZ}, D/(D+K)={frac:.1f}):  max fractional error vs analytic")
    print(f"  central (= hybrid converged): {_fmt_err(res['central_max_fe'])}")
    print(f"  upwind only (pre-central):    {_fmt_err(res['upwind_max_fe'])}")
    print(
        f"  VULCAN-JAX vs VULCAN 2.0: central {res['central_port_fe']:.0e},"
        f" upwind {res['upwind_port_fe']:.0e}"
    )
    print()
    print("STABILITY (pure molecular, K=0):  minimum mole fraction / base")
    for p, c, u in zip(peclet, cmin, umin):
        flag = "  <-- central UNSTABLE (negative)" if c < 0 else ""
        print(f"  Pe={p:5.2f}:  central {c:+.2e}   upwind {u:+.2e}{flag}")
    print()
    print("central alone: accurate but goes negative (spurious oscillation) at cell")
    print("Peclet > 2. upwind alone: unconditionally stable but dissipative. The")
    print("production HYBRID converges under upwind (stable), then finishes in central")
    print("(accurate) -> it reproduces the analytic curve without the instability.")

    # ---- Figure -------------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    blue, red = "#1f77b4", "#d62728"

    # Panel A: accuracy (steady-state profile vs pressure, well-resolved grid).
    p_mbar = col["pco"] / 1e3  # dyn cm^-2 -> mbar
    axA.plot(res["analytic"], p_mbar, "k-", lw=2.4, label="Zhang+2013 analytic")
    axA.plot(
        res["central"], p_mbar, color=blue, lw=1.8, label="central (= hybrid converged)"
    )
    axA.plot(
        res["upwind"],
        p_mbar,
        color=red,
        lw=1.8,
        ls="--",
        label="upwind only (pre-central)",
    )
    # VULCAN 2.0 (upstream op.diffdf / op.diffdf_vm) as open markers on top.
    axA.plot(
        res["central_ref"],
        p_mbar,
        color=blue,
        ls="none",
        marker="o",
        ms=5,
        mfc="none",
        markevery=6,
        label="VULCAN 2.0",
    )
    axA.plot(
        res["upwind_ref"],
        p_mbar,
        color=red,
        ls="none",
        marker="s",
        ms=5,
        mfc="none",
        markevery=6,
    )
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_ylim(p_mbar.max(), p_mbar.min())  # pressure decreasing upward
    axA.set_xlabel("tracer mole fraction")
    axA.set_ylabel("pressure (mbar)")
    axA.set_title(f"A. Accuracy (well resolved, nz={NZ})", fontsize=10)
    axA.grid(True, which="major", alpha=0.25)
    axA.legend(loc="upper right", fontsize=8)
    axA.text(
        0.04,
        0.04,
        f"max error vs analytic\ncentral / hybrid: {_fmt_err(res['central_max_fe'])}\n"
        f"upwind only: {_fmt_err(res['upwind_max_fe'])}",
        transform=axA.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    # Panel B: stability (worst mole fraction vs cell Peclet number).
    axB.axhspan(1.15 * min(float(cmin.min()), -0.05), 0.0, color=red, alpha=0.07)
    axB.axhline(0.0, color="0.4", lw=0.9)
    axB.axvline(2.0, color="0.6", ls=":", lw=1.0)
    axB.plot(
        peclet, cmin, color=blue, lw=1.8, marker="o", ms=4, label="central (original)"
    )
    axB.plot(
        peclet, umin, color=red, lw=1.8, marker="s", ms=4, label="upwind (stabilizer)"
    )
    axB.set_xlabel("cell Peclet number   |vm| dz / D")
    axB.set_ylabel("minimum mole fraction / base value")
    axB.set_title("B. Stability (pure molecular diffusion, K=0)", fontsize=10)
    axB.grid(True, which="major", alpha=0.25)
    axB.legend(loc="lower left", fontsize=8)
    axB.text(
        2.1,
        0.03,
        "Pe > 2:\ncentral goes negative\n(unphysical oscillation)",
        transform=axB.get_xaxis_transform(),
        fontsize=8,
        color="#a00000",
        va="bottom",
    )

    fig.suptitle(
        "Molecular diffusion: the hybrid scheme is accurate (central) and "
        "stable (upwind)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in [e.strip() for e in args.formats.split(",") if e.strip()]:
        out = args.outdir / f"{args.basename}.{ext}"
        fig.savefig(out, dpi=200)
        print(f"\nwrote {out}")

    if args.csv:
        out_csv = args.outdir / f"{args.basename}_data.csv"
        with out_csv.open("w") as fh:
            fh.write(
                "# Panel A: steady-state mole fraction vs pressure "
                f"(nz={NZ}, D/(D+K)={frac:.3f})\n"
            )
            fh.write(
                "panel,p_mbar,analytic,jax_central_hybrid,jax_upwind,"
                "vulcan2_central,vulcan2_upwind\n"
            )
            for k in range(len(p_mbar)):
                fh.write(
                    f"A,{p_mbar[k]:.10e},{res['analytic'][k]:.10e},"
                    f"{res['central'][k]:.10e},{res['upwind'][k]:.10e},"
                    f"{res['central_ref'][k]:.10e},{res['upwind_ref'][k]:.10e}\n"
                )
            fh.write("# Panel B: stability sweep, pure molecular (K=0)\n")
            fh.write("panel,cell_peclet,central_min_frac,upwind_min_frac\n")
            for pe, c, u in zip(peclet, cmin, umin):
                fh.write(f"B,{pe:.10e},{c:.10e},{u:.10e}\n")
        print(f"wrote {out_csv}")

    ok = (
        res["central_max_fe"] < 0.05
        and res["upwind_max_fe"] > 3 * res["central_max_fe"]
        and res["central_port_fe"] < 1e-6
        and bool(np.any(cmin < 0))  # central unstable at high cell Peclet
        and bool(np.all(umin >= 0))  # upwind stable everywhere
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
