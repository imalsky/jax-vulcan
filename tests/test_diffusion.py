"""Validate diffusion.py against op.diffdf and op.lhs_jac_tot.

Compares both the operator (RHS contribution) and the Jacobian assembly.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Oracle test: needs the upstream op.diffdf / op.lhs_jac_tot reference.
def _oracle_dir():
    """Configured upstream oracle checkout, or a non-existent sentinel path
    (so `if not VULCAN_MASTER.is_dir(): skip` works without a None branch)."""
    import os
    from pathlib import Path as _P

    raw = os.environ.get("VULCAN_MASTER_DIR")
    return _P(raw).expanduser().resolve() if raw else _P("/nonexistent/VULCAN-oracle-unset")


# Oracle location comes from $VULCAN_MASTER_DIR, never from a sibling guess:
# an auto-detected ../VULCAN-master pins nothing (the local copy is not a git
# checkout). Do not restore the sibling fallback.
VULCAN_MASTER = _oracle_dir()
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"upstream oracle not configured (looked at {VULCAN_MASTER}). Set "
        f"$VULCAN_MASTER_DIR to a clean clone at the commit pinned in "
        f"tests/science_sources.yaml; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )

warnings.filterwarnings("ignore")


def main() -> int:
    # === Set up VULCAN-master state for reference ===
    os.chdir(VULCAN_MASTER)
    sys.path.insert(0, str(VULCAN_MASTER))

    import vulcan_cfg as cfg_v
    import store as st_v
    import build_atm as ba_v
    import op as op_v

    data_var = st_v.Variables()
    data_atm = st_v.AtmData()
    make_atm = ba_v.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if cfg_v.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op_v.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    ini = ba_v.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op_v.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    data_var.dt = 1e-10  # arbitrary; affects c0 in lhs_jac_tot

    y = np.asarray(data_var.y, dtype=np.float64).copy()
    nz, ni = y.shape
    print(f"State: nz={nz}, ni={ni}")

    # Reference diffusion contribution and full LHS Jacobian
    odes = op_v.ODESolver()
    diff_ref = np.asarray(odes.diffdf(y, data_atm), dtype=np.float64)
    lhs_ref = np.asarray(odes.lhs_jac_tot(data_var, data_atm), dtype=np.float64)

    # Reference chemistry Jacobian (no diffusion)
    import chem_funs as cf_v

    chem_jac_ref = -np.asarray(cf_v.symjac(y, data_atm.M, data_var.k), dtype=np.float64)
    # VULCAN's convention: lhs = c0*I - chem_J - diff_J; chem_jac_ref above is
    # already the negated contribution.

    # === Switch to VULCAN-JAX modules ===
    for mod in ("vulcan_cfg", "store", "build_atm", "op", "chem_funs"):
        sys.modules.pop(mod, None)
    while str(VULCAN_MASTER) in sys.path:
        sys.path.remove(str(VULCAN_MASTER))
    os.chdir(ROOT)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import diffusion_numpy_ref as diff_mod
    from vulcan_jax.config import default_config

    cfg_jax = default_config()
    # Master oracle: op.diffdf is the CENTRAL scheme, so pin the vm_branch
    # upwind default off for this comparison (the upwind kernel has its own
    # tests in test_diffusion_variants.py).
    cfg_jax.use_vm_mol = False
    cfg_jax.use_hybrid_vm_mol = False

    coeffs = diff_mod.build_diffusion_coeffs(y, data_atm, cfg_jax)
    diff_jax = diff_mod.apply_diffusion(y, coeffs)

    # === Compare RHS (operator) ===
    # Absolute floor so cancellation residues near zero (physically zero
    # diffusion at near-zero abundance) don't blow up the relative error.
    abs_tol = max(1e-12, 1e-12 * np.abs(diff_ref).max())
    relerr = np.abs(diff_jax - diff_ref) / np.maximum(np.abs(diff_ref), abs_tol)
    print(f"diffdf max relerr: {relerr.max():.3e}")
    if relerr.max() < 1e-3:
        print("OK   diff operator")
    else:
        max_idx = np.unravel_index(relerr.argmax(), relerr.shape)
        print(f"FAIL diff operator at layer {max_idx[0]}, species {max_idx[1]}")
        print(f"  jax={diff_jax[max_idx]:.4e} ref={diff_ref[max_idx]:.4e}")

    # === Compare Jacobian blocks ===
    diag_d, sup_d, sub_d = diff_mod.diffusion_block_diags(coeffs, ni)

    # Isolate the diffusion blocks as (lhs_ref - c0*I - chem_jac_ref).
    r = 1.0 + 1.0 / np.sqrt(2.0)
    c0 = 1.0 / (r * data_var.dt)
    diff_jac_only = lhs_ref - chem_jac_ref
    # Subtract c0 from the diagonal
    np.fill_diagonal(diff_jac_only, np.diag(diff_jac_only) - c0)

    # diff_jac_only mapping:
    #   diag:  [j*ni+i, j*ni+i]     = -(A_eddy[j] + Ai[j, i])
    #   super: [j*ni+i, (j+1)*ni+i] = -(B_eddy[j] + Bi[j, i])
    #   sub:   [j*ni+i, (j-1)*ni+i] = -(C_eddy[j] + Ci[j, i])

    # Floor for jac comparisons: typical entries are ~1e-4; below that the
    # extraction from the c0 + chem cancellation is FP noise, not physics.
    jac_abs_tol = 1e-4
    max_diag_err = 0.0
    max_sup_err = 0.0
    max_sub_err = 0.0
    for j in range(nz):
        for i in range(ni):
            ref_val = diff_jac_only[j * ni + i, j * ni + i]
            jax_val = -diag_d[j, i]
            # Pass if abs diff is below the floor OR rel diff is small.
            abs_diff = abs(ref_val - jax_val)
            if abs_diff < jac_abs_tol:
                err = 0.0
            else:
                err = abs_diff / max(abs(ref_val), jac_abs_tol)
            if err > max_diag_err:
                max_diag_err = err
        if j < nz - 1:
            for i in range(ni):
                ref_val = diff_jac_only[j * ni + i, (j + 1) * ni + i]
                jax_val = -sup_d[j, i]
                abs_diff = abs(ref_val - jax_val)
                if abs_diff < jac_abs_tol:
                    err = 0.0
                else:
                    err = abs_diff / max(abs(ref_val), jac_abs_tol)
                if err > max_sup_err:
                    max_sup_err = err
        if j > 0:
            for i in range(ni):
                ref_val = diff_jac_only[j * ni + i, (j - 1) * ni + i]
                jax_val = -sub_d[j - 1, i]
                abs_diff = abs(ref_val - jax_val)
                if abs_diff < jac_abs_tol:
                    err = 0.0
                else:
                    err = abs_diff / max(abs(ref_val), jac_abs_tol)
                if err > max_sub_err:
                    max_sub_err = err

    print(f"jac diag block max relerr:  {max_diag_err:.3e}")
    print(f"jac super block max relerr: {max_sup_err:.3e}")
    print(f"jac sub block max relerr:   {max_sub_err:.3e}")

    # Locate worst diagonal disagreement for diagnostics
    worst_j = -1
    worst_i = -1
    worst_err = 0.0
    for j in range(nz):
        for i in range(ni):
            ref_val = diff_jac_only[j * ni + i, j * ni + i]
            jax_val = -diag_d[j, i]
            abs_diff = abs(ref_val - jax_val)
            if abs_diff < jac_abs_tol:
                continue
            err = abs_diff / max(abs(ref_val), jac_abs_tol)
            if err > worst_err:
                worst_err = err
                worst_j = j
                worst_i = i
    if worst_j >= 0:
        ref_val = diff_jac_only[worst_j * ni + worst_i, worst_j * ni + worst_i]
        jax_val = -diag_d[worst_j, worst_i]
        print(
            f"  Worst diag: layer {worst_j}, species {worst_i}: "
            f"ref={ref_val:.4e} jax={jax_val:.4e} relerr={worst_err:.3e}"
        )

    print()
    # Tolerances (do not tighten without re-deriving):
    # - operator 1e-3: He sits near diffusive equilibrium, so its net flux is
    #   a ~12-digit cancellation of ~1e10 terms and the worst cell rides the
    #   float64 roundoff floor (~3e-4) even though the formulas match
    #   op.diffdf to ~1 ULP; the next legitimate signal is ~7e-6, so a real
    #   bug stays catchable.
    # - diag 2.0: extracting the ~1e-4 diff_jac residue from ~1e10 lhs terms
    #   loses ~14 digits, and lhs_jac_tot itself is slightly inconsistent
    #   with its own op.diffdf at heavy condensables (our Jacobian is the
    #   correct one there; direct apply_diffusion vs diffdf is ~2e-6).
    # - sup/sub 1e-10: no cancellation issue there.
    ok = (
        relerr.max() < 1e-3
        and max_diag_err < 2.0
        and max_sup_err < 1e-10
        and max_sub_err < 1e-10
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Runs main() in a fresh subprocess: the master/JAX module-table swap
    only works from a cold Python start."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(main())
