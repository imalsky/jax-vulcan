"""Validate the PRODUCTION JAX diffusion kernel against the NumPy reference
(`diffusion_numpy_ref`), which is itself master-validated elsewhere. JAX-only;
no VULCAN-master oracle required.

Pins, for the default 'gravity' mode and the upwind 'vm' mode:
  1. coefficient arrays (A/B/C eddy + mol) at ~machine precision,
  2. the diffusion operator output on significant cells.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from _helpers import relerr

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

COEF_RTOL = 1e-11  # C_mol FP-ordering noise reaches ~2e-12; eddy terms ~5e-16
OP_RTOL = 1e-4  # the operator is a small residue of large cancellations (floor ~1e-5)
OP_FLOOR = 1e-12  # absolute and peak-relative floor for significant operator cells


@pytest.mark.parametrize("mode", ["gravity", "vm"])
def test_production_kernel_matches_reference(mode):
    """The 'vm' case pins the hot-path upwind (`use_vm_mol`) discretization
    with a nonzero mixed-sign interface drift vm (shape (nz-1, ni))."""
    import jax.numpy as jnp

    import diffusion_numpy_ref as diff_ref
    import vulcan_jax.jax_step as jax_step
    from vulcan_jax.config import default_config
    from vulcan_jax.state import RunState, legacy_view

    vulcan_cfg = default_config()
    # The NumPy reference's gravity mode is the CENTRAL scheme, and the vm case
    # needs the all-zero-vm state that use_vm_mol=False builds.
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape

    if mode == "vm":
        # Inject a nonzero, mixed-sign interface drift so both upwind branches
        # ((vm>0)/(vm<0)) fire in both the production kernel and the reference.
        rng = np.random.default_rng(1)
        data_atm.vm = (rng.standard_normal((nz - 1, ni)) * 50.0).astype(np.float64)
        vulcan_cfg.use_vm_mol = True

    # --- Production kernel ---
    atm_static = jax_step.make_atm_static(data_atm, ni, nz, cfg=vulcan_cfg)
    grav = jax_step.compute_diff_grav(atm_static)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = jax_step._build_diff_coeffs_jax(
        jnp.asarray(y), atm_static, grav
    )
    diff_prod = np.asarray(
        jax_step._apply_diffusion_jax(
            jnp.asarray(y), A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm_static
        )
    )

    # --- NumPy reference ---
    coeffs = diff_ref.build_diffusion_coeffs(y, data_atm, vulcan_cfg, mode=mode)
    diff_numpy = diff_ref.apply_diffusion(y, coeffs)

    # 1. Coefficients are identical formulas -> machine precision (FP-ordering only).
    for label, p, r in (
        ("A_eddy", A_eddy, coeffs.A_eddy),
        ("B_eddy", B_eddy, coeffs.B_eddy),
        ("C_eddy", C_eddy, coeffs.C_eddy),
        ("A_mol", A_mol, coeffs.A_mol),
        ("B_mol", B_mol, coeffs.B_mol),
        ("C_mol", C_mol, coeffs.C_mol),
    ):
        r = np.asarray(r)
        err = relerr(p, r, floor=1e-30 * max(np.abs(r).max(), 1e-300))
        assert err < COEF_RTOL, f"{mode}-mode {label} relerr {err:.3e}"

    # 2. Operator output on significant cells.
    abs_tol = max(OP_FLOOR, OP_FLOOR * np.abs(diff_numpy).max())
    err = relerr(diff_prod, diff_numpy, floor=abs_tol)
    assert err < OP_RTOL, f"{mode}-mode diffusion operator relerr {err:.3e}"
