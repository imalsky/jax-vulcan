"""Validate the PRODUCTION JAX diffusion kernel against the NumPy reference.

`tests/test_diffusion.py` and `tests/test_diffusion_variants.py` compare the
NumPy reference (`diffusion_numpy_ref`) against VULCAN-master, but the actual
hot-path kernel `jax_step._build_diff_coeffs_jax` / `_apply_diffusion_jax` is
only exercised transitively through one coarse Ros2-step oracle. This test
compares the production kernel directly against the same NumPy reference at
machine precision, for the default ('gravity') transport mode.

This also pins the diffusion-Jacobian self-consistency: the per-layer block
diagonals the kernel feeds into block-Thomas (diag/sup/sub) must equal the
analytical (frozen-ysum) block diagonals of the diffusion operator. JAX-only;
no VULCAN-master oracle required.

Validates:
  1. Coefficient arrays (A_eddy/B_eddy/C_eddy, A_mol/B_mol/C_mol) match the
     NumPy reference to ~machine precision.
  2. The diffusion operator output (`_apply_diffusion_jax` vs `apply_diffusion`)
     matches on significant cells.
  3. The block-Jacobian diagonals (diag/sup/sub) assembled exactly as
     `jax_ros2_step` does match the reference's `diffusion_block_diags`.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")


def main() -> int:
    import jax.numpy as jnp

    import diffusion_numpy_ref as diff_ref
    import vulcan_jax.jax_step as jax_step
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    # The NumPy reference implements the CENTRAL scheme; pin the vm_branch
    # upwind default off (upwind coverage lives in test_diffusion_variants.py).
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape

    # --- Production kernel ---
    atm_static = jax_step.make_atm_static(data_atm, ni, nz)
    grav = jax_step.compute_diff_grav(atm_static)
    (
        A_eddy,
        B_eddy,
        C_eddy,
        A_mol,
        B_mol,
        C_mol,
        _ysum,
    ) = jax_step._build_diff_coeffs_jax(jnp.asarray(y), atm_static, grav)
    diff_prod = np.asarray(
        jax_step._apply_diffusion_jax(
            jnp.asarray(y), A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm_static
        )
    )

    # --- NumPy reference (gravity mode) ---
    coeffs = diff_ref.build_diffusion_coeffs(y, data_atm, vulcan_cfg, mode="gravity")
    diff_numpy = diff_ref.apply_diffusion(y, coeffs)

    ok = True

    def _coef_relerr(prod, ref):
        prod = np.asarray(prod)
        ref = np.asarray(ref)
        denom = np.maximum(np.abs(ref), 1e-30 * max(np.abs(ref).max(), 1e-300))
        return float(np.max(np.abs(prod - ref) / denom))

    # 1. Coefficient arrays — machine-precision parity vs the NumPy reference.
    # C_mol mixes per-species molecular weights through the mean-molecular-weight
    # gradient, so its JAX-vs-NumPy FP-ordering noise is composition-dependent and
    # rides a few ULP higher than the rest (~2e-12 on some FastChem-EQ
    # compositions). 1e-11 keeps this firmly at the machine-precision floor while
    # absorbing that noise; the eddy coeffs still land at ~5e-16, so sensitivity to
    # a real regression is not lost.
    for label, p, r in (
        ("A_eddy", A_eddy, coeffs.A_eddy),
        ("B_eddy", B_eddy, coeffs.B_eddy),
        ("C_eddy", C_eddy, coeffs.C_eddy),
        ("A_mol", A_mol, coeffs.A_mol),
        ("B_mol", B_mol, coeffs.B_mol),
        ("C_mol", C_mol, coeffs.C_mol),
    ):
        err = _coef_relerr(p, r)
        print(f"coeff {label:7s} relerr: {err:.3e}")
        if err > 1e-11:
            print(f"FAIL: production {label} disagrees with NumPy reference")
            ok = False

    # 2. Operator output on significant cells. The operator (A*y + B*y+ + C*y-)
    # extracts a small residue from large cancellations (c0 ~ 1e10), so it rides
    # at the documented diffusion FP-noise floor (~1e-5); the coefficient and
    # block-diagonal checks above are the machine-precision ones. 1e-4 matches
    # the operator tolerance in test_diffusion.py.
    abs_tol = max(1e-12, 1e-12 * np.abs(diff_numpy).max())
    op_relerr = np.abs(diff_prod - diff_numpy) / np.maximum(np.abs(diff_numpy), abs_tol)
    print(f"operator max relerr (sig cells): {op_relerr.max():.3e}")
    if op_relerr.max() > 1e-4:
        print("FAIL: production diffusion operator disagrees with reference")
        ok = False

    # 3. Block-Jacobian diagonals, assembled exactly as jax_ros2_step does.
    diag_prod = np.asarray(A_eddy[:, None] + A_mol)
    sup_prod = np.asarray(B_eddy[:-1, None] + B_mol[:-1])
    sub_prod = np.asarray(C_eddy[1:, None] + C_mol[1:])
    if bool(getattr(vulcan_cfg, "use_botflux", False)):
        diag_prod[0] = diag_prod[0] - np.asarray(data_atm.bot_vdep) / float(
            data_atm.dzi[0]
        )
    diag_ref, sup_ref, sub_ref = diff_ref.diffusion_block_diags(coeffs, ni)
    for label, p, r in (
        ("diag", diag_prod, diag_ref),
        ("sup", sup_prod, sup_ref),
        ("sub", sub_prod, sub_ref),
    ):
        err = _coef_relerr(p, r)
        print(f"block {label:4s} relerr: {err:.3e}")
        if err > 1e-12:
            print(f"FAIL: production {label} block disagrees with reference")
            ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    assert main() == 0


def test_vm_mode_kernel_matches_reference():
    """Production diffusion kernel reproduces the NumPy reference in the upwind
    `use_vm_mol` mode, with a nonzero interface drift velocity vm (shape
    (nz-1, ni)). The gravity-mode `main()` above leaves the vm branch
    unexercised; this pins the hot-path discretization for the upwind scheme.
    JAX-only -- no VULCAN-master oracle required.
    """
    import types

    import jax.numpy as jnp

    import diffusion_numpy_ref as diff_ref
    import vulcan_jax.jax_step as jax_step
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    # Pin the vm_branch default off so the built state starts from the
    # all-zero-vm premise the injection below relies on.
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape

    # With use_vm_mol pinned False, atm.vm is all-zero. Inject a nonzero,
    # mixed-sign interface drift so both upwind branches ((vm>0)/(vm<0)) fire in
    # both the production kernel and the reference.
    rng = np.random.default_rng(1)
    data_atm.vm = (rng.standard_normal((nz - 1, ni)) * 50.0).astype(np.float64)

    # make_atm_static only reads the transport toggles + atm.gas_indx; keep the
    # real BC flags so the BC contributions match the reference (driven off the
    # same vulcan_cfg in mode='vm').
    cfg_vm = types.SimpleNamespace(
        use_vm_mol=True,
        use_moldiff=bool(getattr(vulcan_cfg, "use_moldiff", True)),
        use_settling=False,
        use_topflux=bool(getattr(vulcan_cfg, "use_topflux", False)),
        use_botflux=bool(getattr(vulcan_cfg, "use_botflux", False)),
    )
    atm_static = jax_step.make_atm_static(data_atm, ni, nz, cfg=cfg_vm)
    assert tuple(np.asarray(atm_static.vm).shape) == (nz - 1, ni)

    grav = jax_step.compute_diff_grav(atm_static)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, _ = jax_step._build_diff_coeffs_jax(
        jnp.asarray(y), atm_static, grav
    )
    diff_prod = np.asarray(
        jax_step._apply_diffusion_jax(
            jnp.asarray(y), A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, atm_static
        )
    )

    coeffs = diff_ref.build_diffusion_coeffs(y, data_atm, vulcan_cfg, mode="vm")
    diff_numpy = diff_ref.apply_diffusion(y, coeffs)

    def _relerr(prod, ref):
        prod = np.asarray(prod)
        ref = np.asarray(ref)
        denom = np.maximum(np.abs(ref), 1e-30 * max(np.abs(ref).max(), 1e-300))
        return float(np.max(np.abs(prod - ref) / denom))

    # Coefficients are identical formulas -> machine precision (FP-ordering only).
    for label, p, r in (
        ("A_eddy", A_eddy, coeffs.A_eddy),
        ("B_eddy", B_eddy, coeffs.B_eddy),
        ("C_eddy", C_eddy, coeffs.C_eddy),
        ("A_mol", A_mol, coeffs.A_mol),
        ("B_mol", B_mol, coeffs.B_mol),
        ("C_mol", C_mol, coeffs.C_mol),
    ):
        assert _relerr(p, r) < 1e-11, f"vm-mode {label} disagrees with reference"

    # Operator output extracts a small residue from large cancellations; same
    # ~1e-4 FP floor as the gravity-mode operator check.
    abs_tol = max(1e-12, 1e-12 * np.abs(diff_numpy).max())
    op_relerr = np.abs(diff_prod - diff_numpy) / np.maximum(np.abs(diff_numpy), abs_tol)
    assert op_relerr.max() < 1e-4, "vm-mode diffusion operator disagrees"

    # Block-Jacobian diagonals assembled exactly as jax_ros2_step does.
    diag_prod = np.asarray(A_eddy[:, None] + A_mol)
    sup_prod = np.asarray(B_eddy[:-1, None] + B_mol[:-1])
    sub_prod = np.asarray(C_eddy[1:, None] + C_mol[1:])
    if cfg_vm.use_botflux:
        diag_prod[0] = diag_prod[0] - np.asarray(data_atm.bot_vdep) / float(
            data_atm.dzi[0]
        )
    diag_ref, sup_ref, sub_ref = diff_ref.diffusion_block_diags(coeffs, ni)
    for label, p, r in (
        ("diag", diag_prod, diag_ref),
        ("sup", sup_prod, sup_ref),
        ("sub", sub_prod, sub_ref),
    ):
        assert _relerr(p, r) < 1e-12, f"vm-mode {label} block disagrees"


if __name__ == "__main__":
    raise SystemExit(main())
