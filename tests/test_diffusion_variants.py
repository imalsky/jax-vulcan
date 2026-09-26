"""Validate JAX diffusion variants (vm / settling / settling_vm) against VULCAN.

The vm, settling and settling_vm operators are checked against op.diffdf_vm /
op.diffdf_settling on a synthetic state with vm/vs populated.
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

from oracle import oracle_dir_or_skip  # noqa: E402

# The parent verifies the pin and passes a temporary copy (oracle.oracle_dir_or_skip).
VULCAN_MASTER = oracle_dir_or_skip("this diffusion-variant comparison")
warnings.filterwarnings("ignore")


def main() -> int:
    os.chdir(VULCAN_MASTER)
    sys.path.append(str(VULCAN_MASTER))
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    import op  # VULCAN-master's op (oracle)

    os.chdir(ROOT)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import diffusion_numpy_ref as diff_mod

    # Build canonical HD189 state via the typed pre-loop pipeline and
    # derive a legacy `(var, atm, _)` shim for master's `op.ODESolver`,
    # which only reads `var.y` and `atm.*` (both populated by `legacy_view`).
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    nz, ni = data_var.y.shape

    # Synthetic mixed-sign interface velocities (shape (nz-1, ni)) so the
    # sign-dependent upwind branches are actually exercised; setup leaves
    # atm.vm all-zero with use_vm_mol=False.
    rng = np.random.default_rng(0)
    data_atm.vs = (rng.standard_normal((nz - 1, ni)) * 0.01).astype(np.float64)
    data_atm.vm = (rng.standard_normal((nz - 1, ni)) * 50.0).astype(np.float64)

    # Synthetic cfg with mode flags
    class _CfgShim:
        def __init__(self, parent, **kw):
            self._parent = parent
            self._overrides = kw

        def __getattr__(self, name):
            if name in self._overrides:
                return self._overrides[name]
            return getattr(self._parent, name)

    # === Test 'vm' mode (= op.diffdf_vm) ===
    odes = op.ODESolver()
    diff_vm_ref = np.asarray(odes.diffdf_vm(data_var.y, data_atm), dtype=np.float64)
    cfg_vm = _CfgShim(vulcan_cfg, use_vm_mol=True, use_settling=False)
    coeffs_vm = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_vm)
    diff_vm_jax = diff_mod.apply_diffusion(data_var.y, coeffs_vm)

    # Use absolute floor for cancellation residues
    abs_floor = 1e-12 * np.abs(diff_vm_ref).max()
    abs_diff = np.abs(diff_vm_jax - diff_vm_ref)
    pseudo_relerr = abs_diff / np.maximum(np.abs(diff_vm_ref), abs_floor)
    print(f"diffdf_vm:        max relerr (with floor) = {pseudo_relerr.max():.3e}")

    # === Test 'settling' mode (= op.diffdf_settling) ===
    diff_set_ref = np.asarray(
        odes.diffdf_settling(data_var.y, data_atm), dtype=np.float64
    )
    cfg_set = _CfgShim(vulcan_cfg, use_vm_mol=False, use_settling=True)
    coeffs_set = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_set)
    diff_set_jax = diff_mod.apply_diffusion(data_var.y, coeffs_set)

    abs_floor2 = 1e-12 * np.abs(diff_set_ref).max()
    pseudo_relerr_set = np.abs(diff_set_jax - diff_set_ref) / np.maximum(
        np.abs(diff_set_ref), abs_floor2
    )
    print(f"diffdf_settling:  max relerr (with floor) = {pseudo_relerr_set.max():.3e}")

    # === Test 'settling_vm' mode (= op.diffdf_settling_vm) ===
    diff_setvm_ref = np.asarray(
        odes.diffdf_settling_vm(data_var.y, data_atm), dtype=np.float64
    )
    cfg_setvm = _CfgShim(vulcan_cfg, use_vm_mol=True, use_settling=True)
    coeffs_setvm = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_setvm)
    diff_setvm_jax = diff_mod.apply_diffusion(data_var.y, coeffs_setvm)

    # Upstream inconsistency: master's
    # op.diffdf_settling_vm omits the vm advective term at j=0 while
    # op.diffdf_vm keeps it (vm_branch carries the same quirk). VULCAN-JAX
    # stays self-consistent at j=0 across all modes, so in the settling+vm
    # combo the two agree everywhere except the j=0 row, which differs by
    # the omitted vm bottom-flux term. Verify rows 1.. at the FP
    # floor and pin the j=0 gap to that exact term.
    abs_floor3 = 1e-12 * np.abs(diff_setvm_ref).max()
    pseudo_relerr_setvm = np.abs(diff_setvm_jax[1:] - diff_setvm_ref[1:]) / np.maximum(
        np.abs(diff_setvm_ref[1:]), abs_floor3
    )
    print(
        "diffdf_settling_vm (rows 1..): max relerr (with floor) = "
        f"{pseudo_relerr_setvm.max():.3e}"
    )
    vm0 = np.asarray(data_atm.vm)[0]
    dzi0 = float(np.asarray(data_atm.dzi)[0])
    y0 = np.asarray(data_var.y)[0]
    y1 = np.asarray(data_var.y)[1]
    # Operator gap at j=0 = (vm upwind that JAX keeps but master drops) applied
    # to (y0, y1):  -[ (vm0>0)*vm0*y0 + (vm0<0)*vm0*y1 ] / dzi0.
    expected_j0_gap = -((vm0 > 0) * vm0 * y0 + (vm0 < 0) * vm0 * y1) / dzi0
    j0_gap = diff_setvm_jax[0] - diff_setvm_ref[0]
    j0_floor = 1e-10 * np.abs(expected_j0_gap).max()
    j0_relerr = np.abs(j0_gap - expected_j0_gap) / np.maximum(
        np.abs(expected_j0_gap), j0_floor
    )
    print(
        f"diffdf_settling_vm j=0 gap == omitted vm term: max relerr = {j0_relerr.max():.3e}"
    )

    # === gravity mode (default) ===
    diff_gravity_ref = np.asarray(odes.diffdf(data_var.y, data_atm), dtype=np.float64)
    cfg_gravity = _CfgShim(vulcan_cfg, use_vm_mol=False, use_settling=False)
    coeffs_gravity = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_gravity)
    diff_gravity_jax = diff_mod.apply_diffusion(data_var.y, coeffs_gravity)
    abs_floor4 = 1e-12 * np.abs(diff_gravity_ref).max()
    pseudo_relerr_gravity = np.abs(diff_gravity_jax - diff_gravity_ref) / np.maximum(
        np.abs(diff_gravity_ref), abs_floor4
    )
    print(f"diffdf (gravity, default): max relerr = {pseudo_relerr_gravity.max():.3e}")

    print()
    ok = (
        pseudo_relerr.max() < 1e-3
        and pseudo_relerr_set.max() < 1e-5
        and pseudo_relerr_setvm.max() < 1e-5
        and j0_relerr.max() < 1e-6
        and pseudo_relerr_gravity.max() < 1e-3
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Run the master comparison in a fresh Python process."""
    from oracle import run_oracle_subprocess

    run_oracle_subprocess(__file__, "vulcan2_ncho",
                          "cfg_examples/vulcan_cfg_HD189.py")


if __name__ == "__main__":
    sys.exit(main())
