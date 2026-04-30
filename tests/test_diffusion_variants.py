"""Validate JAX diffusion variants (vm / settling / settling_vm) against VULCAN.

VULCAN's op.diffdf_vm and op.diffdf_settling provide reference NumPy
implementations. We don't actually run end-to-end with these (use_vm_mol /
use_settling configs are off in HD189), but the diffusion operators themselves
can be validated on synthetic state with vm/vs populated.
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
sys.path.insert(0, str(ROOT))

# Oracle test: requires VULCAN-master sibling for the upstream op.diffdf_vm /
# op.diffdf_settling reference. Skip cleanly when absent.
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )
sys.path.append(str(VULCAN_MASTER))
warnings.filterwarnings("ignore")


def main() -> int:
    import vulcan_cfg
    import op  # VULCAN-master's op (oracle)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import diffusion_numpy_ref as diff_mod

    # Build canonical HD189 state via the typed pre-loop pipeline and
    # derive a legacy `(var, atm, _)` shim for master's `op.ODESolver`,
    # which only reads `var.y` and `atm.*` (both populated by `legacy_view`).
    from state import RunState, legacy_view
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, _ = legacy_view(rs)
    nz, ni = data_var.y.shape

    # Populate atm.vs with synthetic settling velocities for non-zero variant test
    rng = np.random.default_rng(0)
    data_atm.vs = (rng.standard_normal((nz - 1, ni)) * 0.01).astype(np.float64)
    # atm.vm already populated by make_atm.mol_diff

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
    diff_set_ref = np.asarray(odes.diffdf_settling(data_var.y, data_atm), dtype=np.float64)
    cfg_set = _CfgShim(vulcan_cfg, use_vm_mol=False, use_settling=True)
    coeffs_set = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_set)
    diff_set_jax = diff_mod.apply_diffusion(data_var.y, coeffs_set)

    abs_floor2 = 1e-12 * np.abs(diff_set_ref).max()
    pseudo_relerr_set = np.abs(diff_set_jax - diff_set_ref) / np.maximum(np.abs(diff_set_ref), abs_floor2)
    print(f"diffdf_settling:  max relerr (with floor) = {pseudo_relerr_set.max():.3e}")

    # === Test 'settling_vm' mode (= op.diffdf_settling_vm) ===
    diff_setvm_ref = np.asarray(odes.diffdf_settling_vm(data_var.y, data_atm), dtype=np.float64)
    cfg_setvm = _CfgShim(vulcan_cfg, use_vm_mol=True, use_settling=True)
    coeffs_setvm = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_setvm)
    diff_setvm_jax = diff_mod.apply_diffusion(data_var.y, coeffs_setvm)

    abs_floor3 = 1e-12 * np.abs(diff_setvm_ref).max()
    pseudo_relerr_setvm = np.abs(diff_setvm_jax - diff_setvm_ref) / np.maximum(np.abs(diff_setvm_ref), abs_floor3)
    print(f"diffdf_settling_vm: max relerr (with floor) = {pseudo_relerr_setvm.max():.3e}")

    # === All modes: also test that the original 'gravity' mode still works ===
    diff_gravity_ref = np.asarray(odes.diffdf(data_var.y, data_atm), dtype=np.float64)
    cfg_gravity = _CfgShim(vulcan_cfg, use_vm_mol=False, use_settling=False)
    coeffs_gravity = diff_mod.build_diffusion_coeffs(data_var.y, data_atm, cfg_gravity)
    diff_gravity_jax = diff_mod.apply_diffusion(data_var.y, coeffs_gravity)
    abs_floor4 = 1e-12 * np.abs(diff_gravity_ref).max()
    pseudo_relerr_gravity = np.abs(diff_gravity_jax - diff_gravity_ref) / np.maximum(np.abs(diff_gravity_ref), abs_floor4)
    print(f"diffdf (gravity, default): max relerr = {pseudo_relerr_gravity.max():.3e}")

    print()
    ok = (
        pseudo_relerr.max() < 1e-5
        and pseudo_relerr_set.max() < 1e-5
        and pseudo_relerr_setvm.max() < 1e-5
        and pseudo_relerr_gravity.max() < 1e-5
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
