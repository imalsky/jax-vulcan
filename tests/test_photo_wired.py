"""Validate that op_jax.Ros2JAX.compute_tau / compute_flux / compute_J wired
through photo.py match VULCAN-master's op.ODESolver versions on a real HD189
state. This is the integration test for Workstream B (photochem wiring).
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

# Oracle test: requires VULCAN-master sibling for the upstream
# op.compute_tau / op.compute_flux / op.compute_J reference. Skip when absent.
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
    import vulcan_cfg                               # noqa
    import store, op                                 # noqa
    from atm_setup import Atm
    from ini_abun import InitialAbun

    # Build atmospheric / chemistry state up to the point compute_tau is called.
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    data_var = rate.remove_rate(data_var)
    # Phase 22d: VULCAN-JAX's `op_jax.compute_J` writes `var.k_arr[ridx, :]`
    # (dense) instead of the legacy `var.k` dict. Bridge master's parsed
    # dict into the dense surface so VULCAN-JAX's path can read/write it.
    import network as _net_mod
    import rates as _rates_mod
    _net = _net_mod.parse_network(vulcan_cfg.network)
    nz_local = int(data_atm.Tco.shape[0])
    data_var.k_arr = _rates_mod.k_array_from_dict(_net, data_var.k, nz=nz_local)
    ini_abun = InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    # This is a JAX-vs-master oracle test, so we deliberately use
    # master's `make_bins_read_cross` to populate the legacy dict
    # surface that master's `op.Ros2().compute_tau` reads. The JAX
    # `op_jax.Ros2JAX` lazy-builds its own `PhotoStaticInputs` from
    # `(var, atm)` on first call, so it doesn't need the dicts.
    rate.make_bins_read_cross(data_var, data_atm)
    make_atm.read_sflux(data_var, data_atm)

    import op_jax  # JAX-wired solver

    solver_ref = op.Ros2()
    solver_jax = op_jax.Ros2JAX()

    # 1) Reference: VULCAN's NumPy compute_tau / flux / J
    solver_ref.compute_tau(data_var, data_atm)
    tau_ref = data_var.tau.copy()
    solver_ref.compute_flux(data_var, data_atm)
    aflux_ref = data_var.aflux.copy()
    sflux_ref = data_var.sflux.copy()
    dflux_d_ref = data_var.dflux_d.copy()
    dflux_u_ref = data_var.dflux_u.copy()
    solver_ref.compute_J(data_var, data_atm)
    J_sp_ref = {k: v.copy() for k, v in data_var.J_sp.items()}

    # 2) Reset prior dflux_u/d to zero so JAX's first-call carry matches the
    #    pre-compute_flux state (matches what VULCAN sees on the very first
    #    compute_flux call, since make_bins_read_cross initializes them to zero).
    data_var.dflux_u = np.zeros_like(data_var.dflux_u)
    data_var.dflux_d = np.zeros_like(data_var.dflux_d)
    # And re-call op.compute_flux to get the bit-equivalent reference for the
    # first-call (zero carry) condition. (Steady-state vs first-call differ
    # only in the carry; with carry zero they are identical.)
    solver_ref.compute_flux(data_var, data_atm)
    aflux_ref_first = data_var.aflux.copy()

    # 3) JAX-wired
    data_var.dflux_u = np.zeros_like(data_var.dflux_u)
    data_var.dflux_d = np.zeros_like(data_var.dflux_d)
    solver_jax.compute_tau(data_var, data_atm)
    tau_jax = data_var.tau.copy()
    solver_jax.compute_flux(data_var, data_atm)
    aflux_jax = data_var.aflux.copy()
    solver_jax.compute_J(data_var, data_atm)
    J_sp_jax = {k: v.copy() for k, v in data_var.J_sp.items()}

    ok = True

    # tau comparison
    nonzero = tau_ref > 0
    relerr_tau = np.max(np.abs(tau_jax - tau_ref)[nonzero] / tau_ref[nonzero])
    print(f"compute_tau   relerr: {relerr_tau:.3e}")
    if relerr_tau > 1e-12:
        print("FAIL compute_tau relerr too large")
        ok = False

    # aflux comparison (use first-call reference, since carry is zero)
    abs_tol = 1e-30
    relerr_aflux = np.abs(aflux_jax - aflux_ref_first) / np.maximum(
        np.abs(aflux_ref_first), abs_tol
    )
    nonzero_mask = np.abs(aflux_ref_first) > abs_tol
    if nonzero_mask.any():
        max_relerr_aflux = float(relerr_aflux[nonzero_mask].max())
    else:
        max_relerr_aflux = 0.0
    print(f"compute_flux  relerr: {max_relerr_aflux:.3e}")
    if max_relerr_aflux > 1e-9:
        print("FAIL compute_flux relerr too large")
        ok = False

    # J_sp comparison
    max_J_relerr = 0.0
    n_compared = 0
    for k, ref_arr in J_sp_ref.items():
        if k not in J_sp_jax:
            continue
        ours = np.asarray(J_sp_jax[k], dtype=np.float64)
        if ref_arr.max() < 1e-30:
            continue
        err = float(np.max(np.abs(ours - ref_arr) / np.maximum(np.abs(ref_arr), 1e-30)))
        if err > max_J_relerr:
            max_J_relerr = err
        n_compared += 1
    print(f"compute_J     relerr: {max_J_relerr:.3e} over {n_compared} entries")
    if max_J_relerr > 1e-9:
        print("FAIL compute_J relerr too large")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
