"""Validate JAX photochemistry kernels against VULCAN-master.

Runs VULCAN's compute_tau on the canonical HD189 state, then runs the JAX
version on the same state and compares element-wise.
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

# Oracle test: requires VULCAN-master sibling for the upstream op.compute_tau /
# op.compute_flux / op.compute_J reference. Skip cleanly when absent.
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
    import jax.numpy as jnp
    import vulcan_cfg
    import store
    import build_atm
    import op
    import photo as photo_mod

    # === Set up state with photochemistry ===
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    ini = build_atm.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    # Photochem setup
    if vulcan_cfg.use_photo:
        rate.make_bins_read_cross(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)

    # Run VULCAN's compute_tau
    solver_v = op.Ros2()
    solver_v.compute_tau(data_var, data_atm)
    tau_ref = data_var.tau.copy()
    print(f"VULCAN tau shape: {tau_ref.shape}")
    print(f"  range: [{tau_ref.min():.3e}, {tau_ref.max():.3e}]")

    # === Pack photo data and run JAX version ===
    import chem_funs
    species_list = chem_funs.spec_list

    photo_data = photo_mod.pack_photo_data(data_var, vulcan_cfg, species_list)
    print(f"\nPacked photo data:")
    print(f"  absp species (non-T):  {photo_data.absp_idx.shape[0]}")
    print(f"  absp species (T-dep):  {photo_data.absp_T_idx.shape[0]}")
    print(f"  scat species:          {photo_data.scat_idx.shape[0]}")

    tau_jax = np.asarray(photo_mod.compute_tau_jax(
        jnp.asarray(data_var.y), jnp.asarray(data_atm.dz), photo_data
    ))

    # Compare
    if tau_jax.shape != tau_ref.shape:
        print(f"FAIL shape mismatch: jax={tau_jax.shape} ref={tau_ref.shape}")
        return 1

    abs_tol = 1e-30
    relerr = np.abs(tau_jax - tau_ref) / np.maximum(np.abs(tau_ref), abs_tol)
    max_relerr = relerr.max()
    print(f"\ncompute_tau max relerr: {max_relerr:.3e}")

    # Locate worst
    worst = np.unravel_index(relerr.argmax(), relerr.shape)
    print(f"  worst at layer {worst[0]}, bin {worst[1]}: jax={tau_jax[worst]:.4e} ref={tau_ref[worst]:.4e}")

    # === compute_flux validation ===
    # Reset VULCAN's dflux_u/d to zero so JAX's zero-init matches first-call behavior
    data_var.dflux_u = np.zeros_like(data_var.dflux_u)
    data_var.dflux_d = np.zeros_like(data_var.dflux_d)
    data_var.tau = tau_ref   # restore tau (may have been modified)
    data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))
    solver_v.compute_flux(data_var, data_atm)
    aflux_ref = data_var.aflux.copy()
    print(f"\nVULCAN aflux shape: {aflux_ref.shape}, range [{aflux_ref.min():.3e}, {aflux_ref.max():.3e}]")

    from phy_const import hc
    aflux_jax, _, _, _ = photo_mod.compute_flux_jax(
        jnp.asarray(tau_ref),
        jnp.asarray(data_var.sflux_top),
        jnp.asarray(data_var.ymix),
        photo_data,
        jnp.asarray(data_var.bins),
        float(np.cos(vulcan_cfg.sl_angle)),
        float(vulcan_cfg.edd),
        float(0.0),                 # ag0=0 in phy_const
        float(hc),
        jnp.zeros_like(jnp.asarray(data_var.dflux_u)),  # dflux_u_prev = zeros (first-call match)
        ag0_is_zero=True,
    )
    aflux_jax = np.asarray(aflux_jax)

    abs_tol2 = 1e-30
    relerr2 = np.abs(aflux_jax - aflux_ref) / np.maximum(np.abs(aflux_ref), abs_tol2)
    # Filter to non-trivial flux values to avoid noise comparisons
    nonzero_mask = np.abs(aflux_ref) > 1e-30
    if nonzero_mask.any():
        max_relerr2 = relerr2[nonzero_mask].max()
    else:
        max_relerr2 = 0.0
    print(f"compute_flux max relerr (nontrivial entries): {max_relerr2:.3e}")

    # === compute_J validation ===
    solver_v.compute_J(data_var, data_atm)
    J_ref = dict(data_var.J_sp)
    print(f"\nVULCAN J_sp: {len(J_ref)} (species, branch) entries")

    photo_J = photo_mod.pack_photo_J_data(data_var, vulcan_cfg)
    J_jax = photo_mod.compute_J_jax(jnp.asarray(aflux_jax), photo_J)
    print(f"JAX J_jax:   {len(J_jax)} (species, branch) entries")

    max_relerr3 = 0.0
    n_compared = 0
    n_failed = 0
    for k in photo_J.branch_keys + photo_J.branch_T_keys:
        if k not in J_ref:
            continue
        ref = np.asarray(J_ref[k], dtype=np.float64)
        jax_val = np.asarray(J_jax[k], dtype=np.float64)
        err = np.max(np.abs(jax_val - ref) / np.maximum(np.abs(ref), 1e-30))
        if err > max_relerr3:
            max_relerr3 = err
        n_compared += 1
        if err > 1e-6 and ref.max() > 1e-30:
            n_failed += 1
    print(f"compute_J max relerr: {max_relerr3:.3e} (over {n_compared} branches)")

    print()
    ok = (
        max_relerr < 1e-10
        and max_relerr2 < 1e-6
        and max_relerr3 < 1e-6
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
