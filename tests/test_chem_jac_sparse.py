"""Validate chem_jac_analytical_per_layer (Phase 11 / B.1+B.2) against
the existing jacrev-based dense Jacobian on the HD189 reference state.

The analytical Jacobian is built from stoichiometry directly:
    J[i, j] = Σ_{rxns r} Σ_{out slot s_i, reac slot s_j}
                 sign_i * stoich_i * (stoich_j / y_j) * rate[r]

vs. `chem_jac` which materialises the same matrix via `jax.jacrev`. Both
must agree to machine precision on a real (y, M, k) state.

Standalone test: does NOT require ../VULCAN-master/. Uses VULCAN-JAX's
local rate-coef + atmosphere build path (legacy_io.ReadRate + build_atm).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def main() -> int:
    import vulcan_cfg
    import store, build_atm
    import legacy_io as op
    import chem_funs  # noqa: F401  (network parse triggers chem_funs setup)

    import jax.numpy as jnp
    import network as net_mod
    import chem as chem_mod

    # --- Build canonical HD189 state ---
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
    data_var = rate.remove_rate(data_var)
    ini = build_atm.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)

    # Real HD189 state
    y = jnp.asarray(data_var.y, dtype=jnp.float64)              # [nz, ni]
    M = jnp.asarray(data_atm.M, dtype=jnp.float64)              # [nz]
    nz, ni = y.shape

    # Pack k dict to JAX array [nr+1, nz]
    net = net_mod.parse_network(vulcan_cfg.network)
    net_jax = chem_mod.to_jax(net)
    k_arr = np.zeros((net.nr + 1, nz), dtype=np.float64)
    for i, vec in data_var.k.items():
        k_arr[i] = np.asarray(vec, dtype=np.float64)
    k_arr = jnp.asarray(k_arr)

    print(f"State: nz={nz}, ni={ni}, nr={net.nr}")

    # --- Compute both Jacobians ---
    J_dense = np.asarray(chem_mod.chem_jac(y, M, k_arr, net_jax))           # [nz, ni, ni]
    J_anal = np.asarray(chem_mod.chem_jac_analytical(y, M, k_arr, net_jax)) # [nz, ni, ni]

    # --- Compare ---
    diff = np.abs(J_anal - J_dense)
    abs_max_dense = max(np.abs(J_dense).max(), 1e-300)
    relerr = diff / np.maximum(np.abs(J_dense), 1e-30)
    # Use a sane absolute floor for cells near zero (cancellation noise).
    rel_significant = np.where(np.abs(J_dense) > 1e-12 * abs_max_dense,
                               relerr, 0.0)
    max_rel = float(rel_significant.max())
    max_abs = float(diff.max())

    print(f"chem_jac shape:        {J_dense.shape}")
    print(f"max |J_dense|:         {abs_max_dense:.3e}")
    print(f"max |J_anal - J_dense|:{max_abs:.3e}")
    print(f"max rel err (significant cells): {max_rel:.3e}")

    # Tolerance: analytical and AD-built Jacobians should agree to machine
    # precision per entry. Allow 1e-12 to accommodate float64 reduction-order
    # differences between the two scatter patterns.
    ok = max_rel < 1e-12
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
