"""Validate chem_jac_analytical_per_layer against the jacrev-based dense
Jacobian on the HD189 reference state.

The analytical Jacobian is built from stoichiometry directly:
    J[i, j] = Σ_{rxns r} Σ_{out slot s_i, reac slot s_j}
                 sign_i * stoich_i * (stoich_j / y_j) * rate[r]

vs. `chem_jac` which materialises the same matrix via `jax.jacrev`. Both
must agree to machine precision on a real (y, M, k) state.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

JAC_RTOL = 1e-12  # the same matrix by two summation orders
SIGNIFICANT_FRAC = 1e-12  # entries below this fraction of the peak are cancellation noise


def test_main(hd189_state):
    """chem_jac_analytical vs the jacrev chem_jac on the HD189 state."""
    from _oracles import chem_jac

    import vulcan_jax.chem as chem_mod
    import jax.numpy as jnp
    import vulcan_jax.network as net_mod
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    data_var, data_atm = hd189_state.var, hd189_state.atm
    y = jnp.asarray(data_var.y, dtype=jnp.float64)  # [nz, ni]
    M = jnp.asarray(data_atm.M, dtype=jnp.float64)  # [nz]

    net = net_mod.parse_network(vulcan_cfg.network)
    net_jax = chem_mod.to_jax(net)
    k_arr = jnp.asarray(np.asarray(data_var.k_arr, dtype=np.float64))

    J_dense = np.asarray(chem_jac(y, M, k_arr, net_jax))  # [nz, ni, ni]
    J_anal = np.asarray(
        chem_mod.chem_jac_analytical(y, M, k_arr, net_jax)
    )  # [nz, ni, ni]

    diff = np.abs(J_anal - J_dense)
    abs_max_dense = max(np.abs(J_dense).max(), 1e-300)
    relerr = diff / np.maximum(np.abs(J_dense), 1e-30)

    rel_significant = np.where(np.abs(J_dense) > SIGNIFICANT_FRAC * abs_max_dense, relerr, 0.0)
    max_rel = float(rel_significant.max())
    assert max_rel < JAC_RTOL, f"max rel err (significant cells): {max_rel:.3e}"
