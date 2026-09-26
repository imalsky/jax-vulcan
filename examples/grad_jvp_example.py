"""Forward- and reverse-mode AD through one Ros2 step (`jax_ros2_step`).

The runner's `lax.while_loop` supports forward mode only. A certified
end-to-end tangent comes from `OuterLoop.run_jvp`; reverse-mode
sensitivities at the converged state come from `steady_state_grad`
(see grad_reverse_example.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import jax
import jax.numpy as jnp
import numpy as np

import vulcan_jax.network as _net_mod
import vulcan_jax.chem as _chem_mod
from vulcan_jax.config import default_config
from vulcan_jax.jax_step import jax_ros2_step, AtmStatic


def main() -> int:
    # ---- Build a small synthetic atmosphere just big enough to step. ----
    net = _chem_mod.to_jax(_net_mod.parse_network(default_config().network))
    nz, ni = 12, net.ni
    y = jnp.full((nz, ni), 1e10)
    k_arr = jnp.full((net.nr + 1, nz), 1e-12)
    atm = AtmStatic(
        Kzz=jnp.full((nz - 1,), 1e10),
        Dzz=jnp.full((nz - 1, ni), 1e2),
        dzi=jnp.full((nz - 1,), 1e5),
        vz=jnp.zeros(nz - 1),
        Hpi=jnp.full((nz - 1,), 1e6),
        Ti=jnp.full((nz - 1,), 1500.0),
        Tco=jnp.full((nz,), 1500.0),
        g=jnp.full((nz,), 2140.0),
        ms=jnp.full((ni,), 1e-23),
        alpha=jnp.zeros(ni),
        M=jnp.full((nz,), 1e15),
        vm=jnp.zeros((nz - 1, ni)),
        vs=jnp.zeros((nz - 1, ni)),
        top_flux=jnp.zeros((ni,)),
        bot_flux=jnp.zeros((ni,)),
        bot_vdep=jnp.zeros((ni,)),
        gas_indx_mask=jnp.ones((ni,), dtype=jnp.bool_),
        diff_esc_mask=jnp.zeros((ni,), dtype=jnp.bool_),
        use_vm_mol=jnp.bool_(False),
        use_settling=jnp.bool_(False),
        use_topflux=jnp.bool_(False),
        use_botflux=jnp.bool_(False),
    )

    # ---- Forward-mode tangent: dy_out / dy_in along a random direction ----
    rng = np.random.default_rng(0)
    tangent = jnp.asarray(rng.normal(size=y.shape) * 1e8)

    def step(y_in):
        sol, _ = jax_ros2_step(y_in, k_arr, jnp.float64(1e-3), atm, net)
        return sol

    sol_primal, sol_tangent = jax.jvp(step, (y,), (tangent,))
    print(f"primal y_out[0, 0]    = {float(sol_primal[0, 0]):.6e}")
    print(f"tangent dy_out[0, 0]  = {float(sol_tangent[0, 0]):.6e}")
    print(f"finite primal:  {bool(jnp.all(jnp.isfinite(sol_primal)))}")
    print(f"finite tangent: {bool(jnp.all(jnp.isfinite(sol_tangent)))}")

    # ---- Reverse mode through the per-step kernel ----
    def loss(y_in):
        sol, _ = jax_ros2_step(y_in, k_arr, jnp.float64(1e-3), atm, net)
        return jnp.sum(sol**2)

    g = jax.grad(loss)(y)
    print(f"jax.grad through one Ros2 step finite: {bool(jnp.all(jnp.isfinite(g)))}")

    # ---- Note about the full integration loop ----
    print()
    print(
        "Note: for a certified end-to-end tangent use OuterLoop.run_jvp; for "
        "reverse-mode reaction sensitivities at the converged state see "
        "steady_state_grad.steady_state_reaction_sensitivity and "
        "examples/grad_reverse_example.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
