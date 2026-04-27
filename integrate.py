"""Pure-JAX fixed-dt integration loop for validation + benchmarking.

`jax_integrate_fixed_dt` takes N fixed-dt Ros2 steps in a single
`jax.lax.scan`. JIT'd, vmap-able. Useful for two things:

  1. Validating `jax_ros2_step` against an equivalent Python loop driving
     the same kernel (the test uses this to confirm the JIT/scan path
     gives bit-equivalent output).
  2. Benchmarks where dt is externally controlled and we want the inner
     stepper to be JAX-fused (no Python overhead between steps).

Assumes:
  - Rate constants `k_arr` are FROZEN throughout the integration (no photo
    update, no condensation).
  - No condensation, no ion charge balance, no fix_species clamping.

For the production drop-in path with full feature support — adaptive dt,
photo update, condensation, ion balance, fix-all-bot, convergence-driven
termination, all inside one JIT'd while_loop — use
`outer_loop.OuterLoop`. Phase 10.6 (the previous `jax_integrate_adaptive`
function) is now subsumed by `outer_loop`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from chem import NetworkArrays
from jax_step import AtmStatic, jax_ros2_step

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("n_steps",))
def jax_integrate_fixed_dt(
    y0: jnp.ndarray,            # (nz, ni)
    k_arr: jnp.ndarray,         # (nr+1, nz)
    dt: float,                  # fixed step size
    n_steps: int,               # number of steps (compile-time constant)
    atm: AtmStatic,
    net: NetworkArrays,
):
    """Take N fixed-dt Ros2 steps. Returns (y_final, deltas).

    `n_steps` is a Python int (concrete) so jax.lax.scan can unroll/compile.
    """
    def body_fn(y, _):
        sol, delta_arr = jax_ros2_step(y, k_arr, dt, atm, net)
        delta = jnp.max(delta_arr / jnp.maximum(jnp.abs(sol), 1e-30))
        return sol, delta

    y_final, deltas = jax.lax.scan(body_fn, y0, jnp.arange(n_steps))
    return y_final, deltas
