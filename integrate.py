"""Fixed-dt JAX integration loop for validation and benchmarks.

Assumes frozen rate constants (no photo, no condensation, no fix_species).
For production, use `outer_loop.OuterLoop`.
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
    """Take N fixed-dt Ros2 steps. Returns (y_final, deltas)."""
    def body_fn(y, _):
        sol, delta_arr = jax_ros2_step(y, k_arr, dt, atm, net)
        delta = jnp.max(delta_arr / jnp.maximum(jnp.abs(sol), 1e-30))
        return sol, delta

    y_final, deltas = jax.lax.scan(body_fn, y0, jnp.arange(n_steps))
    return y_final, deltas
