"""Pure-JAX integration loops for VULCAN-JAX.

Two flavors:
  - `jax_integrate_fixed_dt`: takes N fixed-dt Ros2 steps. JIT'd, vmap-able.
    Useful for validation and for use cases where dt is externally controlled
    (e.g. an outer Python loop manages dt adaptation but wants the inner stepper
    to be JAX-fused).
  - `jax_integrate_adaptive`: adaptive dt + accept/reject inside `jax.lax.while_loop`.
    Useful for fully-JAX integration (max GPU/TPU performance, vmap'd parameter
    sweeps).

Both functions assume:
  - Rate constants `k_arr` are FROZEN throughout the integration (no photo
    update). For runs with photochem, the outer Python loop must call
    photo update routines and re-pass `k_arr`.
  - No condensation, no ion charge balance, no fix_species clamping. Those
    require additional state and conditional updates that this minimal loop
    doesn't include yet (see Task #20 follow-ups in STATUS.md).

For the production drop-in path with full feature support, use
`vulcan_jax.py` which goes through `op_jax.Ros2JAX`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chem import NetworkArrays
from jax_step import AtmStatic, jax_ros2_step

jax.config.update("jax_enable_x64", True)


from functools import partial


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
    For `n_steps`-varying calls, use `jax_integrate_adaptive`.
    """
    def body_fn(y, _):
        sol, delta_arr = jax_ros2_step(y, k_arr, dt, atm, net)
        delta = jnp.max(delta_arr / jnp.maximum(jnp.abs(sol), 1e-30))
        return sol, delta

    y_final, deltas = jax.lax.scan(body_fn, y0, jnp.arange(n_steps))
    return y_final, deltas


@jax.jit
def jax_integrate_adaptive(
    y0: jnp.ndarray,            # (nz, ni)
    k_arr: jnp.ndarray,         # (nr+1, nz)
    dt0: float,                 # initial dt
    max_steps: int,             # safety cap
    rtol: float,                # adaptive tolerance
    dt_min: float,
    dt_max: float,
    atm: AtmStatic,
    net: NetworkArrays,
):
    """Adaptive-dt integration loop in pure JAX.

    Mirrors VULCAN's `step_size` logic:
        h_new = h * 0.9 * (rtol / delta) ^ 0.5
        clipped to [dt_min, dt_max]

    Carry state: (y, dt, t, count). Stops when count >= max_steps.

    Caveats:
      - Doesn't reject and retry steps. If `delta > rtol`, dt is reduced for
        the NEXT step but the current step is still accepted. VULCAN-master
        retries the step. Acceptable for smooth atmospheres (HD189-style).
      - Doesn't update photochem rates -- k_arr is frozen.
      - Doesn't apply BC clamps (use_fix_sp_bot, fix_e_indx, fix_species).
    """
    def cond_fn(state):
        _, _, _, count = state
        return count < max_steps

    def body_fn(state):
        y, dt, t, count = state
        sol, delta_arr = jax_ros2_step(y, k_arr, dt, atm, net)
        delta = jnp.max(delta_arr / jnp.maximum(jnp.abs(sol), 1e-30))
        # Adapt dt
        h_new = dt * 0.9 * (rtol / jnp.maximum(delta, 1e-300)) ** 0.5
        h_new = jnp.clip(h_new, dt_min, dt_max)
        return (sol, h_new, t + dt, count + 1)

    init_state = (y0, dt0, 0.0, 0)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    y_final, dt_final, t_final, count_final = final_state
    return y_final, dt_final, t_final, count_final
