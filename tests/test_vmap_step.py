"""jax.vmap(jax_ros2_step) over identical HD189 states must reproduce the single step (sol and delta) to 1e-12."""

from __future__ import annotations

import numpy as np
from _helpers import relerr


def test_vmapped_step_matches_single_step(hd189_state):
    import jax
    import jax.numpy as jnp

    import vulcan_jax.chem as chem_mod
    import vulcan_jax.jax_step as js_mod
    import vulcan_jax.network as net_mod
    from vulcan_jax.config import default_config

    y0 = jnp.asarray(np.asarray(hd189_state.var.y, dtype=np.float64))
    k_arr = jnp.asarray(np.asarray(hd189_state.var.k_arr, dtype=np.float64))
    nz, ni = y0.shape
    net_jax = chem_mod.to_jax(net_mod.parse_network(default_config().network))
    atm_static = js_mod.make_atm_static(hd189_state.atm, ni, nz)
    dt = 1e-10

    single = js_mod.jax_ros2_step(y0, k_arr, dt, atm_static, net_jax)

    # vmap over 4 replicas of y, k_arr; broadcast dt, atm_static, net_jax
    batch = 4
    vstep = jax.jit(jax.vmap(js_mod.jax_ros2_step, in_axes=(0, 0, None, None, None)))
    batched = vstep(
        jnp.stack([y0] * batch), jnp.stack([k_arr] * batch), dt, atm_static, net_jax
    )

    # The solution and the truncation-error array the step controller reads.
    for name, got, ref in zip(("sol", "delta"), batched, single):
        for b in range(batch):
            err = relerr(np.asarray(got)[b], ref)
            assert err < 1e-12, (name, b, err)
