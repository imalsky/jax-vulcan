"""Self-consistent gravity in the hydrostatic atm refresh.

update_mu_dz_jax computes each layer's g from the zco the same sweep
produced, as master's op.update_mu_dz does. Checks:
- g == gs*(Rp/(Rp+zco))**2 to roundoff on both sides of pref_indx.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

GRAVITY_RTOL = 1e-13  # g recomputed from zco agrees to roundoff


def test_refreshed_gravity_matches_refreshed_zco(hd189_state):
    import vulcan_jax.legacy_io as op
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax import atm_refresh

    # Perturb ymix so mu, and with it zco, differ from the pre-loop profile.
    ymix = hd189_state.var.ymix
    ymix = ymix * (1.0 + 1e-3 * np.random.default_rng(0).standard_normal(ymix.shape))
    ymix = ymix / ymix.sum(axis=1, keepdims=True)
    st = outer_loop.OuterLoop(hd189_state.solver, op.Output())._build_refresh_static(
        hd189_state.atm
    )
    _, g, _, _, zco, _, _ = atm_refresh.update_mu_dz_jax(jnp.asarray(ymix), st)
    g, zco = np.asarray(g), np.asarray(zco)
    p, nz, gs, Rp = int(st.pref_indx), g.shape[0], float(st.gs), float(st.Rp)
    # zco has nz+1 entries: the upward scan (i >= pref_indx) takes g[i] from
    # zco[i], the downward scan (i < pref_indx) from zco[i+1].
    z_used = np.concatenate([zco[1 : p + 1], zco[p:nz]])
    assert np.max(np.abs(g / (gs * (Rp / (Rp + z_used)) ** 2) - 1.0)) < GRAVITY_RTOL
    # g varies far more than the tolerance, so the check is not vacuous.
    assert np.ptp(g) / gs > 1e-3
