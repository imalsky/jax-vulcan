"""The two-stream sweeps as associative scans must match the sequential ones.

`photo._two_stream_sweeps` runs the two affine recurrences as associative
scans (depth log2 nz). The sequential lax.scan form below is the oracle:
the two agree to roundoff on random and HD189 inputs, and the lowering has
no while loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import vulcan_jax.photo as photo_mod

# Roundoff bars, relative to the largest entry of the sequential result.
BAR_RANDOM = 1e-12
BAR_HD189 = 1e-11


def _sweeps_sequential(chi, phi, xi, i_d, i_u, dflux_u_prev, mu_ang):
    """Sequential lax.scan form of the two-stream sweeps (oracle)."""
    nz, nbin = chi.shape
    dflux_d_top = jnp.zeros(nbin)

    def down_step(carry, j):
        # dflux_u_j is the prior call's dflux_u, not the current up sweep's value.
        dflux_u_j = dflux_u_prev[j]
        dflux_d_jp1 = carry
        dflux_d_j = (1.0 / chi[j]) * (
            phi[j] * dflux_d_jp1 - xi[j] * dflux_u_j + i_d[j] / mu_ang
        )
        return dflux_d_j, dflux_d_j

    js_down = jnp.arange(nz - 1, -1, -1)
    _, dflux_d_seq = jax.lax.scan(down_step, dflux_d_top, js_down)
    dflux_d = jnp.concatenate([dflux_d_seq[::-1], dflux_d_top[None]], axis=0)

    dflux_u_bot = jnp.zeros(nbin)

    def up_step(carry, j):
        dflux_u_jm1 = carry
        dflux_u_j = (1.0 / chi[j - 1]) * (
            phi[j - 1] * dflux_u_jm1 - xi[j - 1] * dflux_d[j] + i_u[j - 1] / mu_ang
        )
        return dflux_u_j, dflux_u_j

    js_up = jnp.arange(1, nz + 1)
    _, dflux_u_seq = jax.lax.scan(up_step, dflux_u_bot, js_up)
    dflux_u = jnp.concatenate([dflux_u_bot[None], dflux_u_seq], axis=0)
    return dflux_d, dflux_u


def _relerr(new, old) -> float:
    new = np.asarray(new)
    old = np.asarray(old)
    return float(np.abs(new - old).max() / np.abs(old).max())


def test_associative_sweeps_match_sequential():
    """Random inputs of the real shapes, built through the production algebra."""
    rng = np.random.default_rng(20240917)
    nz, nbin = 150, 256
    edd = 0.5
    mu_ang = -0.5

    # chi / xi / phi as compute_flux_jax builds them: chi < 0, |phi/chi| <= 1.
    w0 = rng.uniform(0.0, 1.0 - 1e-8, (nz, nbin))
    delta_tau = rng.uniform(0.0, 3.0, (nz, nbin))
    tran = np.exp(-1.0 / edd * (1.0 - w0) ** 0.5 * delta_tau)
    zeta_p = 0.5 * (1.0 + (1.0 - w0) ** 0.5)
    zeta_m = 0.5 * (1.0 - (1.0 - w0) ** 0.5)
    chi = zeta_m**2 * tran**2 - zeta_p**2
    chi = np.where(chi > -photo_mod.UNDERFLOW_DENOM, -photo_mod.UNDERFLOW_DENOM, chi)
    xi = zeta_p * zeta_m * (1.0 - tran**2)
    phi = (zeta_m**2 - zeta_p**2) * tran

    i_d = rng.uniform(0.0, 1e3, (nz, nbin))
    i_u = rng.uniform(0.0, 1e3, (nz, nbin))
    dflux_u_prev = rng.uniform(0.0, 1e3, (nz + 1, nbin))

    args = [jnp.asarray(x) for x in (chi, phi, xi, i_d, i_u, dflux_u_prev)]
    d_new, u_new = photo_mod._two_stream_sweeps(*args, mu_ang)
    d_old, u_old = _sweeps_sequential(*args, mu_ang)

    assert _relerr(d_new, d_old) < BAR_RANDOM
    assert _relerr(u_new, u_old) < BAR_RANDOM


def test_associative_sweeps_on_hd189_state(hd189_state):
    """Real HD189 photo state: same fluxes, and no while loop in the lowering."""
    from vulcan_jax import phy_const
    from vulcan_jax.config import default_config

    cfg = default_config()
    if not cfg.use_photo:
        pytest.skip("use_photo=False; no two-stream state to compare")

    var, atm, solver = hd189_state.var, hd189_state.atm, hd189_state.solver
    solver.compute_tau(var, atm)  # populates var.tau and the PhotoData cache
    ag0 = float(phy_const.ag0)
    args = (
        jnp.asarray(var.tau),
        jnp.asarray(var.sflux_top),
        jnp.asarray(var.ymix),
        solver._photo_data,
        jnp.asarray(solver._photo_static.bins, dtype=jnp.float64),
        float(np.cos(cfg.sl_angle)),
        float(cfg.edd),
        ag0,
        float(phy_const.hc),
        jnp.asarray(var.dflux_u),
    )
    ag0_is_zero = ag0 == 0.0

    aflux, _, dflux_d, dflux_u = photo_mod.compute_flux_jax(
        *args, ag0_is_zero=ag0_is_zero
    )

    original = photo_mod._two_stream_sweeps
    photo_mod._two_stream_sweeps = _sweeps_sequential
    photo_mod.compute_flux_jax.clear_cache()
    try:
        ref = photo_mod.compute_flux_jax(*args, ag0_is_zero=ag0_is_zero)
        aflux_ref, _, dflux_d_ref, dflux_u_ref = [np.asarray(a) for a in ref]
    finally:
        photo_mod._two_stream_sweeps = original
        photo_mod.compute_flux_jax.clear_cache()

    assert _relerr(aflux, aflux_ref) < BAR_HD189
    assert _relerr(dflux_d, dflux_d_ref) < BAR_HD189
    assert _relerr(dflux_u, dflux_u_ref) < BAR_HD189

    text = photo_mod.compute_flux_jax.lower(*args, ag0_is_zero=ag0_is_zero).as_text()
    assert text.count("stablehlo.while") == 0
