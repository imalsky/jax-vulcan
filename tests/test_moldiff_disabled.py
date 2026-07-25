"""Regression tests for disabled molecular diffusion transport terms."""

from __future__ import annotations

import jax.numpy as jnp


def test_zero_dzz_with_zero_hpi_keeps_molecular_terms_finite() -> None:
    """Disabled molecular diffusion must not evaluate 0 * inf as NaN."""
    from vulcan_jax.jax_step import (
        AtmStatic,
        _build_diff_coeffs_jax,
        compute_diff_grav,
    )

    nz = 5
    ni = 3
    atm = AtmStatic(
        Kzz=jnp.ones(nz - 1, dtype=jnp.float64),
        Dzz=jnp.zeros((nz - 1, ni), dtype=jnp.float64),
        dzi=jnp.ones(nz - 1, dtype=jnp.float64),
        vz=jnp.zeros(nz - 1, dtype=jnp.float64),
        Hpi=jnp.zeros(nz - 1, dtype=jnp.float64),
        Ti=jnp.full(nz - 1, 1000.0, dtype=jnp.float64),
        Tco=jnp.full(nz, 1000.0, dtype=jnp.float64),
        g=jnp.full(nz, 1.0e5, dtype=jnp.float64),
        ms=jnp.ones(ni, dtype=jnp.float64),
        alpha=jnp.zeros(ni, dtype=jnp.float64),
        M=jnp.ones(nz, dtype=jnp.float64),
        vm=jnp.zeros((nz - 1, ni), dtype=jnp.float64),
        vs=jnp.zeros((nz - 1, ni), dtype=jnp.float64),
        top_flux=jnp.zeros(ni, dtype=jnp.float64),
        bot_flux=jnp.zeros(ni, dtype=jnp.float64),
        bot_vdep=jnp.zeros(ni, dtype=jnp.float64),
        gas_indx_mask=jnp.ones(ni, dtype=jnp.bool_),
        diff_esc_mask=jnp.zeros(ni, dtype=jnp.bool_),
        use_vm_mol=False,
        use_settling=False,
        use_topflux=False,
        use_botflux=False,
    )
    y = jnp.ones((nz, ni), dtype=jnp.float64)

    grav = compute_diff_grav(atm)
    coeffs = _build_diff_coeffs_jax(y, atm, grav)

    for array in (*grav, *coeffs):
        assert bool(jnp.all(jnp.isfinite(array)))
