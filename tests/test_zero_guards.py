"""Regression tests for division-by-zero guards.

Exercises the guards added to prevent NaN/Inf when:
  1. ysum (gas-species sum) reaches zero in a layer (jax_step diffusion)
  2. mu (mean molecular weight) reaches zero in a layer (atm_refresh)
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

jax.config.update("jax_enable_x64", True)


def test_diffusion_ysum_zero_layer():
    """Diffusion coefficients must be finite even when a layer has ysum=0."""
    from vulcan_jax.jax_step import AtmStatic, _build_diff_coeffs_jax, compute_diff_grav

    nz, ni = 8, 4
    rng = np.random.default_rng(99)

    y = jnp.abs(jnp.asarray(rng.standard_normal((nz, ni)))) + 1.0
    y = y.at[3, :].set(0.0)

    gas_mask = jnp.ones(ni, dtype=bool)

    atm = AtmStatic(
        Kzz=jnp.full(nz - 1, 1e5),
        Dzz=jnp.zeros((nz - 1, ni)),
        dzi=jnp.full(nz - 1, 1e5),
        vz=jnp.zeros(nz - 1),
        Hpi=jnp.full(nz - 1, 1e7),
        Ti=jnp.full(nz - 1, 1500.0),
        Tco=jnp.full(nz, 1500.0),
        g=jnp.full(nz, 1e3),
        ms=jnp.full(ni, 2.0 * 1.66e-24),
        alpha=jnp.zeros(ni),
        M=jnp.full(nz, 1e18),
        vm=jnp.zeros((nz - 1, ni)),
        vs=jnp.zeros((nz - 1, ni)),
        top_flux=jnp.zeros(ni),
        bot_flux=jnp.zeros(ni),
        bot_vdep=jnp.zeros(ni),
        gas_indx_mask=gas_mask,
        diff_esc_mask=jnp.zeros(ni, dtype=jnp.bool_),
        use_vm_mol=False,
        use_settling=False,
        use_topflux=False,
        use_botflux=False,
    )

    grav = compute_diff_grav(atm)
    A_eddy, B_eddy, C_eddy, A_mol, B_mol, C_mol, ysum = _build_diff_coeffs_jax(
        y, atm, grav
    )

    assert jnp.all(jnp.isfinite(A_eddy)), "A_eddy contains NaN/Inf"
    assert jnp.all(jnp.isfinite(B_eddy)), "B_eddy contains NaN/Inf"
    assert jnp.all(jnp.isfinite(C_eddy)), "C_eddy contains NaN/Inf"


def test_atm_refresh_mu_zero_layer():
    """Scale-height computation must be finite even when mu=0 in a layer."""
    from vulcan_jax.atm_refresh import AtmRefreshStatic, update_mu_dz_jax

    nz, ni = 8, 4
    rng = np.random.default_rng(42)

    ymix = jnp.abs(jnp.asarray(rng.standard_normal((nz, ni)))) + 0.01
    ymix = ymix / jnp.sum(ymix, axis=1, keepdims=True)
    ymix = ymix.at[2, :].set(0.0)

    mol_mass = jnp.array([2.0, 28.0, 44.0, 18.0])
    pco = jnp.logspace(2, -4, nz)
    pico = (
        jnp.concatenate(
            [
                jnp.array([pco[0] * 1.1]),
                0.5 * (pco[:-1] + pco[1:]),
                jnp.array([pco[-1] * 0.9]),
            ]
        )
        * 1e6
    )

    st = AtmRefreshStatic(
        Tco=jnp.full(nz, 1500.0),
        pico=pico,
        mol_mass=mol_mass,
        ms=mol_mass / 6.022e23,
        Dzz_top=jnp.zeros(ni),
        diff_esc_idx=jnp.array([], dtype=jnp.int32),
        pref_indx=0,
        zco_pref=0.0,
        gs=1e3,
        Rp=7e9,
        kb=1.38e-16,
        Navo=6.022e23,
        max_flux=1e13,
        nz=nz,
        ni=ni,
    )

    mu, g, Hp, dz, zco, dzi, Hpi = update_mu_dz_jax(ymix, st)

    assert jnp.all(jnp.isfinite(mu)), "mu contains NaN/Inf"
    assert jnp.all(jnp.isfinite(Hp)), "Hp contains NaN/Inf"
    assert jnp.all(jnp.isfinite(dz)), "dz contains NaN/Inf"
    assert jnp.all(jnp.isfinite(zco)), "zco contains NaN/Inf"
