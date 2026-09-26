"""Pure-JAX atmosphere-refresh kernels: mu/dz/g/Hp + diffusion-limited escape.

The hydrostatic loop is a true sequential dependency
(`zco[i+1] = zco[i] + dz[i]`) and uses two `lax.scan`s — one upward from
`pref_indx` and one downward to 0; the downward scan is length-0 when
`pref_indx == 0`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .phy_const import UNDERFLOW_DENOM


def hydrostatic_step(T, mu, g, p_lo, p_hi, kb, Navo):
    """Pressure scale height `Hp` and layer thickness `dz` for one layer.

    The single definition of the formula: `atm_setup._scan_up_mu_dz_g` and
    `_scan_down_mu_dz_g` call it too. It lives here because `atm_setup`
    imports `atm_refresh`, not the other way round. `UNDERFLOW_DENOM` binds
    only when `mu*g` underflows to zero (a layer with no gas), so it is a
    no-op on any physical column.
    """
    denom = jnp.maximum(mu / Navo * g, UNDERFLOW_DENOM)
    Hp = kb * T / denom
    return Hp, Hp * jnp.log(p_lo / p_hi)


class AtmRefreshStatic(NamedTuple):
    """Closed-over static inputs to the atm-refresh kernels."""

    Tco: jnp.ndarray  # (nz,)        layer-center temperature (K)
    pico: jnp.ndarray  # (nz+1,)      interface pressure (dyne/cm^2)
    mol_mass: jnp.ndarray  # (ni,)        molar mass per species (g/mol)
    ms: jnp.ndarray  # (ni,)        molar mass per species (g/mol; /Navo in formulas)
    Dzz_top: jnp.ndarray  # (ni,)        molecular diffusion at the top interface
    diff_esc_idx: (
        jnp.ndarray
    )  # (n_esc,) int32  species indices for diffusion-limited escape
    pref_indx: int  # static       reference layer index (zco==zco_pref here)
    zco_pref: float  # static       z at the reference layer (0.0 for gas giants)
    gs: float  # surface gravity at zco_pref (cm/s^2)
    Rp: float  # planetary radius (cm)
    kb: float
    Navo: float
    max_flux: float  # cap on |top_flux| (cfg.max_flux)
    nz: int  # static
    ni: int  # static


def update_mu_dz_jax(ymix: jnp.ndarray, st: AtmRefreshStatic):
    """Recompute (mu, g, Hp, dz, zco, dzi, Hpi) from the current ymix (nz, ni).

    Shapes: mu (nz,), g (nz,), Hp (nz,), dz (nz,), zco (nz+1,), dzi (nz-1,),
    Hpi (nz-1,). `Ti` is intentionally not returned — it depends only on the
    static `Tco` and is captured once at OuterLoop init.
    """
    Tco = st.Tco
    pico = st.pico
    mol_mass = st.mol_mass
    pref_indx = st.pref_indx
    gs = st.gs
    Rp = st.Rp
    kb = st.kb
    Navo = st.Navo
    nz = st.nz
    zco_pref = jnp.float64(st.zco_pref)

    mu = jnp.einsum("zi,i->z", ymix, mol_mass)

    # Forward scan upward from `pref_indx` to nz-1.
    def fwd_step(zco_i, i):
        # g at the reference layer is gs by definition; above it, g follows
        # the inverse-square law from Rp+zco[i].
        is_pref = i == pref_indx
        g_i = jnp.where(is_pref, gs, gs * (Rp / (Rp + zco_i)) ** 2)
        Hp_i, dz_i = hydrostatic_step(
            Tco[i], mu[i], g_i, pico[i], pico[i + 1], kb, Navo
        )
        zco_ip1 = zco_i + dz_i
        return zco_ip1, (g_i, Hp_i, dz_i, zco_ip1)

    fwd_indices = jnp.arange(pref_indx, nz, dtype=jnp.int32)
    _, (g_fwd, Hp_fwd, dz_fwd, zco_above) = jax.lax.scan(
        fwd_step,
        zco_pref,
        fwd_indices,
    )

    # Backward scan downward from pref_indx-1 to 0; length 0 when pref_indx==0.
    def bwd_step(zco_ip1, i):
        # The layer sits below zco[i+1], so g uses Rp+zco[i+1].
        g_i = gs * (Rp / (Rp + zco_ip1)) ** 2
        Hp_i, dz_i = hydrostatic_step(
            Tco[i], mu[i], g_i, pico[i], pico[i + 1], kb, Navo
        )
        zco_i = zco_ip1 - dz_i
        return zco_i, (g_i, Hp_i, dz_i, zco_i)

    # `reverse=True` walks i from pref_indx-1 down to 0 and stacks each output
    # at its own index, so the results come out in canonical 0..pref_indx-1
    # order with no flips.
    bwd_indices = jnp.arange(0, pref_indx, dtype=jnp.int32)
    _, (g_bwd, Hp_bwd, dz_bwd, zco_below) = jax.lax.scan(
        bwd_step,
        zco_pref,
        bwd_indices,
        reverse=True,
    )

    g = jnp.concatenate([g_bwd, g_fwd])
    Hp = jnp.concatenate([Hp_bwd, Hp_fwd])
    dz = jnp.concatenate([dz_bwd, dz_fwd])
    zco = jnp.concatenate([zco_below, zco_pref[None], zco_above])

    dzi = 0.5 * (dz[:-1] + dz[1:])
    Hpi = 0.5 * (Hp[:-1] + Hp[1:])

    return mu, g, Hp, dz, zco, dzi, Hpi


def recompute_vm_jax(
    g: jnp.ndarray,
    Hpi: jnp.ndarray,
    dzi: jnp.ndarray,
    Dzz: jnp.ndarray,
    ms: jnp.ndarray,
    alpha: jnp.ndarray,
    Tco: jnp.ndarray,
    kb: float,
    Navo: float,
) -> jnp.ndarray:
    """Recompute the interface molecular-diffusion drift velocity `vm` (nz-1, ni).

    `vm` depends on composition (through `Hpi`) and on `g`, so it is refreshed
    with the geometry every `update_frq` steps, as `op.update_mu_dz` does
    (vm_branch). The frozen `Dzz`, `ms`, `alpha` and `Tco` come from
    `AtmStatic`; `g`/`Hpi`/`dzi` are the refreshed geometry from the carry.
    Formula as `atm_setup.compute_mol_diff`.
    """
    Ti = 0.5 * (Tco[:-1] + Tco[1:])  # (nz-1,)
    delta_Ti = Tco[1:] - Tco[:-1]  # (nz-1,)
    inv_Hi = ms[None, :] * g[:, None] / (Navo * kb * Tco[:, None])  # (nz, ni)
    H_cell = 1.0 / inv_Hi
    Hi_interf = 1.0 / (0.5 * (H_cell[:-1] + H_cell[1:]))  # (nz-1, ni)
    drift = (
        Hi_interf
        - 1.0 / Hpi[:, None]
        + alpha[None, :] / Ti[:, None] * delta_Ti[:, None] / dzi[:, None]
    )
    return -Dzz * drift  # (nz-1, ni)


def update_phi_esc_jax(
    y: jnp.ndarray,
    g: jnp.ndarray,
    Hp: jnp.ndarray,
    top_flux_in: jnp.ndarray,
    st: AtmRefreshStatic,
) -> jnp.ndarray:
    """Diffusion-limited escape flux at TOA for each species in `diff_esc_idx`,
    floored at `-max_flux`; other species pass through unchanged.

    Inputs: y (nz, ni) number densities, g / Hp (nz,) — only the top layer is
    used, top_flux_in (ni,). Returns (ni,), the top boundary flux with the
    escape species overwritten.
    """
    diff_esc_idx = st.diff_esc_idx
    Tco_top = st.Tco[-1]
    kb = st.kb
    Navo = st.Navo
    max_flux = st.max_flux

    y_esc = y[-1, diff_esc_idx]
    Dzz_esc = st.Dzz_top[diff_esc_idx]
    ms_esc = st.ms[diff_esc_idx]
    g_top = g[-1]
    Hp_top = Hp[-1]

    flux = -Dzz_esc * y_esc * (1.0 / Hp_top - ms_esc * g_top / (Navo * kb * Tco_top))
    flux = jnp.maximum(flux, -max_flux)

    return top_flux_in.at[diff_esc_idx].set(flux)
