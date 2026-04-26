"""Pure-JAX atmosphere-refresh kernels (Phase 10.3).

JAX ports of `op.update_mu_dz` (`VULCAN-master/op.py:944-984`) and
`op.update_phi_esc` (`op.py:986-999`). These run inside the JAX
`OuterLoop` runner, gated by `lax.cond` on a `do_atm_refresh` flag fired
every `update_frq` accepted steps — the same cadence as the upstream
NumPy path.

`update_mu_dz` recomputes mean molecular weight, scale height, and the
hydrostatic z-grid from the current `ymix`. The hydrostatic loop is a
true sequential dependency (`zco[i+1] = zco[i] + dz[i]`), so it lives in
two `jax.lax.scan` calls — one upward from `pref_indx` (the layer closest
to the reference pressure, ≈1 bar for gas giants) and one downward from
`pref_indx-1` to 0. With `pref_indx == 0` (rocky-planet path) the
downward scan is a length-zero scan that no-ops.

`update_phi_esc` updates the diffusion-limited escape flux at TOA for
each species in `vulcan_cfg.diff_esc`. For HD189 with `use_topflux=False`
this only affects diagnostics, but we update it anyway to stay
bit-faithful to VULCAN's recorded state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class AtmRefreshStatic(NamedTuple):
    """Closed-over static inputs to the atm-refresh kernels.

    Every field is constant during integration: T-P profile, planetary
    radius / surface gravity, species masses / molar masses, the static
    `pref_indx` boundary, etc.
    """
    Tco:           jnp.ndarray   # (nz,)        layer-center temperature (K)
    pico:          jnp.ndarray   # (nz+1,)      interface pressure (dyne/cm^2)
    mol_mass:      jnp.ndarray   # (ni,)        molar mass per species (g/mol)
    ms:            jnp.ndarray   # (ni,)        per-particle mass (g)
    Dzz_top:       jnp.ndarray   # (ni,)        molecular diffusion at the top interface
    diff_esc_idx:  jnp.ndarray   # (n_esc,) int32  species indices for diffusion-limited escape
    pref_indx:     int           # static       reference layer index (zco==zco_pref here)
    zco_pref:      float         # static       z at the reference layer (0.0 for gas giants)
    gs:            float         # surface gravity at zco_pref (cm/s^2)
    Rp:            float         # planetary radius (cm)
    kb:            float
    Navo:          float
    max_flux:      float         # cap on |top_flux| (vulcan_cfg.max_flux)
    nz:            int           # static
    ni:            int           # static


def update_mu_dz_jax(ymix: jnp.ndarray, st: AtmRefreshStatic):
    """JAX port of `op.update_mu_dz` (op.py:944-984).

    Returns the updated atmosphere geometry derived from the current ymix:

        mu  : (nz,)   mean molar mass per layer
        g   : (nz,)   gravity (gs at pref_indx, gs*(Rp/(Rp+zco))^2 elsewhere)
        Hp  : (nz,)   pressure scale height (kb*Tco / (mu/Navo * g))
        dz  : (nz,)   layer thickness (Hp * log(pico[i]/pico[i+1]))
        zco : (nz+1,) interface heights (cumsum from zco_pref)
        dzi : (nz-1,) interface dz (0.5*(dz[i]+dz[i-1]))
        Hpi : (nz-1,) interface scale height (0.5*(Hp[i]+Hp[i+1]))

    The hydrostatic loop is split into a forward scan from `pref_indx`
    upward and a backward scan from `pref_indx-1` downward — both true
    sequential dependencies on the running `zco`. `Ti` is intentionally
    not returned because it depends only on the static `Tco` (op.py:980,
    `0.5*(Tco + roll(Tco,-1))[:-1]`) — capture it once at OuterLoop init.
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

    # mu vectorized over layers (op.build_atm.mean_mass:520-525).
    mu = jnp.einsum("zi,i->z", ymix, mol_mass)  # (nz,)

    # Forward scan: zco propagates upward from zco_pref; per-layer
    # outputs are (g, Hp, dz, zco_top) for layers pref_indx..nz-1.
    def fwd_step(zco_i, i):
        # At the reference layer, g uses the surface value directly
        # (op.py:957). Above it, g follows 1/r^2 from Rp+zco[i].
        is_pref = i == pref_indx
        g_i = jnp.where(is_pref, gs, gs * (Rp / (Rp + zco_i)) ** 2)
        Hp_i = kb * Tco[i] / (mu[i] / Navo * g_i)
        dz_i = Hp_i * jnp.log(pico[i] / pico[i + 1])
        zco_ip1 = zco_i + dz_i
        return zco_ip1, (g_i, Hp_i, dz_i, zco_ip1)

    fwd_indices = jnp.arange(pref_indx, nz, dtype=jnp.int32)
    _, (g_fwd, Hp_fwd, dz_fwd, zco_above) = jax.lax.scan(
        fwd_step, zco_pref, fwd_indices,
    )
    # zco_above has length nz-pref_indx, giving zco at indices pref_indx+1..nz.

    # Backward scan: zco propagates downward from zco_pref; per-layer
    # outputs are (g, Hp, dz, zco_bot) for layers pref_indx-1..0.
    # When pref_indx == 0 this scan has length 0 and produces empties.
    def bwd_step(zco_ip1, i):
        # g uses Rp+zco[i+1] (op.py:967; the layer is BELOW zco[i+1]).
        g_i = gs * (Rp / (Rp + zco_ip1)) ** 2
        Hp_i = kb * Tco[i] / (mu[i] / Navo * g_i)
        dz_i = Hp_i * jnp.log(pico[i] / pico[i + 1])
        zco_i = zco_ip1 - dz_i
        return zco_i, (g_i, Hp_i, dz_i, zco_i)

    bwd_indices = jnp.arange(pref_indx - 1, -1, -1, dtype=jnp.int32)
    _, (g_bwd_rev, Hp_bwd_rev, dz_bwd_rev, zco_below_rev) = jax.lax.scan(
        bwd_step, zco_pref, bwd_indices,
    )
    # The backward outputs are in DECREASING-i order (i = pref_indx-1, ..., 0).
    # Reverse to canonical i = 0, 1, ..., pref_indx-1 ordering before stacking.
    g_bwd = g_bwd_rev[::-1]
    Hp_bwd = Hp_bwd_rev[::-1]
    dz_bwd = dz_bwd_rev[::-1]
    zco_below = zco_below_rev[::-1]

    g = jnp.concatenate([g_bwd, g_fwd])      # (nz,)
    Hp = jnp.concatenate([Hp_bwd, Hp_fwd])   # (nz,)
    dz = jnp.concatenate([dz_bwd, dz_fwd])   # (nz,)
    zco = jnp.concatenate([zco_below,
                            zco_pref[None],
                            zco_above])      # (nz+1,)

    # dzi (op.py:974-975): 0.5*(dz + roll(dz,1))[1:] == 0.5*(dz[:-1] + dz[1:])
    dzi = 0.5 * (dz[:-1] + dz[1:])           # (nz-1,)

    # Hpi (op.py:981-982): 0.5*(Hp + roll(Hp,-1))[:-1] == 0.5*(Hp[:-1] + Hp[1:])
    Hpi = 0.5 * (Hp[:-1] + Hp[1:])           # (nz-1,)

    return mu, g, Hp, dz, zco, dzi, Hpi


def update_phi_esc_jax(y: jnp.ndarray, g: jnp.ndarray, Hp: jnp.ndarray,
                       top_flux_in: jnp.ndarray, st: AtmRefreshStatic) -> jnp.ndarray:
    """JAX port of `op.update_phi_esc` (op.py:986-999).

    For each species in `diff_esc_idx`, computes the diffusion-limited
    escape flux at the top interface:

        top_flux[sp] = -Dzz[-1, sp] * y[-1, sp] *
                       (1/Hp[-1] - ms[sp] * g[-1] / (Navo * kb * Tco[-1]))

    bounded below by `-max_flux`. Other species' top_flux entries are
    passed through unchanged from `top_flux_in`.
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

    flux = -Dzz_esc * y_esc * (
        1.0 / Hp_top - ms_esc * g_top / (Navo * kb * Tco_top)
    )
    flux = jnp.maximum(flux, -max_flux)

    return top_flux_in.at[diff_esc_idx].set(flux)
