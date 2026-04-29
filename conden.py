"""Pure-JAX condensation kernels.

`update_conden_rates` recomputes the condensation/evaporation rate for
each conden reaction:

    rate[r, z] = Dg[r, z] * (m / (rho_p * r_p^2))[r] * (y[z, sp[r]] - sat_n[r, z])

`apply_h2o_relax_jax` / `apply_nh3_relax_jax` run an implicit-Euler
cold-trap relaxation toward saturation. NH3 additionally clamps the
condensation region to layers at or below the static `conden_top` index.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax.numpy as jnp


class CondenStatic(NamedTuple):
    """Closed-over static inputs to the conden kernels.

    `conden_re_idx[r]` is the k_arr row of the forward (condensation)
    reaction; the reverse (evaporation) sits at `conden_re_idx[r] + 1`.
    `coeff_per_re[r] = m / (rho_p * r_p**2)`; reactions short-circuited
    via `vulcan_cfg.use_relax` get coeff=0 so the kernel writes zeros.
    """
    conden_re_idx:   jnp.ndarray   # (n_conden_re,)        int32
    conden_sp_idx:   jnp.ndarray   # (n_conden_re,)        int32
    Dg_per_re:       jnp.ndarray   # (n_conden_re, nz)     float64
    sat_n_per_re:    jnp.ndarray   # (n_conden_re, nz)     float64
    coeff_per_re:    jnp.ndarray   # (n_conden_re,)        float64

    # `h2o_active` / `nh3_active` are Python bools selected statically from
    # vulcan_cfg.use_relax; when False, the *_relax_jax helpers no-op.
    h2o_active:        bool
    h2o_idx:           int          # species index of 'H2O'
    h2o_l_s_idx:       int          # species index of 'H2O_l_s'
    h2o_Dg:            jnp.ndarray  # (nz,)
    h2o_sat:           jnp.ndarray  # (nz,)  sat_p['H2O']/kb/Tco * humidity
    h2o_m_over_rho_r2: float        # 18/Navo / (rho_p * r_p**2)

    nh3_active:        bool
    nh3_idx:           int
    nh3_l_s_idx:       int
    nh3_Dg:            jnp.ndarray  # (nz,)
    nh3_sat:           jnp.ndarray  # (nz,)
    nh3_m_over_rho_r2: float
    nh3_conden_top:    int          # static — argmin(sat_mix['NH3'])

    n_0:             jnp.ndarray    # (nz,)  total number density
    gas_indx_mask:   jnp.ndarray    # (ni,)  bool — gas-only species mask


def update_conden_rates(k_arr: jnp.ndarray, y: jnp.ndarray,
                        st: CondenStatic) -> jnp.ndarray:
    """Recompute condensation/evaporation rates and overwrite the conden rows
    of `k_arr`. Conden goes to `re`, evap to `re+1`; rows for use_relax
    species are zeroed."""
    y_at_sp = y[:, st.conden_sp_idx].T                  # (n_conden_re, nz)
    rate = (st.Dg_per_re
            * st.coeff_per_re[:, None]
            * (y_at_sp - st.sat_n_per_re))

    k_pos = jnp.maximum(rate, 0.0)
    k_neg = jnp.maximum(-rate, 0.0)

    k_arr_new = k_arr.at[st.conden_re_idx].set(k_pos)
    return k_arr_new.at[st.conden_re_idx + 1].set(k_neg)


def apply_h2o_relax_jax(y: jnp.ndarray, ymix: jnp.ndarray, dt: jnp.ndarray,
                        st: CondenStatic) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Implicit-Euler H2O cold-trap relaxation.

    Condense where `tau > 0` (y > sat), evaporate where `tau < 0`. Mass
    moves into / out of `H2O_l_s`. The final ymix → y projection uses the
    *pre-relax* gas-sum, intentionally; no-op when `h2o_active=False`.
    """
    if not st.h2o_active:
        return y, ymix

    h2o = st.h2o_idx
    h2o_l_s = st.h2o_l_s_idx

    # Tiny floor on the denominator avoids NaN in cells where y[H2O]==sat;
    # those cells fall outside both conden_mask and evap_mask anyway.
    denom = st.h2o_Dg * st.h2o_m_over_rho_r2 * (y[:, h2o] - st.h2o_sat)
    denom_safe = jnp.where(jnp.abs(denom) < 1e-300, 1e-300, denom)
    tau = 1.0 / denom_safe

    sat_mix = st.h2o_sat / st.n_0

    y_conden = (ymix[:, h2o] + dt / tau * sat_mix) / (1.0 + dt / tau)
    ice_loss = (y[:, h2o] - st.h2o_sat) * dt / tau
    ice_loss = jnp.minimum(y[:, h2o_l_s], ice_loss)

    conden_mask = tau > 0
    evap_mask = tau < 0

    delta_h2o_conden = jnp.where(conden_mask, ymix[:, h2o] - y_conden, 0.0)
    delta_h2o_evap = jnp.where(evap_mask, ice_loss / st.n_0, 0.0)

    ymix_new = (ymix.at[:, h2o_l_s].add(delta_h2o_conden)
                    .at[:, h2o].add(-delta_h2o_conden)
                    .at[:, h2o].add(delta_h2o_evap)
                    .at[:, h2o_l_s].add(-delta_h2o_evap))

    ysum = jnp.sum(jnp.where(st.gas_indx_mask[None, :], y, 0.0),
                   axis=1, keepdims=True)
    return ymix_new * ysum, ymix_new


def apply_nh3_relax_jax(y: jnp.ndarray, ymix: jnp.ndarray, dt: jnp.ndarray,
                        st: CondenStatic) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Implicit-Euler NH3 cold-trap relaxation.

    Differences from H2O: no humidity factor, and condensation is clamped
    to layers at or below `nh3_conden_top = argmin(sat_mix['NH3'])`.
    Clips `ymix[NH3_l_s] >= 0` post-update — the unclamped evap branch
    can drive it negative for layers above `conden_top`.
    """
    if not st.nh3_active:
        return y, ymix

    nh3 = st.nh3_idx
    nh3_l_s = st.nh3_l_s_idx
    nz = y.shape[0]

    denom = st.nh3_Dg * st.nh3_m_over_rho_r2 * (y[:, nh3] - st.nh3_sat)
    denom_safe = jnp.where(jnp.abs(denom) < 1e-300, 1e-300, denom)
    tau = 1.0 / denom_safe

    sat_mix = st.nh3_sat / st.n_0

    y_conden = (ymix[:, nh3] + dt / tau * sat_mix) / (1.0 + dt / tau)
    ice_loss = (y[:, nh3] - st.nh3_sat) * dt / tau
    ice_loss = jnp.minimum(y[:, nh3_l_s], ice_loss)

    # Condensation clamped to layers <= conden_top; evap is unclamped.
    layer_idx = jnp.arange(nz, dtype=jnp.int32)
    above_top_mask = layer_idx <= jnp.int32(st.nh3_conden_top)
    conden_mask = (tau > 0) & above_top_mask
    evap_mask = tau < 0

    delta_nh3_conden = jnp.where(conden_mask, ymix[:, nh3] - y_conden, 0.0)
    delta_nh3_evap = jnp.where(evap_mask, ice_loss / st.n_0, 0.0)

    ymix_new = (ymix.at[:, nh3_l_s].add(delta_nh3_conden)
                    .at[:, nh3].add(-delta_nh3_conden)
                    .at[:, nh3].add(delta_nh3_evap)
                    .at[:, nh3_l_s].add(-delta_nh3_evap))

    ymix_new = ymix_new.at[:, nh3_l_s].set(
        jnp.maximum(ymix_new[:, nh3_l_s], 0.0)
    )

    ysum = jnp.sum(jnp.where(st.gas_indx_mask[None, :], y, 0.0),
                   axis=1, keepdims=True)
    return ymix_new * ysum, ymix_new
