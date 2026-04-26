"""Pure-JAX condensation kernels (Phase 10.4).

JAX ports of `op.conden` (`VULCAN-master/op.py:1112-1301`),
`op.h2o_conden_evap_relax` (`op.py:1334-1370`), and
`op.nh3_conden_evap_relax` (`op.py:1372-1424`).

These run inside the JAX `OuterLoop` runner, gated by `lax.cond` on a
`do_conden` flag that fires when `t >= start_conden_time` and the
fix-species switch has not yet flipped, mirroring the upstream NumPy gate
at `op.py:856`.

`update_conden_rates` recomputes the condensation/evaporation rate for
each condensation reaction in the network. All formulas in `op.conden`
share the same continuum-regime shape

    rate[r, z] = Dg[r, z] * (m / (rho_p * r_p^2))[r] *
                 (y[z, sp[r]] - sat_n[r, z])

so per-reaction constants are baked into the closed-over `CondenStatic`
once at OuterLoop init. The H2O / NH3 `use_relax` short-circuit
(`op.py:1124-1126`, `1156-1158`) is implemented by zeroing the
corresponding `coeff_per_re` entries — those reactions then write zeros
into k_arr unconditionally, matching the upstream `var.k[re] = 0`.

`apply_h2o_relax_jax` / `apply_nh3_relax_jax` run the implicit-Euler
cold-trap relaxation: condense where `tau > 0` (i.e. `y > sat`), evaporate
where `tau < 0`. NH3 also clamps the condensation region to layers at or
below `conden_top` (the layer of minimum saturation mixing ratio); that
index is static and baked into the closure.

For HD189 (`use_condense=False`) none of this is built — the OuterLoop
gates conden_branch construction on `vulcan_cfg.use_condense`.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax.numpy as jnp


class CondenStatic(NamedTuple):
    """Closed-over static inputs to the conden kernels.

    All fields are constant during integration: T-P profile, particle
    sizes/densities, saturation pressures, and the species index tables.

    Layout of the conden reaction table:
      - `conden_re_idx[r]` is the k_arr row of the forward (condensation)
        reaction `r`. The reverse (evaporation) reaction lives at
        `conden_re_idx[r] + 1` (matches the network parser's i, i+1 layout
        in `op.py:227-228`).
      - `conden_sp_idx[r]` is the species index of the gas-phase reactant.
      - `Dg_per_re[r, :]` is the per-layer molecular diffusion coefficient
        for that species, with `atm.Dzz[0, :]` reused at the bottom layer
        (matches `np.insert(..., 0, ...)` at `op.py:1138`, `1167`, etc.).
      - `sat_n_per_re[r, :]` is the saturation number density
        `atm.sat_p[sp]/kb/Tco`, multiplied by `vulcan_cfg.humidity` for H2O
        only (op.py:1132 vs others).
      - `coeff_per_re[r]` is `m / (rho_p * r_p**2)` baked once. For
        reactions whose species is in `vulcan_cfg.use_relax`, this is set
        to 0 so `update_conden_rates` writes zero rates (matches the
        `var.k[re] = 0; var.k[re+1] = 0` short-circuit at op.py:1125-1126).
    """
    conden_re_idx:   jnp.ndarray   # (n_conden_re,)        int32
    conden_sp_idx:   jnp.ndarray   # (n_conden_re,)        int32
    Dg_per_re:       jnp.ndarray   # (n_conden_re, nz)     float64
    sat_n_per_re:    jnp.ndarray   # (n_conden_re, nz)     float64
    coeff_per_re:    jnp.ndarray   # (n_conden_re,)        float64

    # Optional H2O relax block. `h2o_active` is a Python bool selected
    # statically from vulcan_cfg.use_relax; when False, all H2O fields
    # below are placeholders and `apply_h2o_relax_jax` is a no-op.
    h2o_active:        bool
    h2o_idx:           int          # species index of 'H2O'
    h2o_l_s_idx:       int          # species index of 'H2O_l_s'
    h2o_Dg:            jnp.ndarray  # (nz,)
    h2o_sat:           jnp.ndarray  # (nz,)  sat_p['H2O']/kb/Tco * humidity
    h2o_m_over_rho_r2: float        # 18/Navo / (rho_p * r_p**2)

    # Optional NH3 relax block (same layout as H2O, plus the static
    # `conden_top` index = argmin(sat_mix['NH3']) used to clamp the
    # condensation region at op.py:1389).
    nh3_active:        bool
    nh3_idx:           int
    nh3_l_s_idx:       int
    nh3_Dg:            jnp.ndarray  # (nz,)
    nh3_sat:           jnp.ndarray  # (nz,)
    nh3_m_over_rho_r2: float
    nh3_conden_top:    int          # static — argmin(sat_mix['NH3'])

    # Static auxiliary: gas_indx_mask of shape (ni,) for the post-relax
    # `var.y = ymix * sum(y[:, gas_indx])` projection at op.py:1368, 1422.
    n_0:             jnp.ndarray    # (nz,)  total number density
    gas_indx_mask:   jnp.ndarray    # (ni,)  bool


def update_conden_rates(k_arr: jnp.ndarray, y: jnp.ndarray,
                        st: CondenStatic) -> jnp.ndarray:
    """JAX port of `op.conden` (op.py:1112-1301).

    For each reaction r in the conden table:

        rate[r, z]   = Dg[r, z] * coeff[r] * (y[z, sp[r]] - sat_n[r, z])
        k_arr[re]    = max( rate, 0)        # condensation
        k_arr[re+1]  = max(-rate, 0)        # evaporation (= |min(rate, 0)|)

    Reactions whose `coeff_per_re` is 0 (use_relax short-circuit) write
    pure zeros into both k_arr rows.

    Args:
        k_arr: (nr+1, nz) reaction-rate table.
        y:     (nz, ni) current species number densities.
        st:    closed-over `CondenStatic`.

    Returns:
        Updated `k_arr` with the n_conden_re forward + n_conden_re reverse
        rows overwritten. Other rows untouched.
    """
    # y_at_sp[r, z] = y[z, conden_sp_idx[r]] — gather the gas-phase species
    # number density for each conden reaction.
    y_at_sp = y[:, st.conden_sp_idx].T                  # (n_conden_re, nz)
    rate = (st.Dg_per_re
            * st.coeff_per_re[:, None]
            * (y_at_sp - st.sat_n_per_re))              # (n_conden_re, nz)

    k_pos = jnp.maximum(rate, 0.0)                       # condensation
    k_neg = jnp.maximum(-rate, 0.0)                      # evaporation

    k_arr_new = k_arr.at[st.conden_re_idx].set(k_pos)
    k_arr_new = k_arr_new.at[st.conden_re_idx + 1].set(k_neg)
    return k_arr_new


def apply_h2o_relax_jax(y: jnp.ndarray, ymix: jnp.ndarray, dt: jnp.ndarray,
                        st: CondenStatic) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX port of `op.h2o_conden_evap_relax` (op.py:1334-1370).

    Implicit-Euler relaxation of H2O toward saturation, with mass moved
    into / out of H2O_l_s. Conditional updates from the NumPy code's
    `np.where(...)` index sets become `jnp.where` masks here:

        tau = 1 / (Dg * m/rho_p/r_p^2 * (y[H2O] - sat))
        condense where tau > 0:  y_conden = (ymix + dt/tau * sat_mix) / (1 + dt/tau)
                                 ymix[H2O_l_s] += ymix[H2O] - y_conden
                                 ymix[H2O]     = y_conden
        evaporate where tau < 0: ice_loss = (y[H2O] - sat) * dt/tau    (negative)
                                 ice_loss = min(y[H2O_l_s], ice_loss)  (cap)
                                 ymix[H2O]     += ice_loss / n_0
                                 ymix[H2O_l_s] -= ice_loss / n_0

    Then `y = ymix * sum(y_old[:, gas_indx])` (op.py:1368) projects ymix
    back onto the gas-only total. Note the `y_old` — the sum uses the
    *pre-relax* y, not the post-relax one.

    Args:
        y:    (nz, ni) pre-relax y. Used for the gas-sum projection AND
              as input to `tau` and `ice_loss`.
        ymix: (nz, ni) pre-relax ymix. Mutated to add/remove H2O and H2O_l_s.
        dt:   ()     step size (matches `var.dt` in op.py:1316).
        st:   `CondenStatic` (uses h2o_*).

    Returns:
        (y_new, ymix_new) — both (nz, ni). When `h2o_active=False`,
        passes through unchanged.
    """
    if not st.h2o_active:
        return y, ymix

    h2o = st.h2o_idx
    h2o_l_s = st.h2o_l_s_idx

    # tau as in op.py:1343. The denominator can vanish in saturated cells
    # (y[H2O] == sat); guard with a tiny floor so tau is finite. Cells
    # where this happens fall outside both `conden_mask` and `evap_mask`,
    # so the guarded value is never used.
    denom = st.h2o_Dg * st.h2o_m_over_rho_r2 * (y[:, h2o] - st.h2o_sat)
    denom_safe = jnp.where(jnp.abs(denom) < 1e-300, 1e-300, denom)
    tau = 1.0 / denom_safe

    sat_mix = st.h2o_sat / st.n_0

    y_conden = (ymix[:, h2o] + dt / tau * sat_mix) / (1.0 + dt / tau)
    ice_loss = (y[:, h2o] - st.h2o_sat) * dt / tau
    ice_loss = jnp.minimum(y[:, h2o_l_s], ice_loss)

    conden_mask = tau > 0     # condensation (op.py:1344)
    evap_mask = tau < 0       # evaporation (op.py:1345)

    delta_h2o_conden = jnp.where(conden_mask, ymix[:, h2o] - y_conden, 0.0)
    delta_h2o_evap = jnp.where(evap_mask, ice_loss / st.n_0, 0.0)

    ymix_new = (ymix.at[:, h2o_l_s].add(delta_h2o_conden)
                    .at[:, h2o].add(-delta_h2o_conden)
                    .at[:, h2o].add(delta_h2o_evap)
                    .at[:, h2o_l_s].add(-delta_h2o_evap))

    # Project ymix back onto the gas-only total (op.py:1368).
    ysum = jnp.sum(jnp.where(st.gas_indx_mask[None, :], y, 0.0),
                   axis=1, keepdims=True)
    y_new = ymix_new * ysum
    return y_new, ymix_new


def apply_nh3_relax_jax(y: jnp.ndarray, ymix: jnp.ndarray, dt: jnp.ndarray,
                        st: CondenStatic) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX port of `op.nh3_conden_evap_relax` (op.py:1372-1424).

    Same shape as `apply_h2o_relax_jax` with two differences:
      1. No humidity factor (`sat = sat_p['NH3']/kb/Tco`, no scaling).
      2. Condensation is clamped to layers `i <= conden_top`, where
         `conden_top = argmin(sat_mix['NH3'])`. This is static (depends
         only on T-p profile and species saturation curve) and baked into
         `nh3_conden_top`.

    Also clips `ymix[NH3_l_s] >= 0` after the update (op.py:1420), since
    the non-clamped evap branch can drive it negative for cells where
    `i > conden_top` is *also* `evap_indx`.
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

    # Condensation clamped to layers <= conden_top (op.py:1389).
    layer_idx = jnp.arange(nz, dtype=jnp.int32)
    above_top_mask = layer_idx <= jnp.int32(st.nh3_conden_top)
    conden_mask = (tau > 0) & above_top_mask
    evap_mask = tau < 0   # no clamp on evap side (matches op.py:1390 commented-out)

    delta_nh3_conden = jnp.where(conden_mask, ymix[:, nh3] - y_conden, 0.0)
    delta_nh3_evap = jnp.where(evap_mask, ice_loss / st.n_0, 0.0)

    ymix_new = (ymix.at[:, nh3_l_s].add(delta_nh3_conden)
                    .at[:, nh3].add(-delta_nh3_conden)
                    .at[:, nh3].add(delta_nh3_evap)
                    .at[:, nh3_l_s].add(-delta_nh3_evap))

    # Clip NH3_l_s >= 0 (op.py:1420).
    ymix_new = ymix_new.at[:, nh3_l_s].set(
        jnp.maximum(ymix_new[:, nh3_l_s], 0.0)
    )

    # Project ymix back onto the gas-only total (op.py:1422).
    ysum = jnp.sum(jnp.where(st.gas_indx_mask[None, :], y, 0.0),
                   axis=1, keepdims=True)
    y_new = ymix_new * ysum
    return y_new, ymix_new
