"""Pure-JAX condensation kernels and the on-graph condensation-profile builder.

`update_conden_rates` recomputes the condensation/evaporation rate for
each conden reaction:

    rate[r, z] = Dg[r, z] * (m / (rho_p * r_p^2))[r] * (y[z, sp[r]] - sat_n[r, z])

`apply_h2o_relax_jax` / `apply_nh3_relax_jax` run an implicit-Euler
cold-trap relaxation toward saturation. NH3 additionally clamps the
condensation region to layers at or below the `conden_top` index.

`make_conden_spec` (host, once per config) + `build_conden_profile`
(pure JAX, per T-P realization) split the condensation state into the
temperature-INdependent metadata (`CondenSpec`: species identity,
particle mass/radius/density coefficients, relax/fix flags) and the
temperature/structure-DEPENDENT arrays (`CondenProfile`: saturation
number densities, growth/diffusion terms, the NH3 cold-trap index, and
the fix-species saturation mixing ratios). `build_conden_profile` is
jit/vmap/jvp-compatible, so a caller that evaluates a live T(P) on the
JAX graph (the retrieval's per-proposal rebuild) can regenerate every
frozen `ProfileVars c_*` quantity from the same live temperature and
splice it into the runner carry. `OuterLoop._build_conden_static`
delegates here too, so the host setup path and the on-graph path share
one implementation.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from .phy_const import Navo, kb

# Numerical floor for `/max(|x|, .)`-style denominators (mirrors the same
# named constant in outer_loop.py). Not a tuning knob — just below-which-is-zero.
_UNDERFLOW_DENOM = 1e-300

# Gas-phase condensates with a fully-ported runtime kinetics path — exactly
# VULCAN-master's op.conden branch set. `atm_setup._SUPPORTED_CONDENSABLES`
# additionally lists H2S, which has saturation data only (capping/cold-trap),
# no conden kinetics — in master too (master silently leaves an unknown
# condensate's rate at zero; VULCAN-JAX raises instead). runtime_validation
# checks condense_sp against these sets upfront; _build_conden_static keeps
# its NotImplementedError as defense-in-depth.
SUPPORTED_CONDEN_KINETICS: tuple[str, ...] = (
    "H2O",
    "NH3",
    "H2SO4",
    "S2",
    "S4",
    "S8",
    "C",
)

# Per-formula physical constants, baked literally from op.conden
# (op.py:1128-1291). Mass values are kept verbatim including the known
# oddities (S2 uses 45.019 g/mol per op.py:1244; S8 uses 360.152 per
# op.py:1290).
GAS_MASS_G_PER_MOL: dict[str, float] = {
    "H2O": 18.0,
    "NH3": 17.0,
    "H2SO4": 98.022,
    "S2": 45.019,
    "S4": 32.06 * 4,
    "S8": 360.152,
    "C": 12.011,
}
GAS_TO_CONDENSATE: dict[str, str] = {
    "H2O": "H2O_l_s",
    "NH3": "NH3_l_s",
    "H2SO4": "H2SO4_l",
    "S2": "S2_l_s",
    "S4": "S4_l_s",
    "S8": "S8_l_s",
    "C": "C_s",
}


class CondenSpec(NamedTuple):
    """Static condensation metadata — everything temperature-INdependent.

    Species identity, k_arr row indices, and the per-reaction coefficient
    `m / (rho_p * r_p**2)` (relax-shorted H2O/NH3 rows get 0.0, matching
    the `var.k[re] = 0` short-circuits at op.py:1124-1126 / 1156-1158).
    Built once per config on the host by `make_conden_spec`; consumed by
    `build_conden_profile` for every T-P realization.
    """

    gas_names: Tuple[str, ...]  # active conden reactions' gas species
    conden_re_idx: Tuple[int, ...]  # k_arr row of each forward reaction
    conden_sp_idx: Tuple[int, ...]  # species column of each gas species
    coeff_per_re: Tuple[float, ...]  # m/(rho_p r_p^2); 0.0 for relax rows
    humidity: float  # relative humidity (H2O saturation only)

    h2o_active: bool  # H2O relax kernel enabled (H2O in use_relax)
    h2o_idx: int
    h2o_l_s_idx: int
    h2o_m_over_rho_r2: float

    nh3_active: bool  # NH3 relax kernel enabled (NH3 in use_relax)
    nh3_idx: int
    nh3_l_s_idx: int
    nh3_m_over_rho_r2: float

    # cfg.fix_species names, in config order; species with saturation data
    # (members of `sat_names`) get a live sat-mix row, others a zero row —
    # mirroring `_Statics.fix_species_sat_mix`'s `sp in atm.sat_mix` gate.
    fix_names: Tuple[str, ...]
    sat_names: Tuple[str, ...]  # cfg.condense_sp (species with sat data)


class CondenProfile(NamedTuple):
    """Dynamic condensation arrays for one T-P/structure realization.

    All fields are JAX arrays computed on-graph by `build_conden_profile`;
    they are exactly the temperature/structure-dependent quantities the
    runner reads from the `ProfileVars` carry (`c_*` fields plus
    `fix_species_sat_mix`).
    """

    Dg_per_re: jnp.ndarray  # (n_conden_re, nz)
    sat_n_per_re: jnp.ndarray  # (n_conden_re, nz)
    h2o_Dg: jnp.ndarray  # (nz,)
    h2o_sat: jnp.ndarray  # (nz,)
    nh3_Dg: jnp.ndarray  # (nz,)
    nh3_sat: jnp.ndarray  # (nz,)
    nh3_conden_top: jnp.ndarray  # () int32 — argmin(sat_mix['NH3'])
    fix_species_sat_mix: jnp.ndarray  # (n_fix, nz)


def make_conden_spec(cfg, var, atm, species_idx) -> CondenSpec:
    """Extract the static condensation metadata from a completed setup.

    Walks `var.conden_re_list` exactly like the legacy packer: a reaction
    contributes a row only when its gas species is in `cfg.condense_sp`;
    an active-but-unported formula raises. H2O/NH3 relax blocks are
    populated when the species is in `cfg.use_relax` (their kinetics rows
    then get coeff 0.0). Host-side, cheap, no JAX arrays.
    """
    relax_set = set(getattr(cfg, "use_relax", []) or [])
    gas_names: list[str] = []
    re_idx: list[int] = []
    sp_idx: list[int] = []
    coeffs: list[float] = []
    for re in var.conden_re_list:
        rf = var.Rf[re]  # 'H2O -> H2O_l_s', etc.
        gas_sp = rf.split(" -> ")[0].strip()
        if gas_sp not in cfg.condense_sp:
            # Reaction is in the network but not active in this run.
            # Match op.conden's behavior: leaves k untouched (=0).
            continue
        if gas_sp not in GAS_MASS_G_PER_MOL:
            raise NotImplementedError(f"conden formula {rf!r} not yet ported to JAX")
        condensate = GAS_TO_CONDENSATE[gas_sp]
        m = GAS_MASS_G_PER_MOL[gas_sp] / Navo
        coeff = m / (float(atm.rho_p[condensate]) * float(atm.r_p[condensate]) ** 2)
        # use_relax short-circuit exists only on upstream H2O/NH3
        # branches (op.py:1124-1126, 1156-1158). Other condensates,
        # including H2SO4, still use their condensation-rate rows.
        if gas_sp in {"H2O", "NH3"} and gas_sp in relax_set:
            coeff = 0.0
        gas_names.append(gas_sp)
        re_idx.append(int(re))
        sp_idx.append(int(species_idx[gas_sp]))
        coeffs.append(coeff)

    h2o_active = "H2O" in relax_set and "H2O" in species_idx
    if h2o_active:
        h2o_idx = int(species_idx["H2O"])
        h2o_l_s_idx = int(species_idx["H2O_l_s"])
        h2o_m_over_rho_r2 = (18.0 / Navo) / (
            float(atm.rho_p["H2O_l_s"]) * float(atm.r_p["H2O_l_s"]) ** 2
        )
    else:
        h2o_idx = h2o_l_s_idx = 0
        h2o_m_over_rho_r2 = 0.0

    nh3_active = "NH3" in relax_set and "NH3" in species_idx
    if nh3_active:
        nh3_idx = int(species_idx["NH3"])
        nh3_l_s_idx = int(species_idx["NH3_l_s"])
        nh3_m_over_rho_r2 = (17.0 / Navo) / (
            float(atm.rho_p["NH3_l_s"]) * float(atm.r_p["NH3_l_s"]) ** 2
        )
    else:
        nh3_idx = nh3_l_s_idx = 0
        nh3_m_over_rho_r2 = 0.0

    fix_names = tuple(getattr(cfg, "fix_species", []) or [])
    return CondenSpec(
        gas_names=tuple(gas_names),
        conden_re_idx=tuple(re_idx),
        conden_sp_idx=tuple(sp_idx),
        coeff_per_re=tuple(coeffs),
        humidity=float(getattr(cfg, "humidity", 1.0)),
        h2o_active=h2o_active,
        h2o_idx=h2o_idx,
        h2o_l_s_idx=h2o_l_s_idx,
        h2o_m_over_rho_r2=h2o_m_over_rho_r2,
        nh3_active=nh3_active,
        nh3_idx=nh3_idx,
        nh3_l_s_idx=nh3_l_s_idx,
        nh3_m_over_rho_r2=nh3_m_over_rho_r2,
        fix_names=fix_names,
        sat_names=tuple(cfg.condense_sp),
    )


def build_conden_profile(
    spec: CondenSpec,
    Tco: jnp.ndarray,
    pco: jnp.ndarray,
    n_0: jnp.ndarray,
    Dzz: jnp.ndarray,
) -> CondenProfile:
    """Rebuild every T/structure-dependent condensation array on-graph.

    Parameters: `Tco`/`pco`/`n_0` are (nz,) layer temperature, pressure
    (dyne/cm^2), and total number density; `Dzz` is the (nz-1, ni)
    molecular-diffusion coefficient (interface-centered). All may be
    traced — the function is jit/vmap/jvp-compatible: species identity
    comes from the static `spec` (Python-level, unrolled at trace time),
    and the only discrete output is the NH3 cold-trap `argmin` index
    (integer-valued, carries no tangent).

    Formulas match op.conden / `_apply_condense` exactly:
      sat_n   = sat_p(T)/kB/T          (× humidity for H2O)
      Dg      = Dzz[:, sp] with the bottom interface value repeated
      sat_mix = min(1, sat_p(T)/p)     (× humidity for H2O, after the clip)
      NH3 cold trap = argmin(sat_n_NH3 / n_0)
    """
    # Import here (not module top) to avoid a cycle: atm_setup imports
    # vulcan_cfg at module load, which must stay independent of conden.
    from .atm_setup import sat_p_jax

    Tco = jnp.asarray(Tco, dtype=jnp.float64)
    pco = jnp.asarray(pco, dtype=jnp.float64)
    n_0 = jnp.asarray(n_0, dtype=jnp.float64)
    Dzz = jnp.asarray(Dzz, dtype=jnp.float64)
    nz = Tco.shape[0]
    if Dzz.ndim != 2 or Dzz.shape[0] != nz - 1:
        raise ValueError(
            f"Dzz shape {Dzz.shape} inconsistent with nz={nz}: expected (nz-1, ni)"
        )

    def _Dg_col(sp_col: int) -> jnp.ndarray:
        # Dg = atm.Dzz[:, sp] with [0] reused at the bottom (op.py:1138).
        col = Dzz[:, sp_col]
        return jnp.concatenate([col[:1], col])

    def _sat_n(name: str) -> jnp.ndarray:
        sat_n = sat_p_jax(name, Tco) / kb / Tco
        if name == "H2O":
            sat_n = sat_n * spec.humidity
        return sat_n

    if spec.gas_names:
        Dg_per_re = jnp.stack([_Dg_col(c) for c in spec.conden_sp_idx])
        sat_n_per_re = jnp.stack([_sat_n(name) for name in spec.gas_names])
    else:
        Dg_per_re = jnp.zeros((0, nz), dtype=jnp.float64)
        sat_n_per_re = jnp.zeros((0, nz), dtype=jnp.float64)

    zz = jnp.zeros((nz,), dtype=jnp.float64)
    h2o_Dg = _Dg_col(spec.h2o_idx) if spec.h2o_active else zz
    h2o_sat = _sat_n("H2O") if spec.h2o_active else zz
    nh3_Dg = _Dg_col(spec.nh3_idx) if spec.nh3_active else zz
    nh3_sat = _sat_n("NH3") if spec.nh3_active else zz
    if spec.nh3_active:
        # conden_top = argmin(sat_mix['NH3']) = argmin(sat_n / n_0);
        # discrete by nature — moves layer-by-layer as T changes.
        nh3_conden_top = jnp.argmin(nh3_sat / n_0).astype(jnp.int32)
    else:
        nh3_conden_top = jnp.int32(0)

    if spec.fix_names:
        rows = []
        for name in spec.fix_names:
            if name in spec.sat_names:
                # `_apply_condense` order: clip to 1 first, THEN humidity.
                sm = jnp.minimum(1.0, sat_p_jax(name, Tco) / pco)
                if name == "H2O":
                    sm = sm * spec.humidity
                rows.append(sm)
            else:
                rows.append(zz)
        fix_species_sat_mix = jnp.stack(rows)
    else:
        fix_species_sat_mix = jnp.zeros((0, nz), dtype=jnp.float64)

    return CondenProfile(
        Dg_per_re=Dg_per_re,
        sat_n_per_re=sat_n_per_re,
        h2o_Dg=h2o_Dg,
        h2o_sat=h2o_sat,
        nh3_Dg=nh3_Dg,
        nh3_sat=nh3_sat,
        nh3_conden_top=nh3_conden_top,
        fix_species_sat_mix=fix_species_sat_mix,
    )


class CondenStatic(NamedTuple):
    """Closed-over static inputs to the conden kernels.

    `conden_re_idx[r]` is the k_arr row of the forward (condensation)
    reaction; the reverse (evaporation) sits at `conden_re_idx[r] + 1`.
    `coeff_per_re[r] = m / (rho_p * r_p**2)`; reactions short-circuited
    via `vulcan_cfg.use_relax` get coeff=0 so the kernel writes zeros.
    """

    conden_re_idx: jnp.ndarray  # (n_conden_re,)        int32
    conden_sp_idx: jnp.ndarray  # (n_conden_re,)        int32
    Dg_per_re: jnp.ndarray  # (n_conden_re, nz)     float64
    sat_n_per_re: jnp.ndarray  # (n_conden_re, nz)     float64
    coeff_per_re: jnp.ndarray  # (n_conden_re,)        float64

    # `h2o_active` / `nh3_active` are Python bools selected statically from
    # vulcan_cfg.use_relax; when False, the *_relax_jax helpers no-op.
    h2o_active: bool
    h2o_idx: int  # species index of 'H2O'
    h2o_l_s_idx: int  # species index of 'H2O_l_s'
    h2o_Dg: jnp.ndarray  # (nz,)
    h2o_sat: jnp.ndarray  # (nz,)  sat_p['H2O']/kb/Tco * humidity
    h2o_m_over_rho_r2: float  # 18/Navo / (rho_p * r_p**2)

    nh3_active: bool
    nh3_idx: int
    nh3_l_s_idx: int
    nh3_Dg: jnp.ndarray  # (nz,)
    nh3_sat: jnp.ndarray  # (nz,)
    nh3_m_over_rho_r2: float
    # argmin(sat_mix['NH3']). Python int when closure-baked (single-profile);
    # 0-d int32 array when spliced per lane from ProfileVars in the batched
    # runner — the kernel only compares it against jnp.arange, so both work.
    nh3_conden_top: int | jnp.ndarray

    n_0: jnp.ndarray  # (nz,)  total number density
    gas_indx_mask: jnp.ndarray  # (ni,)  bool — gas-only species mask


def update_conden_rates(
    k_arr: jnp.ndarray, y: jnp.ndarray, st: CondenStatic
) -> jnp.ndarray:
    """Recompute condensation/evaporation rates and overwrite the conden rows
    of `k_arr`. Conden goes to `re`, evap to `re+1`; rows for use_relax
    species are zeroed."""
    y_at_sp = y[:, st.conden_sp_idx].T  # (n_conden_re, nz)
    rate = st.Dg_per_re * st.coeff_per_re[:, None] * (y_at_sp - st.sat_n_per_re)

    k_pos = jnp.maximum(rate, 0.0)
    k_neg = jnp.maximum(-rate, 0.0)

    k_arr_new = k_arr.at[st.conden_re_idx].set(k_pos)
    return k_arr_new.at[st.conden_re_idx + 1].set(k_neg)


def apply_h2o_relax_jax(
    y: jnp.ndarray, ymix: jnp.ndarray, dt: jnp.ndarray, st: CondenStatic
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    # there tau becomes ~+1e300, so dt/tau ~ 0 makes y_conden ~ ymix and the
    # condensation delta is effectively zero.
    denom = st.h2o_Dg * st.h2o_m_over_rho_r2 * (y[:, h2o] - st.h2o_sat)
    denom_safe = jnp.where(jnp.abs(denom) < _UNDERFLOW_DENOM, _UNDERFLOW_DENOM, denom)
    tau = 1.0 / denom_safe

    sat_mix = st.h2o_sat / st.n_0

    y_conden = (ymix[:, h2o] + dt / tau * sat_mix) / (1.0 + dt / tau)
    ice_loss = (y[:, h2o] - st.h2o_sat) * dt / tau
    ice_loss = jnp.minimum(y[:, h2o_l_s], ice_loss)

    conden_mask = tau > 0
    evap_mask = tau < 0

    delta_h2o_conden = jnp.where(conden_mask, ymix[:, h2o] - y_conden, 0.0)
    delta_h2o_evap = jnp.where(evap_mask, ice_loss / st.n_0, 0.0)

    ymix_new = (
        ymix.at[:, h2o_l_s]
        .add(delta_h2o_conden)
        .at[:, h2o]
        .add(-delta_h2o_conden)
        .at[:, h2o]
        .add(delta_h2o_evap)
        .at[:, h2o_l_s]
        .add(-delta_h2o_evap)
    )

    ysum = jnp.sum(jnp.where(st.gas_indx_mask[None, :], y, 0.0), axis=1, keepdims=True)
    return ymix_new * ysum, ymix_new


def apply_nh3_relax_jax(
    y: jnp.ndarray, ymix: jnp.ndarray, dt: jnp.ndarray, st: CondenStatic
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    denom_safe = jnp.where(jnp.abs(denom) < _UNDERFLOW_DENOM, _UNDERFLOW_DENOM, denom)
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

    ymix_new = (
        ymix.at[:, nh3_l_s]
        .add(delta_nh3_conden)
        .at[:, nh3]
        .add(-delta_nh3_conden)
        .at[:, nh3]
        .add(delta_nh3_evap)
        .at[:, nh3_l_s]
        .add(-delta_nh3_evap)
    )

    ymix_new = ymix_new.at[:, nh3_l_s].set(jnp.maximum(ymix_new[:, nh3_l_s], 0.0))

    ysum = jnp.sum(jnp.where(st.gas_indx_mask[None, :], y, 0.0), axis=1, keepdims=True)
    return ymix_new * ysum, ymix_new
