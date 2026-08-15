"""Differentiable on-graph atmosphere builder.

`atm_setup.py` computes every atmosphere-derived array (Tco, M/n_0, dz, Hp, g,
Dzz, vm, vs, ...) at pre-loop setup and freezes it to NumPy before the runner.
Most of that math is *already* pure JAX -- it is only `np.asarray`'d at the
boundary. This module re-expresses the same cascade as one differentiable
function, :func:`build_atm_static`, so gradients flow from the physical inputs
(T(P) profile / T_irr, surface gravity, planet radius, the pressure grid, eddy
and molecular diffusion, composition) all the way to the `AtmStatic` the Ros2
step consumes.

Usage -- pair it with forward-mode AD (`jax.lax.while_loop` supports `jvp`)::

    phys, spec = make_physical_inputs(cfg, var, atm, species_list)
    def loss_of_Tirr(T_irr):
        tp = phys_tp_params.at[1].set(T_irr)          # Heng+14 (T_int, T_irr, ...)
        Tco = analytical_TP_H14(phys.pco, tp, gs=gs, Pb=Pb)
        atm_static = build_atm_static(phys._replace(Tco=Tco), spec)
        return loss(runner(state, atm_static))
    dL_dTirr = jax.jvp(loss_of_Tirr, (T_irr,), (1.0,))[1]

`spec` is static configuration (species list, atm_base, toggles, the
discrete reference layer ``pref_indx``); differentiate w.r.t. the
:class:`PhysicalInputs` pytree only, closing over `spec`.

Differentiable here: pco (-> P_b/P_t via :func:`pco_from_endpoints`), Tco
(-> T_irr via `analytical_TP_H14`), ymix (mean molecular weight -> scale
height), Kzz (-> profile params via `atm_setup.kzz_profile_jax`), vz, gs, Rp.
Reachable downstream: M/n_0, mu, g, Hp, dz, dzi, Ti, Hpi, Dzz, Dzz_cen, vm, vs.

NOT differentiable here (by design):
  * FastChem equilibrium initial abundances (subprocess wall). Use the
    `const_lowT` initialiser for a differentiable elemental-abundance path, or
    pass `ymix` as a leaf.
  * Photolysis T-dependent cross-section re-interpolation
    (`photo_setup._bin_T_dependent`): cross-sections enter `PhotoStaticInputs`
    as injectable JAX arrays (so dL/d(cross-section) works), but their T-rebake
    is still host-side, so dL/dT *through* the cross-sections does not.
  * The thermal-diffusion factor `alpha` and the hydrostatic reference layer
    `pref_indx` are discrete/static lookups, held fixed (correct to first
    order in a small physical perturbation).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .atm_setup import (
    _Dzz_gen_for_base,
    _VISCOSITY_TABLE,
    _scan_down_mu_dz_g,
    _scan_up_mu_dz_g,
    compute_mean_mass,
    compute_pico,
    settling_coeff_array,
    settling_velocity_jax,
    surface_gravity,
)
from .jax_step import AtmStatic
from .atm_refresh import recompute_vm_jax
from .phy_const import Navo, kb

jax.config.update("jax_enable_x64", True)


class PhysicalInputs(NamedTuple):
    """Differentiable atmosphere inputs (the leaves AD flows through).

    Every field is a JAX array. `pco`/`Tco`/`ymix`/`Kzz`/`vz` are per-layer or
    per-interface profiles; `gs`/`Rp` are scalars. To differentiate w.r.t. a
    *parameterisation* (T_irr, K_deep, P_b), compose the matching on-graph
    front-end (`analytical_TP_H14`, `atm_setup.kzz_profile_jax`,
    :func:`pco_from_endpoints`) in front of the leaf it feeds.
    """

    pco: jnp.ndarray  # (nz,) cell-centre pressures, dyne/cm^2
    Tco: jnp.ndarray  # (nz,) temperature, K
    ymix: jnp.ndarray  # (nz, ni) mixing ratios (feeds mean molecular weight)
    Kzz: jnp.ndarray  # (nz-1,) eddy diffusion at interfaces
    vz: jnp.ndarray  # (nz-1,) vertical advection at interfaces
    gs: jnp.ndarray  # scalar surface gravity, cm/s^2
    Rp: jnp.ndarray  # scalar planet radius, cm


class AtmSpec(NamedTuple):
    """Static, non-differentiated configuration for :func:`build_atm_static`.

    Holds discrete/host-side choices (species identities, background gas,
    transport toggles, the hydrostatic reference layer) plus the per-species
    constant arrays. Close over it; do not trace through it.
    """

    nz: int
    ni: int
    pref_indx: int  # hydrostatic reference layer (g = gs); discrete, held fixed
    atm_base: str  # background gas -> Dzz fit + viscosity polynomial
    ms: jnp.ndarray  # (ni,) species molar masses, g/mol
    alpha: jnp.ndarray  # (ni,) thermal-diffusion factors (static lookup)
    gas_indx_mask: jnp.ndarray  # (ni,) bool, gas-phase species
    nongas_mask: jnp.ndarray  # (ni,) bool, condensed/non-gas species
    settle_coeff: jnp.ndarray  # (ni,) rho_p * r_p**2, zero for gas species
    top_flux: jnp.ndarray  # (ni,)
    bot_flux: jnp.ndarray  # (ni,)
    bot_vdep: jnp.ndarray  # (ni,)
    diff_esc_mask: jnp.ndarray  # (ni,) bool, species in cfg.diff_esc
    use_moldiff: bool
    use_vm_mol: bool
    use_settling: bool
    use_topflux: bool
    use_botflux: bool
    # (no use_condense: the on-graph builder never reads it. Condensation is a
    # per-step runtime process driven from the ProfileVars carry, not a
    # structural-cascade input, so the field sat here unread until it was
    # removed on 2026-08-14. Do not re-add it "for symmetry".)


def pco_from_endpoints(P_b, P_t, nz: int) -> jnp.ndarray:
    """Log-spaced pressure grid from the boundary pressures (dyne/cm^2).

    Mirrors `state._AtmData.pco = np.logspace(log10(P_b), log10(P_t), nz)` on
    the graph, so dL/dP_b and dL/dP_t compose through :func:`build_atm_static`.
    """
    P_b = jnp.asarray(P_b, dtype=jnp.float64)
    P_t = jnp.asarray(P_t, dtype=jnp.float64)
    frac = jnp.linspace(0.0, 1.0, nz)
    return 10.0 ** (jnp.log10(P_b) + frac * (jnp.log10(P_t) - jnp.log10(P_b)))


def _mu_dz_g(phys: PhysicalInputs, spec: AtmSpec):
    """On-graph reproduction of `atm_setup.compute_mu_dz_g` (no host asarray).

    Returns (mu, g, Hp, dz, dzi, Ti, Hpi). The discrete `pref_indx` anchor is
    taken from `spec` and held fixed, so the two height-integration scans run
    exactly as in the host path.
    """
    nz = spec.nz
    pref = spec.pref_indx
    pico = compute_pico(phys.pco)
    mu = compute_mean_mass(phys.ymix, spec.ms)

    gz_up, Hp_up, dz_up, _z_after_up = _scan_up_mu_dz_g(
        pref, phys.gs, phys.Rp, phys.Tco, mu, pico, nz
    )
    gz = jnp.zeros(nz, dtype=jnp.float64).at[pref:nz].set(gz_up)
    Hp = jnp.zeros(nz, dtype=jnp.float64).at[pref:nz].set(Hp_up)
    dz = jnp.zeros(nz, dtype=jnp.float64).at[pref:nz].set(dz_up)

    if pref > 0:
        gz_dn, Hp_dn, dz_dn, _z_dn = _scan_down_mu_dz_g(
            pref, phys.gs, phys.Rp, phys.Tco, mu, pico, z_at_pref=0.0
        )
        gz = gz.at[:pref].set(gz_dn)
        Hp = Hp.at[:pref].set(Hp_dn)
        dz = dz.at[:pref].set(dz_dn)

    dzi = 0.5 * (dz[1:] + jnp.roll(dz, 1)[1:])
    Ti = 0.5 * (phys.Tco[:-1] + jnp.roll(phys.Tco, -1)[:-1])
    Hpi = 0.5 * (Hp[:-1] + jnp.roll(Hp, -1)[:-1])
    return mu, gz, Hp, dz, dzi, Ti, Hpi


def _mol_diff(phys: PhysicalInputs, spec: AtmSpec, n_0, gz, Hp, dz):
    """On-graph reproduction of `atm_setup.compute_mol_diff`.

    Returns (Dzz, Dzz_cen, vm). All zero when `use_moldiff` is off.
    """
    nz, ni = spec.nz, spec.ni
    if not spec.use_moldiff:
        return (
            jnp.zeros((nz - 1, ni), dtype=jnp.float64),  # Dzz (interface)
            jnp.zeros((nz, ni), dtype=jnp.float64),  # Dzz_cen (cell)
            jnp.zeros((nz - 1, ni), dtype=jnp.float64),  # vm (interface)
        )

    Dzz_gen = _Dzz_gen_for_base(spec.atm_base)
    Tco = phys.Tco
    Tco_i = (Tco[1:] + Tco[:-1]) * 0.5
    n0_i = (n_0[1:] + n_0[:-1]) * 0.5
    Dzz = jax.vmap(lambda mi: Dzz_gen(Tco_i, n0_i, mi), out_axes=1)(spec.ms)
    Dzz_cen = jax.vmap(lambda mi: Dzz_gen(Tco, n_0, mi), out_axes=1)(spec.ms)

    # Non-gaseous species get zero diffusion on the interfaces; because the
    # advective term vm = -Dzz * drift below reuses this interface Dzz, the
    # zeroing carries straight into vm (no separate condensation gate needed).
    nongas = spec.nongas_mask[None, :]
    Dzz = jnp.where(nongas, 0.0, Dzz)

    vm = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    if spec.use_vm_mol:
        # Interface-centered upwind advective velocity (nz-1, ni), matching
        # VULCAN's vm_branch op.update_mu_dz. ONE implementation, in
        # atm_refresh (see its docstring); Hpi/dzi are the same interface means
        # _mu_dz_g uses, recomputed here so this stays a pure function of
        # (gz, Hp, dz).
        Hpi = 0.5 * (Hp[:-1] + Hp[1:])
        dzi = 0.5 * (dz[1:] + dz[:-1])
        vm = recompute_vm_jax(gz, Hpi, dzi, Dzz, spec.ms, spec.alpha, Tco,
                              kb, Navo)
    return Dzz, Dzz_cen, vm


def build_atm_static(phys: PhysicalInputs, spec: AtmSpec) -> AtmStatic:
    """Assemble a differentiable `AtmStatic` from physical inputs.

    Reproduces the host setup chain (`compute_mu_dz_g` -> `compute_mol_diff` ->
    `compute_settling_velocity` -> `make_atm_static`) entirely on the JAX graph.
    The result is field-for-field equal to the frozen `AtmStatic` the runner
    consumes (see `tests/test_atm_jax.py`), but carries tangents w.r.t. `phys` --
    for the configuration the runner actually uses (`atm_type`
    `file`/`analytical`/`isothermal`, `use_moldiff=on`). Two non-default modes
    differ, in both cases because this builder is the *more* self-consistent one:
      * `atm_type='table'`: production runs `f_pico` before `load_TPK` rewrites
        `pco` from the file and never recomputes `pico`, so it integrates the
        hydrostatic height with a stale `pico`; here `pico = compute_pico(phys.pco)`
        is consistent (so `g`/`dzi`/`Hpi` diverge ~1-12% in that mode).
      * `use_moldiff=off`: `Ti`/`Hpi` are computed as interface averages here,
        whereas production leaves them at legacy defaults (`Ti = Tco` full length,
        `Hpi = 0`); this is runtime-inert (gated behind `Dzz == 0`).
    """
    nz, ni = spec.nz, spec.ni

    # Number density n_0 = M = p / (kB T) (load_TPK), then the height
    # integration, molecular diffusion and settling.
    M = phys.pco / (kb * phys.Tco)
    n_0 = M
    _mu, gz, Hp, dz, dzi, Ti, Hpi = _mu_dz_g(phys, spec)
    Dzz, _Dzz_cen, vm = _mol_diff(phys, spec, n_0, gz, Hp, dz)

    use_vm = bool(spec.use_vm_mol and spec.use_moldiff)
    use_set = bool(spec.use_settling and spec.use_moldiff)

    if use_set:
        na, a, b = _VISCOSITY_TABLE[spec.atm_base]
        vs = settling_velocity_jax(na, a, b, phys.Tco, gz, spec.settle_coeff)
    else:
        vs = jnp.zeros((nz - 1, ni), dtype=jnp.float64)

    # Mirror make_atm_static's final toggle gating.
    if not use_vm:
        vm = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
    if not spec.use_moldiff:
        Dzz = jnp.zeros((nz - 1, ni), dtype=jnp.float64)

    return AtmStatic(
        Kzz=phys.Kzz,
        Dzz=Dzz,
        dzi=dzi,
        vz=phys.vz,
        Hpi=Hpi,
        Ti=Ti,
        Tco=phys.Tco,
        g=gz,
        ms=spec.ms,
        alpha=spec.alpha,
        M=M,
        vm=vm,
        vs=vs,
        top_flux=spec.top_flux,
        bot_flux=spec.bot_flux,
        bot_vdep=spec.bot_vdep,
        gas_indx_mask=spec.gas_indx_mask,
        diff_esc_mask=spec.diff_esc_mask,
        use_vm_mol=use_vm,
        use_settling=use_set,
        use_topflux=spec.use_topflux,
        use_botflux=spec.use_botflux,
    )


def make_physical_inputs(
    cfg, var, atm, species_list: list[str]
) -> tuple[PhysicalInputs, AtmSpec]:
    """Bridge a legacy (cfg, var, atm) setup into (PhysicalInputs, AtmSpec).

    Pulls the differentiable profiles into `PhysicalInputs` and the discrete
    configuration into `AtmSpec`, so `build_atm_static` reproduces the
    `make_atm_static` the production runner uses. Call after the host setup has
    populated `atm` (Tco, Kzz, ms, alpha, pref_indx, fluxes, ...).
    """
    nz = int(np.asarray(atm.Tco).shape[0])
    ni = len(species_list)

    phys = PhysicalInputs(
        pco=jnp.asarray(atm.pco, dtype=jnp.float64),
        Tco=jnp.asarray(atm.Tco, dtype=jnp.float64),
        ymix=jnp.asarray(var.ymix, dtype=jnp.float64),
        Kzz=jnp.asarray(atm.Kzz, dtype=jnp.float64),
        vz=jnp.asarray(atm.vz, dtype=jnp.float64),
        gs=jnp.asarray(surface_gravity(cfg), dtype=jnp.float64),
        Rp=jnp.asarray(float(cfg.Rp), dtype=jnp.float64),
    )

    gas_mask = np.zeros(ni, dtype=bool)
    gas_mask[np.asarray(atm.gas_indx, dtype=int)] = True
    nongas = np.zeros(ni, dtype=bool)
    for sp in cfg.non_gas_sp:
        if sp in species_list:
            nongas[species_list.index(sp)] = True
    # Diffusion-limited-escape species get an extra Jacobian top-diagonal term
    # (master op.py:2144-2148); mirrors make_atm_static so the two builders
    # stay field-for-field identical.
    _diff_esc_mask = np.zeros(ni, dtype=bool)
    for sp in getattr(cfg, "diff_esc", []) or []:
        if sp in species_list:
            _diff_esc_mask[species_list.index(sp)] = True

    if bool(getattr(cfg, "use_settling", False)):
        settle_coeff = settling_coeff_array(
            cfg,
            list(species_list),
            getattr(atm, "rho_p", {}),
            getattr(atm, "r_p", {}),
        )
    else:
        settle_coeff = np.zeros(ni, dtype=np.float64)

    spec = AtmSpec(
        nz=nz,
        ni=ni,
        pref_indx=int(atm.pref_indx),
        atm_base=str(cfg.atm_base),
        ms=jnp.asarray(atm.ms, dtype=jnp.float64),
        alpha=jnp.asarray(atm.alpha, dtype=jnp.float64),
        gas_indx_mask=jnp.asarray(gas_mask),
        nongas_mask=jnp.asarray(nongas),
        settle_coeff=jnp.asarray(settle_coeff, dtype=jnp.float64),
        top_flux=jnp.asarray(atm.top_flux, dtype=jnp.float64),
        bot_flux=jnp.asarray(atm.bot_flux, dtype=jnp.float64),
        bot_vdep=jnp.asarray(atm.bot_vdep, dtype=jnp.float64),
        diff_esc_mask=jnp.asarray(_diff_esc_mask),
        # getattr defaults mirror make_atm_static (jax_step.py) so the two
        # builders agree on a config that omits a toggle.
        use_moldiff=bool(getattr(cfg, "use_moldiff", True)),
        use_vm_mol=bool(getattr(cfg, "use_vm_mol", False)),
        use_settling=bool(getattr(cfg, "use_settling", False)),
        use_topflux=bool(getattr(cfg, "use_topflux", False)),
        use_botflux=bool(getattr(cfg, "use_botflux", False)),
    )
    return phys, spec
