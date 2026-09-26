"""Differentiable on-graph atmosphere builder.

`atm_setup.py` computes every atmosphere-derived array (Tco, M/n_0, dz, Hp, g,
Dzz, vm, vs, ...) at pre-loop setup and freezes it to NumPy before the runner.
This module re-expresses the same cascade as one differentiable
function, :func:`build_atm_static`, so gradients flow from the physical inputs
(T(P) profile / T_irr, surface gravity, planet radius, the pressure grid, eddy
and molecular diffusion, composition) all the way to the `AtmStatic` the Ros2
step consumes.

Usage -- pair it with forward-mode AD (`jax.lax.while_loop` supports `jvp`;
`OuterLoop.run_jvp` certifies the tangent). With `integ` an `OuterLoop`,
`init_state, _ = integ.prepare_runstate(rs)` and a caller's `loss_fn(y)`::

    phys, spec = make_physical_inputs(cfg, var, atm, species_list)
    tp = jnp.asarray(cfg.para_anaTP)                  # Heng+14 (T_int, T_irr, ...)

    def loss_of_Tirr(T_irr):
        Tco = analytical_TP_H14(phys.pco, tp.at[1].set(T_irr),
                                gs=phys.gs, Pb=phys.pco[0])
        atm_static = build_atm_static(phys._replace(Tco=Tco), spec)
        return loss_fn(integ._runner(init_state, atm_static).y)

    dL_dTirr = jax.jvp(loss_of_Tirr, (tp[1],), (1.0,))[1]

`spec` is static configuration (species list, atm_base, toggles, the
discrete reference layer ``pref_indx``); differentiate w.r.t. the
:class:`PhysicalInputs` pytree only, closing over `spec`.

Differentiable here: pco (-> P_b/P_t via :func:`pco_from_endpoints`), Tco
(-> T_irr via `analytical_TP_H14`), ymix (mean molecular weight -> scale
height), Kzz (-> profile params via `atm_setup.kzz_profile_jax`), vz, gs, Rp.
Reachable downstream: M/n_0, mu, g, Hp, dz, dzi, Ti, Hpi, Dzz, Dzz_cen, vm, vs.

Not differentiable here:
  * The equilibrium initial abundances (`ini_abun.eq_seed` returns a zero
    tangent). Use the `const_lowT` initialiser for a differentiable
    elemental-abundance path, or pass `ymix` as a leaf.
  * Photolysis T-dependent cross-section re-interpolation
    (`photo_setup._bin_T_dependent`): cross-sections enter `PhotoStaticInputs`
    as injectable JAX arrays (so dL/d(cross-section) works), but their T-rebake
    is host-side, so dL/dT *through* the cross-sections does not.
  * The thermal-diffusion factor `alpha` and the hydrostatic reference layer
    `pref_indx` are discrete/static lookups, held fixed (correct to first
    order in a small physical perturbation).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .atm_setup import (
    _VISCOSITY_TABLE,
    compute_mean_mass,
    compute_pico,
    mol_diff_jax,
    mu_dz_g_jax,
    settling_coeff_array,
    settling_velocity_jax,
    surface_gravity,
)
from .jax_step import AtmStatic
from .phy_const import kb


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
    # No use_condense: condensation is a per-step process driven from the
    # ProfileVars carry.


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
    """Mean mass plus the hydrostatic cascade, with no host `np.asarray`.

    The height integration is `atm_setup.mu_dz_g_jax`, the same body the host
    setup path runs; only the inputs differ (`pico` is recomputed from the
    differentiable `pco`, and the discrete `pref_indx` anchor comes from
    `spec` and is held fixed). Returns (mu, g, Hp, dz, dzi, Ti, Hpi).
    """
    pico = compute_pico(phys.pco)
    mu = compute_mean_mass(phys.ymix, spec.ms)
    gz, Hp, dz, _zco, _zmco, dzi, Ti, Hpi = mu_dz_g_jax(
        spec.pref_indx, phys.gs, phys.Rp, phys.Tco, mu, pico, spec.nz
    )
    return mu, gz, Hp, dz, dzi, Ti, Hpi


def _mol_diff(phys: PhysicalInputs, spec: AtmSpec, n_0, gz, Hp, dz):
    """Molecular diffusion for `build_atm_static`: (Dzz, Dzz_cen, vm).

    The body is `atm_setup.mol_diff_jax`, the same one the host setup path
    runs. All three are zero when `use_moldiff` is off.
    """
    nz, ni = spec.nz, spec.ni
    if not spec.use_moldiff:
        return (
            jnp.zeros((nz - 1, ni), dtype=jnp.float64),  # Dzz (interface)
            jnp.zeros((nz, ni), dtype=jnp.float64),  # Dzz_cen (cell)
            jnp.zeros((nz - 1, ni), dtype=jnp.float64),  # vm (interface)
        )
    return mol_diff_jax(
        spec.atm_base,
        phys.Tco,
        n_0,
        gz,
        Hp,
        dz,
        spec.ms,
        spec.alpha,
        spec.nongas_mask,
        use_vm_mol=spec.use_vm_mol,
    )


def build_atm_static(phys: PhysicalInputs, spec: AtmSpec) -> AtmStatic:
    """Assemble a differentiable `AtmStatic` from physical inputs.

    Reproduces the host setup chain (`compute_mu_dz_g` -> `compute_mol_diff` ->
    `compute_settling_velocity` -> `make_atm_static`) on the JAX graph: field
    for field equal to the runner's `AtmStatic` for `atm_type`
    `file`/`analytical`/`isothermal` with `use_moldiff` on
    (`tests/test_atm_jax.py`), with tangents w.r.t. `phys`. It differs for
    `atm_type='table'`, where production keeps upstream's stale `pico`, and with
    `use_moldiff` off, where `Ti`/`Hpi` differ but are runtime-inert.
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
    # diff_esc species get the top-diagonal escape term (op.py:2102-2107);
    # same mask as make_atm_static.
    _diff_esc_mask = np.zeros(ni, dtype=bool)
    for sp in cfg.diff_esc:
        if sp in species_list:
            _diff_esc_mask[species_list.index(sp)] = True

    if bool(cfg.use_settling):
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
        use_moldiff=bool(cfg.use_moldiff),
        use_vm_mol=bool(cfg.use_vm_mol),
        use_settling=bool(cfg.use_settling),
        use_topflux=bool(cfg.use_topflux),
        use_botflux=bool(cfg.use_botflux),
    )
    return phys, spec
