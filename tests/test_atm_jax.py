"""Tests for the differentiable on-graph atmosphere builder (`atm_jax.py`).

Two things are pinned:

  1. Equivalence -- `build_atm_static` reproduces the frozen `AtmStatic` the
     production runner consumes (`make_atm_static`), field for field, at
     machine precision, for the configuration the runner uses (`atm_type`
     file/analytical/isothermal, `use_moldiff=on`). The integrated check runs on
     the real HD189 setup; two targeted checks cover the `use_vm_mol` and
     settling branches the HD189 config does not exercise. (`atm_type='table'`
     and `use_moldiff=off` intentionally differ -- build_atm_static is the more
     self-consistent one there; see the build_atm_static docstring.)
  2. Differentiability -- forward-mode tangents through `build_atm_static`
     match finite differences for dz/dgs and M/Dzz/dTco; the Heng+14 T(P)
     front-end and the pressure-grid / Kzz-profile front-ends differentiate too.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from vulcan_jax import atm_setup
from vulcan_jax.atm_jax import (
    AtmSpec,
    PhysicalInputs,
    build_atm_static,
    make_physical_inputs,
    pco_from_endpoints,
)
from vulcan_jax.atm_setup import analytical_TP_H14
from vulcan_jax.composition import species as _SPECIES
from vulcan_jax.jax_step import make_atm_static


def _rel(a, b):
    """Max relative error, with an absolute floor tied to the field's own
    magnitude so all-zero reference fields compare cleanly (0 vs 0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    floor = 1e-300 + 1e-30 * float(np.abs(b).max())
    denom = np.maximum(np.abs(b), floor)
    return float(np.max(np.abs(a - b) / denom))


# ---------------------------------------------------------------------------
# 1. Equivalence with the production make_atm_static path.
# ---------------------------------------------------------------------------


def test_build_atm_static_matches_make_atm_static(hd189_state):
    """Integrated: every AtmStatic field equals the frozen production build."""
    import vulcan_jax.vulcan_cfg as cfg

    var, atm = hd189_state.var, hd189_state.atm
    nz, ni = np.asarray(var.y).shape

    ref = make_atm_static(atm, ni, nz, cfg=cfg)
    phys, spec = make_physical_inputs(cfg, var, atm, list(_SPECIES))
    got = build_atm_static(phys, spec)

    float_fields = (
        "Kzz",
        "Dzz",
        "dzi",
        "vz",
        "Hpi",
        "Ti",
        "Tco",
        "g",
        "ms",
        "alpha",
        "M",
        "vm",
        "vs",
        "top_flux",
        "bot_flux",
        "bot_vdep",
    )
    for f in float_fields:
        # Machine precision; the height-integration averages (dzi/Hpi) carry a
        # few ULP of FP-ordering noise, everything else is bit-identical.
        assert _rel(getattr(got, f), getattr(ref, f)) < 1e-12, f

    assert np.array_equal(np.asarray(got.gas_indx_mask), np.asarray(ref.gas_indx_mask))
    for flag in ("use_vm_mol", "use_settling", "use_topflux", "use_botflux"):
        assert bool(getattr(got, flag)) == bool(getattr(ref, flag)), flag


def _synthetic_mol_diff_setup():
    """Small H2-base column with a condensible, for the vm/settling branches."""
    species_list = ["H2", "He", "CH4", "H2O_l_s"]
    masses = {"H2": 2.016, "He": 4.0026, "CH4": 16.04, "H2O_l_s": 18.015}
    ms_arr = np.array([masses[s] for s in species_list], dtype=np.float64)
    nz, ni = 16, len(species_list)
    rng = np.random.default_rng(0)
    Tco = np.linspace(2200.0, 700.0, nz)
    n_0 = np.logspace(18, 10, nz)
    g = np.full(nz, 2140.0)
    Hp = np.full(nz, 5e6) + rng.uniform(0, 1e5, nz)
    dz = np.full(nz, 4e5) + rng.uniform(0, 1e4, nz)
    alpha = atm_setup._alpha_array_for_base("H2", species_list, lambda s: masses[s])
    return species_list, ms_arr, nz, ni, Tco, n_0, g, Hp, dz, alpha


def test_mol_diff_vm_branch_matches_host():
    """_mol_diff (vm mode) reproduces compute_mol_diff at machine precision."""
    from vulcan_jax.atm_jax import _mol_diff

    species_list, ms_arr, nz, ni, Tco, n_0, g, Hp, dz, alpha = (
        _synthetic_mol_diff_setup()
    )
    cfg = types.SimpleNamespace(
        use_moldiff=True,
        atm_base="H2",
        non_gas_sp=["H2O_l_s"],
        use_vm_mol=True,
        use_condense=True,
    )
    ref = atm_setup.compute_mol_diff(
        cfg, Tco, n_0, g, Hp, dz, ms_arr, alpha, species_list
    )
    nongas = np.array([s in cfg.non_gas_sp for s in species_list])
    phys = PhysicalInputs(
        pco=jnp.ones(nz),
        Tco=jnp.asarray(Tco),
        ymix=jnp.ones((nz, ni)),
        Kzz=jnp.ones(nz - 1),
        vz=jnp.zeros(nz - 1),
        gs=jnp.float64(2140.0),
        Rp=jnp.float64(8e9),
    )
    spec = AtmSpec(
        nz=nz,
        ni=ni,
        pref_indx=0,
        atm_base="H2",
        ms=jnp.asarray(ms_arr),
        alpha=jnp.asarray(alpha),
        gas_indx_mask=jnp.asarray(~nongas),
        nongas_mask=jnp.asarray(nongas),
        settle_coeff=jnp.zeros(ni),
        top_flux=jnp.zeros(ni),
        bot_flux=jnp.zeros(ni),
        bot_vdep=jnp.zeros(ni),
        use_moldiff=True,
        use_vm_mol=True,
        use_settling=False,
        use_condense=True,
        use_topflux=False,
        use_botflux=False,
    )
    Dzz, Dzz_cen, vm = _mol_diff(
        phys, spec, jnp.asarray(n_0), jnp.asarray(g), jnp.asarray(Hp), jnp.asarray(dz)
    )
    assert _rel(Dzz, ref["Dzz"]) < 1e-12
    assert _rel(Dzz_cen, ref["Dzz_cen"]) < 1e-12
    assert _rel(vm, ref["vm"]) < 1e-12


def test_settling_velocity_jax_matches_host():
    """settling_velocity_jax (via the coeff array) reproduces the host vs."""
    species_list, ms_arr, nz, ni, Tco, n_0, g, Hp, dz, alpha = (
        _synthetic_mol_diff_setup()
    )
    cfg = types.SimpleNamespace(
        use_settling=True,
        atm_base="H2",
        non_gas_sp=["H2O_l_s"],
    )
    rho_p = {"H2O_l_s": 1.0}
    r_p = {"H2O_l_s": 1e-4}
    ref = atm_setup.compute_settling_velocity(cfg, Tco, g, species_list, rho_p, r_p)
    coeff = atm_setup.settling_coeff_array(cfg, species_list, rho_p, r_p)
    na, a, b = atm_setup._VISCOSITY_TABLE["H2"]
    got = atm_setup.settling_velocity_jax(
        na, a, b, jnp.asarray(Tco), jnp.asarray(g), jnp.asarray(coeff)
    )
    assert _rel(got, ref) < 1e-12
    # Only the condensible column is populated.
    assert np.allclose(np.asarray(got)[:, :3], 0.0)
    assert np.any(np.asarray(got)[:, 3] != 0.0)


# ---------------------------------------------------------------------------
# 2. Differentiability -- forward-mode tangents vs finite differences.
# ---------------------------------------------------------------------------


def test_jvp_dz_wrt_gs_matches_fd(hd189_state):
    """dz responds to surface gravity; jvp matches a central difference."""
    import vulcan_jax.vulcan_cfg as cfg

    var, atm = hd189_state.var, hd189_state.atm
    phys, spec = make_physical_inputs(cfg, var, atm, list(_SPECIES))

    def loss(gs):
        a = build_atm_static(phys._replace(gs=gs), spec)
        return jnp.sum(a.dzi)

    gs0 = float(phys.gs)
    _, tangent = jax.jvp(loss, (jnp.float64(gs0),), (jnp.float64(1.0),))
    eps = gs0 * 1e-6
    fd = (loss(jnp.float64(gs0 + eps)) - loss(jnp.float64(gs0 - eps))) / (2 * eps)
    assert np.isfinite(float(tangent)) and float(tangent) != 0.0
    assert abs(float(tangent) - float(fd)) / (abs(float(fd)) + 1e-300) < 1e-5


def test_jvp_M_Dzz_wrt_Tco_matches_fd(hd189_state):
    """Temperature is a differentiable leaf: scaling Tco moves M and Dzz, and
    the jvp matches a central difference. This is the headline T-sensitivity
    (warming the column); a specific T(P) parameterisation just composes in
    front of the Tco leaf (see `analytical_TP_H14`)."""
    import vulcan_jax.vulcan_cfg as cfg

    var, atm = hd189_state.var, hd189_state.atm
    phys, spec = make_physical_inputs(cfg, var, atm, list(_SPECIES))
    Tco0 = phys.Tco

    def loss(scale):
        a = build_atm_static(phys._replace(Tco=Tco0 * scale), spec)
        return jnp.sum(a.M) + jnp.sum(a.Dzz) + jnp.sum(a.dzi)

    _, tangent = jax.jvp(loss, (jnp.float64(1.0),), (jnp.float64(1.0),))
    eps = 1e-6
    fd = (loss(jnp.float64(1.0 + eps)) - loss(jnp.float64(1.0 - eps))) / (2 * eps)
    assert np.isfinite(float(tangent)) and float(tangent) != 0.0
    assert abs(float(tangent) - float(fd)) / (abs(float(fd)) + 1e-300) < 1e-5


def test_analytical_TP_front_end_differentiates():
    """The Heng+14 T(P) front-end composes onto the Tco leaf: dTco/dT_irr is
    finite. Evaluated on a NARROW pressure range -- `jax.scipy.special.expn`'s
    forward-mode is very slow when its argument spans a deep column's many
    decades, so fast dL/dT_irr over a full atmosphere wants a cheaper T(P) (or
    pass Tco directly). The plumbing (Tco differentiable) is unaffected."""
    pco = jnp.asarray(np.logspace(5.0, 4.0, 8))  # narrow -> small expn argument
    base = jnp.asarray([500.0, 1200.0, 0.1, 0.02, 0.7, 0.7], dtype=jnp.float64)

    def tprofile(T_irr):
        return jnp.sum(analytical_TP_H14(pco, base.at[1].set(T_irr), gs=2140.0, Pb=1e9))

    _, tangent = jax.jvp(tprofile, (jnp.float64(1200.0),), (jnp.float64(1.0),))
    assert np.isfinite(float(tangent)) and float(tangent) != 0.0


# Note: build_atm_static is proven field-for-field identical to the production
# make_atm_static by test_build_atm_static_matches_make_atm_static above, so it
# feeds jax_ros2_step / outer_loop.runner exactly as the frozen production
# AtmStatic does -- no separate through-solver test is needed. Forward-mode T
# sensitivity through a full run composes via the runner's lax.while_loop (with
# rates_jax for the T->k path); see examples/grad_physical_example.py.


def test_pco_from_endpoints_matches_logspace_and_differentiates():
    nz = 40
    P_b, P_t = 1e9, 1e-2
    got = np.asarray(pco_from_endpoints(jnp.float64(P_b), jnp.float64(P_t), nz))
    ref = np.logspace(np.log10(P_b), np.log10(P_t), nz)
    assert _rel(got, ref) < 1e-12
    # dpco/dP_b is finite.
    tangent = jax.jvp(
        lambda Pb: jnp.sum(pco_from_endpoints(Pb, jnp.float64(P_t), nz)),
        (jnp.float64(P_b),),
        (jnp.float64(1.0),),
    )[1]
    assert np.isfinite(float(tangent)) and float(tangent) != 0.0


def test_kzz_profile_jax_differentiates():
    """dKzz/dK_deep is finite where the JM16 floor is not active."""
    pico_int = jnp.asarray(np.logspace(7, -1, 30))

    def total(K_deep):
        return jnp.sum(
            atm_setup.kzz_profile_jax(
                "JM16",
                pico_int,
                K_deep=K_deep,
                K_max=1e10,
                K_p_lev=0.1,
                const_Kzz=1e9,
            )
        )

    tangent = jax.jvp(total, (jnp.float64(1e3),), (jnp.float64(1.0),))[1]
    assert np.isfinite(float(tangent))


def test_kzz_profile_jax_defaults_per_branch():
    """Each profile reads only its own param; the others default to 0.0."""
    pico = jnp.asarray(np.logspace(7, -1, 12))
    # 'const' needs only const_Kzz; passing nothing else must work.
    assert np.allclose(
        np.asarray(atm_setup.kzz_profile_jax("const", pico, const_Kzz=3.0)), 3.0
    )
    # JM16 needs only K_deep; Pfunc only K_max/K_p_lev.
    jm = atm_setup.kzz_profile_jax("JM16", pico, K_deep=1e6)
    assert np.all(np.asarray(jm) >= 1e6 - 1)
    pf = atm_setup.kzz_profile_jax("Pfunc", pico, K_max=1e10, K_p_lev=0.1)
    assert np.all(np.asarray(pf) >= 1e10 - 1)


def test_load_TPK_missing_kzz_param_fails_loud():
    """A JM16 config without K_deep must raise, not silently floor to 0.0."""
    import types

    from vulcan_jax.atm_setup import compute_pico, load_TPK

    pco = np.logspace(9, -2, 20)
    pico = np.asarray(compute_pico(pco))
    cfg = types.SimpleNamespace(
        atm_type="isothermal", Kzz_prof="JM16", vz_prof="const",
        use_Kzz=True, use_vz=False, const_Kzz=1e10, const_vz=0.0,
        K_max=2e5, K_p_lev=0.05, Tiso=1500.0, P_b=1e9, gs=2140.0,
    )  # note: no K_deep
    with pytest.raises(AttributeError):
        load_TPK(cfg, pco, pico=pico)
