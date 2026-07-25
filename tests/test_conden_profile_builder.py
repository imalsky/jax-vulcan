"""Unit tests for the on-graph condensation-profile builder.

`conden.build_conden_profile` must reproduce, from a live (traced) T-P and
structure, every temperature/structure-dependent quantity the legacy host
packer froze at setup: saturation number densities, growth/diffusion terms,
the H2O/NH3 relax inputs, the NH3 cold-trap index, and the fix-species
saturation mixing ratios.

The oracle here is an INDEPENDENT NumPy re-implementation of the upstream
formulas (op.conden's rate inputs at op.py:1128-1291, build_atm's saturation
formulas, and ini_abun._apply_condense's sat_mix construction) — not a call
back into the code under test. The host packer `_build_conden_static` now
delegates to the same builder, so this file carries the numerical parity
burden for both paths (the delegation itself was verified bit-exact against
the pre-refactor packer on isothermal and non-isothermal columns).

Also covered: temperature sensitivity (saturation curves and condensation
boundaries move with T; the NH3 cold-trap argmin moves layer-wise), and the
jit / vmap / forward-mode jvp contracts the retrieval's per-proposal rebuild
relies on.
"""

from types import SimpleNamespace

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from vulcan_jax import conden
from vulcan_jax.phy_const import Navo, kb

jax.config.update("jax_enable_x64", True)

NZ = 40
NI = 6
# Fake species table: columns of the fabricated Dzz array.
SPECIES_IDX = {"H2O": 0, "NH3": 1, "S8": 2, "H2O_l_s": 3, "NH3_l_s": 4, "S8_l_s": 5}
HUMIDITY = 0.7
R_P = {"H2O_l_s": 5e-3, "NH3_l_s": 5e-5, "S8_l_s": 1e-4}
RHO_P = {"H2O_l_s": 0.9, "NH3_l_s": 0.7, "S8_l_s": 2.07}


def _fake_setup(
    condense_sp=("H2O", "NH3", "S8"),
    use_relax=("H2O", "NH3"),
    fix_species=("H2O", "H2O_l_s", "S8"),
):
    """Minimal (cfg, var, atm) fakes for make_conden_spec — no network import."""
    cfg = SimpleNamespace(
        condense_sp=list(condense_sp),
        use_relax=list(use_relax),
        humidity=HUMIDITY,
        fix_species=list(fix_species),
    )
    var = SimpleNamespace(
        conden_re_list=[10, 14, 20],
        Rf={10: "H2O -> H2O_l_s", 14: "NH3 -> NH3_l_s", 20: "S8 -> S8_l_s"},
    )
    atm = SimpleNamespace(r_p=dict(R_P), rho_p=dict(RHO_P))
    return cfg, var, atm


def _profiles():
    """(T iso, T sloped, pco, n_0, Dzz) — 160-250 K, away from every sat-curve kink.

    Two pressure decades only: with a deep column the 1/p factor swamps the
    Clausius-Clapeyron drop and the sat-mix minimum pins to the bottom layer,
    which would make the cold-trap assertions vacuous."""
    rng = np.random.default_rng(7)
    pco = np.logspace(6, 4, NZ)  # dyne/cm^2
    T_iso = np.full(NZ, 230.0)
    # Sloped profile with a mid-column minimum (real cold trap inside the grid).
    z = np.linspace(0.0, 1.0, NZ)
    T_slope = 250.0 - 90.0 * np.exp(-((z - 0.6) ** 2) / 0.05)
    n_0 = pco / (kb * T_slope)
    Dzz = rng.uniform(1e4, 1e7, size=(NZ - 1, NI))
    return T_iso, T_slope, pco, n_0, Dzz


# --- independent NumPy oracle (upstream formulas, re-typed from master) ----


def _sat_p_np(sp, T):
    T = np.asarray(T, dtype=np.float64)
    if sp == "H2O":  # Murray ice/liquid split at 0 C
        T_C = T - 273.0
        ice = 6111.5 * np.exp((23.036 * T_C - T_C**2 / 333.7) / (T_C + 279.82))
        liq = 6112.1 * np.exp((18.729 * T_C - T_C**2 / 227.3) / (T_C + 257.87))
        return np.where(T_C < 0, ice, 0.0) + np.where(T_C > 0, liq, 0.0)
    if sp == "NH3":
        return np.exp(10.53 - 2161.0 / T - 86596.0 / T**2) * 1e6
    if sp == "S8":
        return np.where(
            T < 413.0,
            np.exp(20.0 - 11800.0 / T) * 1e6,
            np.exp(9.6 - 7510.0 / T) * 1e6,
        )
    raise KeyError(sp)


def _oracle(spec, T, pco, n_0, Dzz):
    """Master-form conden arrays: op.conden rate inputs + _apply_condense sat_mix."""
    out = {}
    Dg_rows, sat_rows = [], []
    for name, col in zip(spec.gas_names, spec.conden_sp_idx):
        Dg_rows.append(np.insert(Dzz[:, col], 0, Dzz[0, col]))
        sat_n = _sat_p_np(name, T) / kb / T
        if name == "H2O":
            sat_n = sat_n * HUMIDITY
        sat_rows.append(sat_n)
    out["Dg_per_re"] = np.stack(Dg_rows)
    out["sat_n_per_re"] = np.stack(sat_rows)
    out["h2o_Dg"] = np.insert(Dzz[:, SPECIES_IDX["H2O"]], 0, Dzz[0, SPECIES_IDX["H2O"]])
    out["h2o_sat"] = _sat_p_np("H2O", T) / kb / T * HUMIDITY
    out["nh3_Dg"] = np.insert(Dzz[:, SPECIES_IDX["NH3"]], 0, Dzz[0, SPECIES_IDX["NH3"]])
    out["nh3_sat"] = _sat_p_np("NH3", T) / kb / T
    out["nh3_conden_top"] = int(np.argmin(out["nh3_sat"] / n_0))
    fix_rows = []
    for name in spec.fix_names:
        if name in spec.sat_names:
            sm = np.minimum(1.0, _sat_p_np(name, T) / pco)
            if name == "H2O":
                sm = sm * HUMIDITY
            fix_rows.append(sm)
        else:
            fix_rows.append(np.zeros_like(T))
    out["fix_species_sat_mix"] = np.stack(fix_rows)
    return out


@pytest.fixture(scope="module")
def spec():
    cfg, var, atm = _fake_setup()
    return conden.make_conden_spec(cfg, var, atm, SPECIES_IDX)


def test_make_conden_spec_metadata(spec):
    """Static metadata: identity, k_arr rows, relax coeff short-circuits."""
    assert spec.gas_names == ("H2O", "NH3", "S8")
    assert spec.conden_re_idx == (10, 14, 20)
    assert spec.conden_sp_idx == (0, 1, 2)
    # H2O and NH3 are in use_relax -> kinetics coefficient short-circuited to 0
    assert spec.coeff_per_re[0] == 0.0
    assert spec.coeff_per_re[1] == 0.0
    m_s8 = conden.GAS_MASS_G_PER_MOL["S8"] / Navo
    assert spec.coeff_per_re[2] == pytest.approx(
        m_s8 / (RHO_P["S8_l_s"] * R_P["S8_l_s"] ** 2), rel=1e-15
    )
    assert spec.h2o_active and spec.nh3_active
    assert spec.h2o_idx == 0 and spec.h2o_l_s_idx == 3
    assert spec.nh3_idx == 1 and spec.nh3_l_s_idx == 4
    assert spec.humidity == HUMIDITY
    assert spec.fix_names == ("H2O", "H2O_l_s", "S8")


def test_make_conden_spec_inactive_species_skipped():
    """A network conden reaction outside condense_sp contributes no row."""
    cfg, var, atm = _fake_setup(condense_sp=("S8",), use_relax=(), fix_species=())
    sp = conden.make_conden_spec(cfg, var, atm, SPECIES_IDX)
    assert sp.gas_names == ("S8",)
    assert not sp.h2o_active and not sp.nh3_active


def test_builder_matches_upstream_formulas(spec):
    """Machine-precision parity with the independent NumPy oracle, both on an
    isothermal and a non-isothermal column."""
    T_iso, T_slope, pco, n_0, Dzz = _profiles()
    for T in (T_iso, T_slope):
        prof = conden.build_conden_profile(spec, T, pco, n_0, Dzz)
        ref = _oracle(spec, T, pco, n_0, Dzz)
        for field, want in ref.items():
            got = np.asarray(getattr(prof, field))
            np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0, err_msg=field)


def test_temperature_moves_saturation_and_boundaries(spec):
    """Warming the column raises every saturation number density and moves the
    supersaturation boundary; the NH3 cold-trap argmin follows the T minimum."""
    T_iso, T_slope, pco, n_0, Dzz = _profiles()
    p_cold = conden.build_conden_profile(spec, T_slope, pco, n_0, Dzz)
    p_warm = conden.build_conden_profile(spec, T_slope + 15.0, pco, n_0, Dzz)
    # d(sat_n)/dT > 0 for all three condensables in this regime
    assert np.all(np.asarray(p_warm.sat_n_per_re) > np.asarray(p_cold.sat_n_per_re))
    # Condensation boundary: an S8 abundance pinned mid-way between the
    # column's saturation extremes crosses saturation somewhere by
    # construction, and the crossing layers move once the column warms
    # (a 15 K shift moves sat_n by ~e^3 at these temperatures).
    sat_cold = np.asarray(p_cold.sat_n_per_re)[2]
    y_s8 = np.sqrt(sat_cold.min() * sat_cold.max()) * np.ones(NZ)
    sup_cold = y_s8 > sat_cold
    sup_warm = y_s8 > np.asarray(p_warm.sat_n_per_re)[2]
    assert sup_cold.any() and not sup_cold.all(), "boundary must lie inside grid"
    assert (sup_cold != sup_warm).any(), "boundary must move with T"
    # Cold-trap index: mid-column T minimum vs a monotonically colder top.
    T_mono = np.linspace(260.0, 160.0, NZ)
    top_slope = int(
        conden.build_conden_profile(spec, T_slope, pco, n_0, Dzz).nh3_conden_top
    )
    top_mono = int(
        conden.build_conden_profile(spec, T_mono, pco, n_0, Dzz).nh3_conden_top
    )
    assert top_slope != top_mono


def test_jit_and_vmap_consistency(spec):
    T_iso, T_slope, pco, n_0, Dzz = _profiles()

    def build(T):
        return conden.build_conden_profile(spec, T, pco, n_0, Dzz)

    def _assert_profile_close(got, want, label):
        # exp-chain fusion under jit/vmap drifts by ~1 ulp; the integer
        # cold-trap index must stay exact.
        for f in conden.CondenProfile._fields:
            a = np.asarray(getattr(got, f))
            b = np.asarray(getattr(want, f))
            if f == "nh3_conden_top":
                np.testing.assert_array_equal(a, b, err_msg=f"{f} {label}")
            else:
                np.testing.assert_allclose(
                    a, b, rtol=1e-14, atol=0.0, err_msg=f"{f} {label}"
                )

    eager = build(jnp.asarray(T_slope))
    jitted = jax.jit(build)(jnp.asarray(T_slope))
    _assert_profile_close(jitted, eager, "jit vs eager")

    T_batch = jnp.stack([jnp.asarray(T_iso), jnp.asarray(T_slope)])
    batched = jax.vmap(build)(T_batch)
    for lane, T in enumerate((T_iso, T_slope)):
        solo = build(jnp.asarray(T))
        lane_prof = conden.CondenProfile(
            *[
                jnp.asarray(np.asarray(getattr(batched, f))[lane])
                for f in conden.CondenProfile._fields
            ]
        )
        _assert_profile_close(lane_prof, solo, f"vmap lane {lane}")


def test_jvp_matches_finite_difference(spec):
    """Forward-mode d(sat quantities)/dT against centered finite differences,
    away from the phase-boundary kinks (T in 160-250 K; H2O kink at 273 K,
    S8 break at 413 K). The integer cold-trap index carries no tangent."""
    _, T_slope, pco, n_0, Dzz = _profiles()
    T0 = jnp.asarray(T_slope)
    dT = jnp.ones_like(T0)

    def smooth_outputs(T):
        p = conden.build_conden_profile(spec, T, pco, n_0, Dzz)
        return p.sat_n_per_re, p.h2o_sat, p.nh3_sat, p.fix_species_sat_mix

    _, tangents = jax.jvp(smooth_outputs, (T0,), (dT,))
    eps = 1e-3
    plus = smooth_outputs(T0 + eps)
    minus = smooth_outputs(T0 - eps)
    for got, hi, lo, name in zip(
        tangents, plus, minus, ("sat_n_per_re", "h2o_sat", "nh3_sat", "fix_sat_mix")
    ):
        fd = (np.asarray(hi) - np.asarray(lo)) / (2 * eps)
        got = np.asarray(got)
        assert np.all(np.isfinite(got)), name
        scale = np.maximum(np.abs(fd), np.max(np.abs(fd)) * 1e-12 + 1e-300)
        # fix_sat_mix rows clipped at 1.0 have exactly-zero derivative on both
        # sides; compare only where FD is nonzero.
        mask = fd != 0.0
        np.testing.assert_allclose(
            got[mask] / scale[mask], fd[mask] / scale[mask], rtol=5e-6, err_msg=name
        )


def test_zero_reaction_and_inactive_blocks_shapes():
    """use_condense with no live reactions/relax blocks packs zero-size or
    zero-filled arrays with stable shapes (the runner still builds a branch)."""
    cfg, var, atm = _fake_setup(condense_sp=(), use_relax=(), fix_species=())
    sp = conden.make_conden_spec(cfg, var, atm, SPECIES_IDX)
    T_iso, _, pco, n_0, Dzz = _profiles()
    prof = conden.build_conden_profile(sp, T_iso, pco, n_0, Dzz)
    assert prof.Dg_per_re.shape == (0, NZ)
    assert prof.sat_n_per_re.shape == (0, NZ)
    assert prof.fix_species_sat_mix.shape == (0, NZ)
    assert np.all(np.asarray(prof.h2o_sat) == 0.0)
    assert np.all(np.asarray(prof.nh3_sat) == 0.0)
    assert int(prof.nh3_conden_top) == 0
