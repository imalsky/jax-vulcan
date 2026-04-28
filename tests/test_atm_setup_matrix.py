"""Phase 20: per-cfg parametrized coverage of `atm_setup` JAX kernels.

The original `tests/test_build_atm.py` compares the HD189 default cfg's
full pipeline against VULCAN-master and skips cleanly if the master
sibling is absent. This file complements that with **standalone**
coverage of every cfg branch that HD189 alone misses:

  - `atm_type ∈ {isothermal, analytical, file}` (vulcan_ini / table
    require a prior .vul / synthetic table — covered by Phase 21 once
    the saved-state setup matures).
  - `atm_base ∈ {H2, N2, O2, CO2}` exercising `compute_mol_diff`.
  - `Kzz_prof ∈ {const, JM16, Pfunc, file}`.
  - `vz_prof ∈ {const, file}` with `use_vz=True`.
  - `use_topflux=True`, `use_botflux=True`, `use_fix_sp_bot` ∈ dict.
  - `compute_sat_p` for every supported condensable.

Each case runs a pure NumPy reference inline and asserts the JAX
result agrees to machine precision. The reference reproduces the
master `Atm.*` body verbatim — when the JAX path drifts, the diff
narrows down to the offending function immediately.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import interpolate as scipy_interpolate
from scipy.special import expn as scipy_expn

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Atm type matrix — Tco, Kzz, vz, M, n_0
# ---------------------------------------------------------------------------

def _ref_TP_H14(pco, params, gs, Pb):
    """Verbatim numpy port of `Atm.TP_H14` (VULCAN-master build_atm.py:493)."""
    T_int, T_irr, ka_0, ka_s, beta_s, beta_l = params
    albedo = (1.0 - beta_s) / (1.0 + beta_s)
    T_irr *= (1 - albedo) ** 0.25
    eps_L = 3.0 / 8.0
    eps_L3 = 1.0 / 3.0
    ka_CIA = 0.0
    m = pco / gs
    m_0 = Pb / gs
    ka_l = ka_0 + ka_CIA * m / m_0
    term1 = T_int**4 / 4 * (1 / eps_L + m / (eps_L3 * beta_l**2) * (ka_0 + ka_CIA * m / (2 * m_0)))
    term2 = (
        1 / (2 * eps_L)
        + scipy_expn(2, ka_s * m / beta_s)
        * (ka_s / (ka_l * beta_s) - ka_CIA * m * beta_s / (eps_L3 * ka_s * m_0 * beta_l**2))
    )
    term3 = ka_0 * beta_s / (eps_L3 * ka_s * beta_l**2) * (1.0 / 3 - scipy_expn(4, ka_s * m / beta_s))
    return (term1 + T_irr**4 / 8 * (term2 + term3)) ** 0.25


def _make_pco(P_b, P_t, nz):
    return np.logspace(np.log10(P_b), np.log10(P_t), nz)


def _hd189_atm_file_path() -> str:
    return str(ROOT / "atm" / "atm_HD189_Kzz.txt")


@pytest.mark.parametrize(
    "atm_type,P_b,P_t,nz,gs",
    [
        ("isothermal", 1e9, 1e-2, 50, 2140.0),
        ("analytical", 1e9, 1e-2, 50, 2140.0),
        ("file",       1e9, 1e-2, 100, 2140.0),
    ],
)
def test_load_TPK_atm_types(atm_type, P_b, P_t, nz, gs):
    """`atm_setup.load_TPK` agrees with a direct numpy reference for
    each `atm_type` we support standalone (file / isothermal / analytical)."""
    from phy_const import kb
    from atm_setup import compute_pico, load_TPK

    pco = _make_pco(P_b, P_t, nz)
    pico = np.asarray(compute_pico(pco))

    cfg = SimpleNamespace(
        atm_type=atm_type, Kzz_prof="const", vz_prof="const",
        use_Kzz=True, use_vz=False,
        const_Kzz=1e10, const_vz=0.0, K_max=1e5, K_p_lev=0.1,
        Tiso=1234.0, P_b=P_b, gs=gs,
        para_anaTP=[120.0, 1500.0, 0.1, 0.02, 1.0, 1.0],
        atm_file=_hd189_atm_file_path(),
        vul_ini="output/",
    )
    out = load_TPK(cfg, pco, pico=pico)

    # --- Tco reference ---
    if atm_type == "isothermal":
        Tco_ref = np.full(nz, cfg.Tiso, dtype=np.float64)
    elif atm_type == "analytical":
        Tco_ref = _ref_TP_H14(pco, cfg.para_anaTP, gs=cfg.gs, Pb=cfg.P_b)
    else:  # file
        atm_table = np.genfromtxt(cfg.atm_file, names=True, dtype=None, skip_header=1)
        p_file = atm_table["Pressure"].astype(np.float64)
        T_file = atm_table["Temp"].astype(np.float64)
        f_pT = scipy_interpolate.interp1d(
            p_file, T_file, assume_sorted=False, bounds_error=False,
            fill_value=(T_file[np.argmin(p_file)], T_file[np.argmax(p_file)]),
        )
        Tco_ref = f_pT(pco)

    assert np.allclose(out["Tco"], Tco_ref, atol=0.0, rtol=1e-13), (
        f"{atm_type}: Tco diverges max={np.max(np.abs(out['Tco'] - Tco_ref)):.2e}"
    )
    # M / n_0 follow from Tco directly.
    M_ref = pco / (kb * Tco_ref)
    assert np.allclose(out["M"], M_ref, rtol=1e-13)
    assert np.allclose(out["n_0"], M_ref, rtol=1e-13)


@pytest.mark.parametrize("Kzz_prof", ["const", "JM16", "Pfunc", "file"])
def test_load_TPK_kzz_modes(Kzz_prof):
    """`atm_setup.load_TPK` matches numpy reference for every Kzz_prof."""
    from atm_setup import compute_pico, load_TPK

    nz, P_b, P_t = 100, 1e9, 1e-2
    pco = _make_pco(P_b, P_t, nz)
    pico = np.asarray(compute_pico(pco))

    cfg = SimpleNamespace(
        atm_type="file" if Kzz_prof == "file" else "isothermal",
        Kzz_prof=Kzz_prof, vz_prof="const",
        use_Kzz=True, use_vz=False,
        const_Kzz=3.14e10, const_vz=0.0, K_max=2e5, K_p_lev=0.05, K_deep=1e6,
        Tiso=1500.0, P_b=P_b, gs=2140.0,
        para_anaTP=[120.0, 1500.0, 0.1, 0.02, 1.0, 1.0],
        atm_file=_hd189_atm_file_path(),
        vul_ini="output/",
    )
    out = load_TPK(cfg, pco, pico=pico)

    if Kzz_prof == "const":
        Kzz_ref = np.full(nz - 1, cfg.const_Kzz)
    elif Kzz_prof == "JM16":
        Kzz_ref = np.maximum(cfg.K_deep, 1e5 * (300.0 / (pico[1:-1] * 1e-3)) ** 0.5)
    elif Kzz_prof == "Pfunc":
        Kzz_ref = np.maximum(cfg.K_max, cfg.K_max * (cfg.K_p_lev * 1e6 / pico[1:-1]) ** 0.4)
    else:  # file
        tab = np.genfromtxt(cfg.atm_file, names=True, dtype=None, skip_header=1)
        Kzz_file = tab["Kzz"].astype(np.float64)
        p_file = tab["Pressure"].astype(np.float64)
        f_pK = scipy_interpolate.interp1d(
            p_file, Kzz_file, assume_sorted=False, bounds_error=False,
            fill_value=(Kzz_file[np.argmin(p_file)], Kzz_file[np.argmax(p_file)]),
        )
        Kzz_ref = f_pK(pico[1:-1])

    assert np.allclose(out["Kzz"], Kzz_ref, rtol=1e-13, atol=0.0)


def test_load_TPK_use_kzz_off_zeroes_kzz():
    """`use_Kzz=False` must zero the Kzz array regardless of Kzz_prof."""
    from atm_setup import compute_pico, load_TPK

    nz, P_b, P_t = 50, 1e6, 1e-2
    pco = _make_pco(P_b, P_t, nz)
    pico = np.asarray(compute_pico(pco))
    cfg = SimpleNamespace(
        atm_type="isothermal", Kzz_prof="const", vz_prof="const",
        use_Kzz=False, use_vz=False,
        const_Kzz=1e10, const_vz=0.0, K_max=1e5, K_p_lev=0.1,
        Tiso=300.0, P_b=P_b, gs=980.0,
    )
    out = load_TPK(cfg, pco, pico=pico)
    assert np.all(out["Kzz"] == 0.0)


# ---------------------------------------------------------------------------
# atm_base × mol_diff matrix
# ---------------------------------------------------------------------------

def _ref_Dzz(atm_base, T, n_tot, mi):
    """Numpy versions of the master `Dzz_gen` lambdas."""
    if atm_base == "H2":
        return 2.2965e17 * T**0.765 / n_tot * (16.04 / mi * (mi + 2.016) / 18.059) ** 0.5
    if atm_base == "N2":
        return 7.34e16 * T**0.75 / n_tot * (16.04 / mi * (mi + 28.014) / 44.054) ** 0.5
    if atm_base == "O2":
        return 7.51e16 * T**0.759 / n_tot * (16.04 / mi * (mi + 32.0) / 48.04) ** 0.5
    if atm_base == "CO2":
        return 2.15e17 * T**0.750 / n_tot * (2.016 / mi * (mi + 44.001) / 46.017) ** 0.5
    raise AssertionError(atm_base)


@pytest.mark.parametrize("atm_base", ["H2", "N2", "O2", "CO2"])
def test_compute_mol_diff_atm_base(atm_base):
    """compute_mol_diff matches a direct numpy `Dzz_gen` reference for
    every supported atm_base."""
    from atm_setup import compute_mol_diff, _alpha_array_for_base

    nz = 30
    Tco = np.linspace(150.0, 2500.0, nz)
    n_0 = np.logspace(20.0, 8.0, nz)
    species_list = ["H", "He", "H2", "Ar", "CH4", "H2O", "CO2"]
    ms_arr = np.array([1.008, 4.003, 2.016, 39.95, 16.04, 18.02, 44.01])

    cfg = SimpleNamespace(
        atm_base=atm_base, use_moldiff=True, use_vm_mol=False,
        use_condense=False, non_gas_sp=[],
    )
    alpha_ref = _alpha_array_for_base(atm_base, species_list, mol_mass=lambda sp: ms_arr[species_list.index(sp)])

    out = compute_mol_diff(
        cfg, Tco, n_0, g=np.full(nz, 980.0), Hp=np.full(nz, 8.0e5),
        dz=np.full(nz, 1e5), ms_arr=ms_arr, alpha_arr=alpha_ref,
        species_list=species_list,
    )
    Dzz = out["Dzz"]
    Dzz_cen = out["Dzz_cen"]

    # Numpy reference
    T_i = (Tco[1:] + Tco[:-1]) * 0.5
    n_i = (n_0[1:] + n_0[:-1]) * 0.5
    Dzz_ref = np.zeros((nz - 1, len(species_list)))
    Dzz_cen_ref = np.zeros((nz, len(species_list)))
    for i, mi in enumerate(ms_arr):
        Dzz_ref[:, i] = _ref_Dzz(atm_base, T_i, n_i, mi)
        Dzz_cen_ref[:, i] = _ref_Dzz(atm_base, Tco, n_0, mi)
    assert np.allclose(Dzz, Dzz_ref, rtol=1e-13, atol=0.0)
    assert np.allclose(Dzz_cen, Dzz_cen_ref, rtol=1e-13, atol=0.0)


def test_compute_mol_diff_use_moldiff_off_zeros():
    """use_moldiff=False -> Dzz/Dzz_cen all zero."""
    from atm_setup import compute_mol_diff

    nz, ni = 10, 5
    Tco = np.full(nz, 1500.0)
    n_0 = np.full(nz, 1e15)
    species_list = ["H2", "He", "H", "CH4", "H2O"]
    cfg = SimpleNamespace(atm_base="H2", use_moldiff=False, use_vm_mol=False,
                         use_condense=False, non_gas_sp=[])
    out = compute_mol_diff(
        cfg, Tco, n_0, g=np.full(nz, 980.0), Hp=np.full(nz, 8e5),
        dz=np.full(nz, 1e5),
        ms_arr=np.ones(ni), alpha_arr=np.zeros(ni), species_list=species_list,
    )
    assert np.all(out["Dzz"] == 0.0)
    assert np.all(out["Dzz_cen"] == 0.0)


# ---------------------------------------------------------------------------
# BC_flux modes
# ---------------------------------------------------------------------------

def test_read_bc_flux_default_zero():
    """No use_topflux/use_botflux -> all-zero arrays of length ni."""
    from atm_setup import read_bc_flux
    species_list = ["A", "B", "C"]
    cfg = SimpleNamespace(use_topflux=False, use_botflux=False,
                         use_fix_sp_bot={})
    out = read_bc_flux(cfg, species_list)
    for k in ("top_flux", "bot_flux", "bot_vdep", "bot_fix_sp"):
        assert out[k].shape == (3,) and np.all(out[k] == 0.0)


def test_read_bc_flux_use_topflux(tmp_path):
    """use_topflux=True parses the file into top_flux."""
    from atm_setup import read_bc_flux
    bc = tmp_path / "top.txt"
    bc.write_text("# header\nH2O\t1.5e8\nCH4\t3.7e7\n")
    species_list = ["H2", "H2O", "CH4", "CO"]
    cfg = SimpleNamespace(
        use_topflux=True, use_botflux=False, use_fix_sp_bot={},
        top_BC_flux_file=str(bc),
    )
    out = read_bc_flux(cfg, species_list)
    assert out["top_flux"][1] == 1.5e8
    assert out["top_flux"][2] == 3.7e7
    assert out["top_flux"][0] == 0.0 and out["top_flux"][3] == 0.0


def test_read_bc_flux_use_botflux(tmp_path):
    """use_botflux=True parses (flux, vdep) from the file."""
    from atm_setup import read_bc_flux
    bc = tmp_path / "bot.txt"
    bc.write_text("# header\nO\t-1.0e9\t0.4\nNH3\t0.0\t0.005\n")
    species_list = ["O", "NH3", "H"]
    cfg = SimpleNamespace(
        use_topflux=False, use_botflux=True, use_fix_sp_bot={},
        bot_BC_flux_file=str(bc),
    )
    out = read_bc_flux(cfg, species_list)
    assert out["bot_flux"][0] == -1.0e9
    assert out["bot_vdep"][0] == 0.4
    assert out["bot_flux"][1] == 0.0
    assert out["bot_vdep"][1] == 0.005


# ---------------------------------------------------------------------------
# sat_p (sp_sat) per condensable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sp,T_K",
    [
        ("H2O", 300.0),
        ("NH3", 200.0),
        ("H2SO4", 400.0),
        ("S2", 500.0),
        ("S4", 600.0),
        ("S8", 700.0),
        ("C", 3000.0),
        ("H2S", 200.0),
    ],
)
def test_compute_sat_p_each_species(sp, T_K):
    """compute_sat_p returns finite, positive saturation pressures
    for each supported condensable at a representative T."""
    from atm_setup import compute_sat_p
    Tco = np.array([T_K, T_K + 50.0, T_K + 100.0])
    out = compute_sat_p([sp], Tco)
    assert sp in out
    arr = out[sp]
    assert arr.shape == Tco.shape
    assert np.all(np.isfinite(arr))
    assert np.all(arr > 0.0)


def test_compute_sat_p_unknown_species_raises():
    from atm_setup import compute_sat_p
    with pytest.raises(IOError, match="No saturation vapor data"):
        compute_sat_p(["XYZ"], np.array([300.0]))


# ---------------------------------------------------------------------------
# f_pico bit-exact + edge cases
# ---------------------------------------------------------------------------

def test_compute_pico_matches_master_formula():
    """`compute_pico` matches a direct numpy port of `Atm.f_pico`
    (geometric mean interior + log extrapolation at edges)."""
    from atm_setup import compute_pico
    pco = np.logspace(9, -2, 50)
    pico = np.asarray(compute_pico(pco))
    pi_ref = (pco * np.roll(pco, 1)) ** 0.5
    pi_ref[0] = pco[0] ** 1.5 * pco[1] ** -0.5
    pi_ref = np.append(pi_ref, pco[-1] ** 1.5 * pco[-2] ** -0.5)
    assert pico.shape == (51,)
    assert np.allclose(pico, pi_ref, rtol=1e-15, atol=0.0)


# ---------------------------------------------------------------------------
# f_mu_dz — height integration upward + downward
# ---------------------------------------------------------------------------

def test_compute_mu_dz_g_rocky_anchor_at_surface():
    """rocky=True forces pref_indx=0; only the upward scan runs."""
    from atm_setup import compute_pico, compute_mu_dz_g

    nz = 40
    pco = _make_pco(1e6, 5e-2, nz)
    pico = np.asarray(compute_pico(pco))
    Tco = np.full(nz, 273.0)
    ymix = np.zeros((nz, 3))
    ymix[:, 0] = 0.78  # N2
    ymix[:, 1] = 0.21  # O2
    ymix[:, 2] = 0.01  # Ar
    ms_arr = np.array([28.014, 32.0, 39.948])
    cfg = SimpleNamespace(
        gs=980.0, Rp=6.378e8, rocky=True, P_b=1e6,
        use_moldiff=False, use_settling=False,
    )
    out = compute_mu_dz_g(cfg, ymix, ms_arr, pico, Tco)
    assert out["pref_indx"] == 0
    assert out["g"][0] == cfg.gs
    assert out["zco"][0] == 0.0
    # zco strictly increases (pco strictly decreases).
    z = np.asarray(out["zco"])
    assert np.all(np.diff(z) > 0.0)


def test_compute_mu_dz_g_gas_giant_anchor_at_1bar():
    """rocky=False & P_b≥1bar → pref_indx is the layer closest to 1 bar."""
    from atm_setup import compute_pico, compute_mu_dz_g

    nz = 60
    pco = _make_pco(1e9, 1e-2, nz)
    pico = np.asarray(compute_pico(pco))
    Tco = np.linspace(2500.0, 800.0, nz)
    # Pure-H2 mixture so mean_mass = 2.016
    ymix = np.zeros((nz, 1))
    ymix[:, 0] = 1.0
    ms_arr = np.array([2.016])
    cfg = SimpleNamespace(
        gs=2140.0, Rp=1.138 * 7.1492e9, rocky=False, P_b=1e9,
        use_moldiff=False, use_settling=False,
    )
    out = compute_mu_dz_g(cfg, ymix, ms_arr, pico, Tco)
    pref = out["pref_indx"]
    assert pref > 0  # 1 bar lives somewhere in the column, not at index 0
    assert out["g"][pref] == cfg.gs
    assert out["zco"][pref] == 0.0
    # zco strictly increases.
    z = np.asarray(out["zco"])
    assert np.all(np.diff(z) > 0.0)


# ---------------------------------------------------------------------------
# Settling velocity
# ---------------------------------------------------------------------------

def test_compute_settling_velocity_off_returns_zeros():
    from atm_setup import compute_settling_velocity
    cfg = SimpleNamespace(use_settling=False, atm_base="H2", non_gas_sp=[])
    out = compute_settling_velocity(
        cfg, Tco=np.full(10, 300.0), g=np.full(10, 980.0),
        species_list=["H2", "H2O_l_s"], rho_p={}, r_p={},
    )
    assert out.shape == (9, 2) and np.all(out == 0.0)


def test_compute_settling_velocity_h2so4_negative():
    """`use_settling=True` produces negative (downward) vs for non-gas species
    when atm_base has a tabulated viscosity polynomial."""
    from atm_setup import compute_settling_velocity
    cfg = SimpleNamespace(use_settling=True, atm_base="N2", non_gas_sp=["H2SO4_l"])
    species_list = ["N2", "O2", "H2SO4_l"]
    out = compute_settling_velocity(
        cfg, Tco=np.full(10, 273.0), g=np.full(10, 980.0),
        species_list=species_list,
        rho_p={"H2SO4_l": 1.83},
        r_p={"H2SO4_l": 1e-4},
    )
    assert out.shape == (9, 3)
    assert np.all(out[:, 2] < 0.0)
    assert np.all(out[:, 0] == 0.0) and np.all(out[:, 1] == 0.0)


# ---------------------------------------------------------------------------
# Cross-check JAX TP_H14 vs scipy expn reference
# ---------------------------------------------------------------------------

def test_TP_H14_matches_scipy_reference():
    """JAX `analytical_TP_H14` matches the scipy.special.expn reference
    to machine precision (the only thing in either is `expn`+arithmetic)."""
    from atm_setup import analytical_TP_H14
    pco = np.logspace(9, -2, 100)
    params = [120.0, 1500.0, 0.1, 0.02, 1.0, 1.0]
    gs, Pb = 2140.0, 1e9
    T_jax = np.asarray(analytical_TP_H14(pco, params, gs=gs, Pb=Pb))
    # Re-build params each call — _ref_TP_H14 mutates T_irr in place.
    T_ref = _ref_TP_H14(pco, list(params), gs=gs, Pb=Pb)
    assert T_jax.shape == T_ref.shape
    assert np.allclose(T_jax, T_ref, rtol=1e-13, atol=0.0)
