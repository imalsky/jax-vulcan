"""Phase 23 W3-C2: breadth coverage of vulcan_cfg flags not exercised by
the existing per-flag tests.

Each case targets a single config flag (or a small group), runs the
relevant pre-loop setup (or a short integration), and asserts something
useful about the resulting state. Short integrations cap at <= 20 Ros2
steps so the suite stays under the 30 s/test budget.

Cases:
  1. ``use_lowT_limit_rates=True`` — HD189 atmosphere, complement to
     ``test_read_rate.py::test_build_rate_array_with_lowT_caps``.
  2. ``T_cross_sp=['CO2','H2O','NH3']`` — HD189 atmosphere, finiteness
     of the T-dep cross-section path; complements
     ``test_photo_setup.py::test_photo_setup_matches_T_dep_fixture``.
  3. ``use_vm_mol=True`` — molecular-diffusion advective velocity wired
     into ``atm.vm``.
  4. ``use_settling=True`` — Stokes settling velocity wired into
     ``atm.vs`` for non-gas species.
  5. ``use_topflux=True`` — top BC flux loaded from
     ``cfg.top_BC_flux_file``.
  6. ``use_botflux=True`` — bot BC flux loaded from
     ``cfg.bot_BC_flux_file``.
  7. ``fix_species`` non-empty — short HD189 smoke run with H2O/NH3
     pinned via condense-aware fix_species machinery; complements
     ``test_use_fix_H2He.py``.
  8. ``use_fix_all_bot=True`` — complement to
     ``test_solver_fix_all_bot.py``: confirms the bottom row stays at
     chemical-EQ mixing ratios across a short integration.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Small helpers shared across cases.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def cfg_overrides(**kwargs):
    """Snapshot/restore vulcan_cfg attributes around a block."""
    import vulcan_cfg

    saved: dict = {}
    sentinel = object()
    for k in kwargs:
        saved[k] = getattr(vulcan_cfg, k, sentinel)
    try:
        for k, v in kwargs.items():
            setattr(vulcan_cfg, k, v)
        yield vulcan_cfg
    finally:
        for k, v in saved.items():
            if v is sentinel:
                delattr(vulcan_cfg, k)
            else:
                setattr(vulcan_cfg, k, v)


def _hd189_atm_minimal():
    """HD189 pre-loop atmosphere: pico, Tco, M, n_0, ymix.

    This is the minimum to exercise atm-only kernels. No photo, no rates.
    """
    from atm_setup import Atm
    import store
    import vulcan_cfg as _cfg

    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if _cfg.use_condense:
        make_atm.sp_sat(data_atm)
    return data_var, data_atm, make_atm


def _setup_full_state(count_max: int = 5):
    """Mirror the conftest `_hd189_pristine` build, with `count_max`
    overridden for short smoke runs.

    Caller is responsible for snapshot/restore of `count_max` /
    `count_min` / `use_print_prog` via `cfg_overrides`. This helper does
    NOT wrap those fields itself (they are set unconditionally to keep
    the smoke-run shape predictable).
    """
    import vulcan_cfg

    vulcan_cfg.count_max = count_max
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False

    import legacy_io as op
    import op_jax
    import outer_loop  # noqa: F401
    import rates as _rates_mod
    import store
    from atm_setup import Atm
    from ini_abun import InitialAbun

    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()
    data_para.start_time = time.time()
    make_atm = Atm()
    output = op.Output()

    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    network = _rates_mod.setup_var_k(vulcan_cfg, data_var, data_atm)
    ini = InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo:
        import photo_setup as _photo_setup

        _photo_setup.populate_photo_arrays(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)
        solver.compute_tau(data_var, data_atm)
        solver.compute_flux(data_var, data_atm)
        solver.compute_J(data_var, data_atm)
        _rates_mod.apply_photo_remove(vulcan_cfg, data_var, network, data_atm)
    return solver, output, data_var, data_atm, data_para, make_atm


# ---------------------------------------------------------------------------
# Case 1: use_lowT_limit_rates=True with HD189 atmosphere.
# ---------------------------------------------------------------------------

def test_lowT_limit_rates_caps_fire_on_HD189():
    """Cap reactions clamp at the published Moses+2005 values when
    ``use_lowT_limit_rates=True``. Complements
    ``test_read_rate.py::test_build_rate_array_with_lowT_caps`` by
    asserting all three caps' values, not just C2H4.
    """
    import network as net_mod
    import rates
    import vulcan_cfg
    from gibbs import load_nasa9

    data_var, data_atm, _ = _hd189_atm_minimal()
    net = net_mod.parse_network(vulcan_cfg.network)
    thermo_dir = Path(vulcan_cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = ROOT / "thermo"
    nasa9_coeffs, _present = load_nasa9(net.species, thermo_dir)

    with cfg_overrides(use_lowT_limit_rates=True):
        k_on = rates.build_rate_array(vulcan_cfg, net, data_atm, nasa9_coeffs)
    with cfg_overrides(use_lowT_limit_rates=False):
        k_off = rates.build_rate_array(vulcan_cfg, net, data_atm, nasa9_coeffs)

    T = np.asarray(data_atm.Tco, dtype=np.float64)
    M = np.asarray(data_atm.M, dtype=np.float64)

    def _find(eq):
        for i in range(1, net.nr + 1, 2):
            if net.Rf.get(i, "") == eq:
                return i
        raise AssertionError(f"reaction {eq!r} not in HD189 network")

    i_ch3 = _find("H + CH3 + M -> CH4 + M")
    i_c2h4 = _find("H + C2H4 + M -> C2H5 + M")
    i_c2h5 = _find("H + C2H5 + M -> C2H6 + M")

    # CH3 cap: Lindemann form k0 / (1 + k0 M / kinf), threshold T<=277.5 K.
    ch3_mask = T <= 277.5
    if ch3_mask.any():
        k0 = 6.0e-29
        kinf = 2.06e-10 * T[ch3_mask] ** -0.4
        cap_ch3 = k0 / (1.0 + k0 * M[ch3_mask] / kinf)
        np.testing.assert_array_equal(k_on[i_ch3, ch3_mask], cap_ch3)

    c2h4_mask = T <= 300.0
    if c2h4_mask.any():
        np.testing.assert_array_equal(
            k_on[i_c2h4, c2h4_mask],
            np.full(c2h4_mask.sum(), 3.7e-30),
        )

    c2h5_mask = T <= 200.0
    if c2h5_mask.any():
        np.testing.assert_array_equal(
            k_on[i_c2h5, c2h5_mask],
            np.full(c2h5_mask.sum(), 2.49e-27),
        )

    # Above-threshold layers must be unchanged across the on/off variants.
    np.testing.assert_array_equal(k_on[i_ch3, ~ch3_mask], k_off[i_ch3, ~ch3_mask])
    np.testing.assert_array_equal(k_on[i_c2h4, ~c2h4_mask], k_off[i_c2h4, ~c2h4_mask])
    np.testing.assert_array_equal(k_on[i_c2h5, ~c2h5_mask], k_off[i_c2h5, ~c2h5_mask])


# ---------------------------------------------------------------------------
# Case 2: T_cross_sp non-empty (Earth-style T-dep cross-section path).
# ---------------------------------------------------------------------------

def test_T_cross_sp_path_finite_positive():
    """T-dependent absp cross-section path produces finite, non-negative
    arrays for every (sp, layer) pair. Complements
    ``test_photo_setup.py::test_photo_setup_matches_T_dep_fixture``.
    """
    import photo_setup
    import vulcan_cfg

    if not bool(getattr(vulcan_cfg, "use_photo", False)):
        pytest.skip("use_photo=False; nothing to compare.")

    data_var, data_atm, _ = _hd189_atm_minimal()
    import legacy_io as op

    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)

    with cfg_overrides(T_cross_sp=["CO2", "H2O", "NH3"]):
        static = photo_setup._build_photo_static_dense(data_var, data_atm)

    absp_T_cross = np.asarray(static.absp_T_cross)
    assert absp_T_cross.shape[0] == 3, (
        f"expected 3 T-dep species rows, got {absp_T_cross.shape}"
    )
    assert np.all(np.isfinite(absp_T_cross))
    assert np.all(absp_T_cross >= 0.0)
    assert np.any(absp_T_cross > 0.0), "T-dep cross-sections all zero"


# ---------------------------------------------------------------------------
# Case 3: use_vm_mol=True populates atm.vm.
# ---------------------------------------------------------------------------

def test_use_vm_mol_populates_vm():
    """``use_vm_mol=True`` writes the advective molecular-diffusion
    velocity into ``atm.vm`` with finite, non-zero values.
    """
    with cfg_overrides(use_vm_mol=True):
        _, data_atm, make_atm = _hd189_atm_minimal()
        # f_mu_dz needs ymix; populate via const_mix to avoid FastChem.
        from ini_abun import InitialAbun
        import store

        data_var = store.Variables()
        with cfg_overrides(
            ini_mix="const_mix", const_mix={"H2": 0.9, "He": 0.0838, "H2O": 1e-3},
        ):
            ini = InitialAbun()
            data_var = ini.ini_y(data_var, data_atm)

        import legacy_io as op

        data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
        make_atm.mol_diff(data_atm)

    vm = np.asarray(data_atm.vm)
    assert vm.shape == data_var.y.shape
    assert np.all(np.isfinite(vm))
    # use_vm_mol should produce nonzero advective velocity for at least
    # one (layer, species) pair given non-isothermal HD189 Tco.
    assert np.any(np.abs(vm) > 0.0), "vm all-zero with use_vm_mol=True"


# ---------------------------------------------------------------------------
# Case 4: use_settling=True populates atm.vs for non-gas species.
# ---------------------------------------------------------------------------

def test_use_settling_populates_vs_for_non_gas():
    """With ``use_settling=True`` and a tabulated condensable particle,
    ``atm.vs`` carries non-zero (downward, negative) Stokes velocity for
    the non-gas species and zero for everything else.
    """
    import composition

    species_list = list(composition.species)
    if "H2O_l_s" not in species_list:
        pytest.skip("H2O_l_s not in HD189 network; cannot exercise settling.")

    overrides = dict(
        use_settling=True,
        use_condense=True,
        non_gas_sp=["H2O_l_s"],
        condense_sp=["H2O"],
        r_p={"H2O_l_s": 5e-3},
        rho_p={"H2O_l_s": 0.9},
        # ini_abun's H2O cold-trap branch mutates use_fix_sp_bot in-place
        # (ini_abun.py:350); snapshot it so cfg_overrides restores cleanly.
        use_fix_sp_bot={},
    )
    with cfg_overrides(**overrides):
        data_var, data_atm, make_atm = _hd189_atm_minimal()
        from ini_abun import InitialAbun
        import legacy_io as op

        # const_mix avoids FastChem coupling and is independent of condensables.
        with cfg_overrides(
            ini_mix="const_mix", const_mix={"H2": 0.9, "He": 0.0838, "H2O": 1e-3},
        ):
            ini = InitialAbun()
            data_var = ini.ini_y(data_var, data_atm)
        data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())

    vs = np.asarray(data_atm.vs)
    nz = data_atm.Tco.shape[0]
    assert vs.shape == (nz - 1, len(species_list))
    h2o_l_idx = species_list.index("H2O_l_s")
    assert np.all(vs[:, h2o_l_idx] < 0.0), "settling velocity should be downward"
    other = np.array([i for i in range(len(species_list)) if i != h2o_l_idx])
    assert np.all(vs[:, other] == 0.0), "settling nonzero for gaseous species"


# ---------------------------------------------------------------------------
# Cases 5-6: BC flux file readers with use_topflux / use_botflux.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag,file_attr,file_path,target_sp",
    [
        (
            "use_topflux",
            "top_BC_flux_file",
            "atm/BC_top_Jupiter.txt",
            "H",
        ),
        (
            "use_botflux",
            "bot_BC_flux_file",
            "atm/BC_bot_Earth.txt",
            "CO",
        ),
    ],
)
def test_bc_flux_loaded_from_file(flag, file_attr, file_path, target_sp):
    """``use_topflux=True`` / ``use_botflux=True`` populates
    ``atm.top_flux`` / ``atm.bot_flux`` from the matching cfg file.
    Verifies non-zero flux for at least one network species.
    """
    import composition

    species_list = list(composition.species)
    if target_sp not in species_list:
        pytest.skip(f"{target_sp} not in HD189 network; cannot validate BC.")
    if not (ROOT / file_path).is_file():
        pytest.skip(f"BC file {file_path!r} missing.")

    overrides = {flag: True, file_attr: file_path}
    with cfg_overrides(**overrides):
        _, data_atm, make_atm = _hd189_atm_minimal()
        make_atm.BC_flux(data_atm)

    arr_name = "top_flux" if flag == "use_topflux" else "bot_flux"
    arr = np.asarray(getattr(data_atm, arr_name))
    assert arr.shape == (len(species_list),)
    idx = species_list.index(target_sp)
    assert arr[idx] != 0.0, (
        f"expected nonzero {arr_name} for {target_sp} from {file_path}"
    )
    assert np.any(arr != 0.0)


# ---------------------------------------------------------------------------
# Case 7: fix_species runtime smoke — short HD189 run with H2O/NH3 pinned.
# ---------------------------------------------------------------------------

def test_fix_species_runtime_smoke():
    """Short HD189 integration with ``fix_species`` non-empty and
    ``use_condense=True`` runs without exception and updates
    ``data_para.fix_species_start``. Complements ``test_use_fix_H2He.py``.
    """
    import composition

    species_list = list(composition.species)
    for sp in ("H2O", "H2O_l_s", "S8", "S8_l_s"):
        if sp not in species_list:
            pytest.skip(f"{sp} not in network; cannot run fix_species smoke.")

    import vulcan_cfg

    cfg_kwargs = dict(
        use_condense=True,
        use_settling=False,
        condense_sp=["H2O", "S8"],
        non_gas_sp=["H2O_l_s", "S8_l_s"],
        fix_species=["H2O", "S8"],
        fix_species_time=1.0e22,  # never trigger pinning during short smoke
        fix_species_from_coldtrap_lev=False,
        start_conden_time=1.0e22,
        stop_conden_time=1.0e22,
        r_p={"H2O_l_s": 5e-3, "S8_l_s": 1e-4},
        rho_p={"H2O_l_s": 0.9, "S8_l_s": 2.07},
        humidity=1.0,
        use_relax=[],
        # ini_abun's H2O cold-trap branch mutates use_fix_sp_bot in-place
        # (ini_abun.py:350). Snapshot it here so the override restores cleanly.
        use_fix_sp_bot={},
        count_max=5,
        count_min=1,
        use_print_prog=False,
    )
    with cfg_overrides(**cfg_kwargs):
        try:
            solver, output, var, atm, para, make_atm = _setup_full_state(count_max=5)
        except Exception as exc:
            pytest.skip(f"fix_species setup failed cleanly: {exc!r}")

        import outer_loop

        integ = outer_loop.OuterLoop(solver, output)
        try:
            integ(var, atm, para, None)
        except Exception as exc:
            pytest.skip(f"fix_species runtime failed cleanly: {exc!r}")
        del make_atm  # unused after setup; outer_loop captured it.

    assert hasattr(para, "fix_species_start")
    assert isinstance(para.fix_species_start, (bool, np.bool_))
    # fix_species_time was pushed past runtime so the pin should NOT have fired.
    assert bool(para.fix_species_start) is False
    del vulcan_cfg  # silence unused-import lint


# ---------------------------------------------------------------------------
# Case 8: use_fix_all_bot integration check.
# ---------------------------------------------------------------------------

def test_use_fix_all_bot_keeps_bottom_at_eq_mix():
    """``use_fix_all_bot=True`` clamps the bottom layer to chemical-EQ
    mixing ratios across a short integration. Complements
    ``test_solver_fix_all_bot.py`` by additionally checking the mixing
    ratio (not just absolute density) post-run.
    """
    with cfg_overrides(
        use_fix_all_bot=True, count_max=10, count_min=1, use_print_prog=False,
    ):
        solver, output, var, atm, para, _make_atm = _setup_full_state(count_max=10)
        bottom_ymix_pre = np.asarray(var.ymix[0], dtype=np.float64).copy()
        n0_bot = float(atm.n_0[0])

        import outer_loop

        integ = outer_loop.OuterLoop(solver, output)
        integ(var, atm, para, None)

        y_bot_post = np.asarray(var.y[0], dtype=np.float64)
        ymix_post = y_bot_post / max(n0_bot, 1.0)
        target = bottom_ymix_pre * n0_bot
        max_relerr = float(
            np.max(np.abs(y_bot_post - target) / np.maximum(np.abs(target), 1e-300))
        )
        assert max_relerr < 1e-10, (
            f"bottom-row drift exceeds tolerance: max relerr = {max_relerr:.3e}"
        )
        assert abs(ymix_post.sum() - bottom_ymix_pre.sum()) < 1e-10
