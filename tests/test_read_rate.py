"""Phase 22a-1 unit tests for the new entry points in `rates.py`.

Covers:
  - `apply_lowT_caps`: synthetic Tco that fires all three caps; check the
    cap values match the Moses+2005 formulas hardcoded in
    `legacy_io.ReadRate.lim_lowT_rates`.
  - `apply_remove_list`: literal zero of the indices passed in, no auto-zero
    of the paired forward/reverse partner.
  - `build_rate_array` end-to-end vs the legacy
    `read_rate -> rev_rate -> remove_rate` chain on the HD189 reference
    state (uses `hd189_state` fixture; no VULCAN-master sibling required
    because the legacy chain lives in vendored `legacy_io`).

Run from VULCAN-JAX/:
    pytest tests/test_read_rate.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_rxn_idx(net, eq: str) -> int:
    """Return the parser-i (1-based) of a forward reaction by equation text."""
    for i in range(1, net.nr + 1, 2):
        if net.Rf.get(i, "") == eq:
            return i
    raise AssertionError(f"reaction {eq!r} not found in network")


def _load_nasa9_local(net):
    """Load NASA-9 coefficients fresh.

    We deliberately don't read `nasa9_coeffs` because the suite
    has a known module-state ordering issue (see the Phase 16 conftest
    note): tests that pop `chem_funs` from `sys.modules` can re-resolve it
    to upstream's SymPy-generated `chem_funs.py`, which doesn't expose the
    JAX-side private attribute. Loading directly via `gibbs.load_nasa9`
    avoids the question.
    """
    import gibbs
    import vulcan_cfg
    from pathlib import Path

    thermo_dir = Path(vulcan_cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = ROOT / "thermo"
    coeffs, _present = gibbs.load_nasa9(net.species, thermo_dir)
    return coeffs


# ---------------------------------------------------------------------------
# apply_lowT_caps
# ---------------------------------------------------------------------------

def test_lowT_caps_fire_all_three():
    """All 3 caps should fire below their thresholds with the exact Moses+2005
    formulas. Above the thresholds, k stays untouched."""
    import network as net_mod
    import rates
    import vulcan_cfg

    net = net_mod.parse_network(vulcan_cfg.network)

    i_ch3 = _find_rxn_idx(net, "H + CH3 + M -> CH4 + M")
    i_c2h4 = _find_rxn_idx(net, "H + C2H4 + M -> C2H5 + M")
    i_c2h5 = _find_rxn_idx(net, "H + C2H5 + M -> C2H6 + M")

    # 4-layer atm spanning all four temperature regimes:
    #   T=150 -> CH3 cap (T<=277.5), C2H4 cap (T<=300), C2H5 cap (T<=200)
    #   T=250 -> CH3 cap, C2H4 cap, no C2H5 cap (T>200)
    #   T=290 -> no CH3 cap (T>277.5), C2H4 cap, no C2H5 cap
    #   T=350 -> no caps fire
    T = np.array([150.0, 250.0, 290.0, 350.0], dtype=np.float64)
    M = np.array([1e18, 1e16, 1e14, 1e12], dtype=np.float64)

    # Seed k with an obviously-non-cap value so we can detect untouched layers.
    nz = T.shape[0]
    k_in = np.full((net.nr + 1, nz), 7.77e-12, dtype=np.float64)
    k_in[0] = 0.0  # row 0 unused (1-based indexing)

    k_out = rates.apply_lowT_caps(net, k_in, T, M)

    # Reference values straight out of legacy_io.lim_lowT_rates (Moses+2005).
    k0 = 6.0e-29
    kinf = 2.06e-10 * T**-0.4
    cap_ch3 = k0 / (1.0 + k0 * M / kinf)
    cap_c2h4 = 3.7e-30
    cap_c2h5 = 2.49e-27

    # CH3 row: cap below 277.5, untouched above.
    assert k_out[i_ch3, 0] == pytest.approx(cap_ch3[0], rel=0, abs=0)
    assert k_out[i_ch3, 1] == pytest.approx(cap_ch3[1], rel=0, abs=0)
    assert k_out[i_ch3, 2] == pytest.approx(7.77e-12)  # T=290 > 277.5, untouched
    assert k_out[i_ch3, 3] == pytest.approx(7.77e-12)  # T=350 > 277.5, untouched

    # C2H4 row: cap below 300, untouched at 350.
    assert k_out[i_c2h4, 0] == cap_c2h4
    assert k_out[i_c2h4, 1] == cap_c2h4
    assert k_out[i_c2h4, 2] == cap_c2h4  # T=290 <= 300, capped
    assert k_out[i_c2h4, 3] == pytest.approx(7.77e-12)

    # C2H5 row: cap below 200, untouched above.
    assert k_out[i_c2h5, 0] == cap_c2h5
    assert k_out[i_c2h5, 1] == pytest.approx(7.77e-12)  # T=250 > 200, untouched
    assert k_out[i_c2h5, 2] == pytest.approx(7.77e-12)
    assert k_out[i_c2h5, 3] == pytest.approx(7.77e-12)

    # Other forward rows untouched.
    other_idx = [j for j in range(1, net.nr + 1, 2)
                 if j not in (i_ch3, i_c2h4, i_c2h5)]
    for j in other_idx[:5]:
        assert np.array_equal(k_out[j], k_in[j])


def test_lowT_caps_no_op_when_T_above_thresholds():
    """If T > 300 K everywhere, no cap fires and k is unchanged."""
    import network as net_mod
    import rates
    import vulcan_cfg

    net = net_mod.parse_network(vulcan_cfg.network)
    nz = 5
    T = np.full(nz, 1500.0, dtype=np.float64)   # HD189-like
    M = np.full(nz, 1e15, dtype=np.float64)

    k_in = np.random.RandomState(0).uniform(1e-15, 1e-10, size=(net.nr + 1, nz))
    k_out = rates.apply_lowT_caps(net, k_in, T, M)
    assert np.array_equal(k_out, k_in)


# ---------------------------------------------------------------------------
# apply_remove_list
# ---------------------------------------------------------------------------

def test_remove_list_zeros_only_listed_indices():
    """`apply_remove_list` zeros literally the indices in `remove_list` and
    nothing else (does not auto-zero the paired forward/reverse partner)."""
    import network as net_mod
    import rates
    import vulcan_cfg

    net = net_mod.parse_network(vulcan_cfg.network)
    nz = 4
    rng = np.random.RandomState(1)
    k_in = rng.uniform(1.0, 2.0, size=(net.nr + 1, nz))
    k_in[0] = 0.0

    # Lone forward (1 only); pair (3, 4); lone reverse (6 only).
    k_out = rates.apply_remove_list(net, k_in, [1, 3, 4, 6])

    assert np.all(k_out[1] == 0.0)
    assert np.all(k_out[3] == 0.0)
    assert np.all(k_out[4] == 0.0)
    assert np.all(k_out[6] == 0.0)
    # Their unmentioned partners stay nonzero -- this is the literal-only rule.
    assert np.array_equal(k_out[2], k_in[2])
    assert np.array_equal(k_out[5], k_in[5])
    assert np.array_equal(k_out[7], k_in[7])


def test_remove_list_none_or_empty_is_noop():
    import network as net_mod
    import rates
    import vulcan_cfg

    net = net_mod.parse_network(vulcan_cfg.network)
    k_in = np.ones((net.nr + 1, 3), dtype=np.float64)
    assert np.array_equal(rates.apply_remove_list(net, k_in, None), k_in)
    assert np.array_equal(rates.apply_remove_list(net, k_in, []), k_in)


# ---------------------------------------------------------------------------
# build_rate_array end-to-end
# ---------------------------------------------------------------------------

def test_build_rate_array_matches_legacy_hd189(hd189_state):
    """End-to-end: `build_rate_array` should match the legacy
    `read_rate -> rev_rate -> remove_rate` chain bit-exactly on HD189
    (HD189 has `use_lowT_limit_rates=False`, `remove_list=[]`).

    The conftest fixture also runs `compute_J` which overwrites the photo
    rate slots with computed J-rates -- those live in a separate code path
    that Phase 22b will rewrite. Mask those (and ion / conden /
    radiative slots) from this comparison; `build_rate_array` is the
    chemistry-rate half only.
    """
    import network as net_mod
    import rates
    import vulcan_cfg

    net = net_mod.parse_network(vulcan_cfg.network)
    nasa9_coeffs = _load_nasa9_local(net)
    k_jax = rates.build_rate_array(
        vulcan_cfg, net, hd189_state.atm, nasa9_coeffs
    )

    nr = int(net.nr)
    nz = int(hd189_state.atm.Tco.shape[0])
    # Phase 22d: hd189_state.var carries the dense rate array on `k_arr`
    # (the legacy `var.k` dict was retired). Compare against that.
    k_legacy = np.asarray(hd189_state.var.k_arr, dtype=np.float64)
    if k_legacy.shape != (nr + 1, nz):
        raise AssertionError(
            f"hd189_state.var.k_arr shape {k_legacy.shape} != ({nr+1}, {nz})"
        )

    # Reactions whose rate row is set outside the read_rate -> rev_rate ->
    # remove_rate chain (and which Phase 22a does not own).
    skip_mask = np.zeros(nr + 1, dtype=bool)
    for i in range(1, nr + 1):
        if (net.is_photo[i] or net.is_ion[i]
                or net.is_conden[i] or net.is_radiative[i]):
            skip_mask[i] = True

    rows = np.flatnonzero(~skip_mask)
    rows = rows[rows >= 1]  # drop the unused 0-row

    diff = np.abs(k_jax[rows] - k_legacy[rows])
    denom = np.maximum(np.abs(k_legacy[rows]), 1e-300)
    relerr = diff / denom
    relerr = np.where(np.abs(k_legacy[rows]) < 1e-300, diff, relerr)

    max_relerr = float(relerr.max())
    worst_local = int(np.unravel_index(np.argmax(relerr), relerr.shape)[0])
    worst = int(rows[worst_local])
    print(f"build_rate_array vs legacy: max relerr = {max_relerr:.3e} "
          f"(at parser-i = {worst}, Rf = {net.Rf.get(worst, '?')!r}); "
          f"compared {rows.size} reactions; skipped "
          f"{int(skip_mask.sum())} photo/ion/conden/radiative rows")
    assert max_relerr <= 1e-13, (
        f"build_rate_array deviates from legacy at i={worst} "
        f"(Rf={net.Rf.get(worst, '?')!r}): relerr={max_relerr:.3e}"
    )


def test_build_rate_array_with_lowT_caps(hd189_state):
    """With `use_lowT_limit_rates=True`, the resulting forward array should
    have the three cap reactions clamped at layers where T is low. HD189's
    upper atmosphere has Tco ~ 200-700 K, which fires at least the C2H4 cap
    (T<=300) for the cooler upper layers."""
    import network as net_mod
    import rates
    import vulcan_cfg

    net = net_mod.parse_network(vulcan_cfg.network)
    nasa9_coeffs = _load_nasa9_local(net)

    # Run with caps off (baseline) and on (capped).
    cfg_off = type("CfgOff", (), {
        "use_lowT_limit_rates": False,
        "remove_list": list(getattr(vulcan_cfg, "remove_list", [])),
    })()
    cfg_on = type("CfgOn", (), {
        "use_lowT_limit_rates": True,
        "remove_list": list(getattr(vulcan_cfg, "remove_list", [])),
    })()

    k_off = rates.build_rate_array(
        cfg_off, net, hd189_state.atm, nasa9_coeffs
    )
    k_on = rates.build_rate_array(
        cfg_on, net, hd189_state.atm, nasa9_coeffs
    )

    i_c2h4 = _find_rxn_idx(net, "H + C2H4 + M -> C2H5 + M")
    T = np.asarray(hd189_state.atm.Tco)

    # Where T <= 300, k_on should equal exactly 3.7e-30 (post-Lindemann
    # cap value); elsewhere unchanged.
    cap_mask = T <= 300.0
    if cap_mask.any():
        np.testing.assert_array_equal(
            k_on[i_c2h4, cap_mask],
            np.full(cap_mask.sum(), 3.7e-30),
        )
    np.testing.assert_array_equal(k_on[i_c2h4, ~cap_mask], k_off[i_c2h4, ~cap_mask])
