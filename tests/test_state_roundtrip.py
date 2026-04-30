"""Typed pre-loop pytree round-trip.

Asserts that `pytree_from_store(var, atm)` produces a `RunState` whose
re-write via `apply_pytree_to_store` and re-read via
`pytree_from_store` yields a tree-equal pytree. Pins the schema so
future setup-pipeline edits cannot silently drop a field.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def _equal_leaf(a, b) -> bool:
    """Bit-exact equality on a single leaf (handles both float scalars
    and ndarrays / jax arrays)."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return False
    if a_arr.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(a_arr, b_arr, equal_nan=True))
    return bool(np.array_equal(a_arr, b_arr))


def _pytree_equal(p, q) -> bool:
    leaves_p, treedef_p = jax.tree_util.tree_flatten(p)
    leaves_q, treedef_q = jax.tree_util.tree_flatten(q)
    if treedef_p != treedef_q:
        return False
    if len(leaves_p) != len(leaves_q):
        return False
    for la, lb in zip(leaves_p, leaves_q):
        if not _equal_leaf(la, lb):
            return False
    return True


def test_roundtrip_hd189(hd189_state):
    """`pytree_from_store -> apply_pytree_to_store -> pytree_from_store`
    is identity on the canonical HD189 reference state."""
    from state import pytree_from_store, apply_pytree_to_store

    var = hd189_state.var
    atm = hd189_state.atm

    # Snapshot once.
    pt_a = pytree_from_store(var, atm)
    # Round-trip through the legacy containers and snapshot again.
    apply_pytree_to_store(pt_a, var, atm)
    pt_b = pytree_from_store(var, atm)

    assert _pytree_equal(pt_a, pt_b), (
        "RunState round-trip is lossy — a field is being dropped or "
        "perturbed by apply_pytree_to_store / pytree_from_store. Check "
        "for missing arrays in state.AtmInputs / RateInputs / PhotoInputs."
    )


def test_roundtrip_field_set_complete(hd189_state):
    """The pytree must capture all fields needed for the runner. This
    test enumerates the AtmData attribute names on the HD189 reference
    state and asserts each runner-visible array attribute is present in
    the AtmInputs slot."""
    from state import pytree_from_store, AtmInputs

    pt = pytree_from_store(hd189_state.var, hd189_state.atm)

    # Every AtmInputs field must have a non-None / non-empty backing.
    for name in AtmInputs._fields:
        leaf = getattr(pt.atm, name)
        assert leaf is not None, f"AtmInputs.{name} is None"
        arr = np.asarray(leaf)
        assert arr.size > 0 or name in {"top_flux", "bot_flux", "bot_vdep", "bot_fix_sp"}, (
            f"AtmInputs.{name} unexpectedly empty (shape={arr.shape})"
        )

    # Rate constants — densified (nr+1, nz) array, non-empty.
    k_arr = np.asarray(pt.rate.k)
    assert k_arr.ndim == 2, f"rate.k must be 2D, got {k_arr.shape}"
    assert k_arr.shape[1] == hd189_state.atm.Tco.shape[0], (
        f"rate.k second dim must be nz, got {k_arr.shape}"
    )

    # PhotoInputs — sflux_top is populated when use_photo=True.
    import vulcan_cfg
    if bool(getattr(vulcan_cfg, "use_photo", False)):
        sflux_arr = np.asarray(pt.photo.sflux_top)
        assert sflux_arr.size > 0, (
            "PhotoInputs.sflux_top empty even though use_photo=True"
        )


def test_runstate_output_parameter_schema(hd189_state):
    """RunState-backed `.vul` output exposes VULCAN-master parameter keys."""
    import legacy_io
    import vulcan_cfg
    from state import runstate_from_store

    rs = runstate_from_store(hd189_state.var, hd189_state.atm, hd189_state.para)
    _, _, param = legacy_io._synthesize_save_dicts(
        rs,
        vulcan_cfg,
        photo_static=getattr(hd189_state.solver, "_photo_static", None),
    )

    expected = {
        "nega_y",
        "small_y",
        "delta",
        "count",
        "nega_count",
        "loss_count",
        "delta_count",
        "end_case",
        "solver_str",
        "switch_final_photo_frq",
        "where_varies_most",
        "pic_count",
        "fix_species_start",
        "tableau20",
        "start_time",
    }
    assert expected <= set(param), f"missing parameter keys: {expected - set(param)}"
    assert param["solver_str"] == "solver"
    assert np.asarray(param["where_varies_most"]).shape == hd189_state.var.y.shape
    assert len(param["tableau20"]) == 20


def test_load_stellar_flux_no_photo():
    """`load_stellar_flux(cfg)` returns an empty payload when use_photo=False
    so callers can call it unconditionally."""
    from state import load_stellar_flux

    class _Cfg:
        use_photo = False

    flux = load_stellar_flux(_Cfg())
    assert flux.wavelength_nm.shape == (0,)
    assert flux.flux.shape == (0,)
    assert flux.def_bin_min == 0.0
    assert flux.def_bin_max == 0.0


def test_load_stellar_flux_hd189():
    """`load_stellar_flux(vulcan_cfg)` reads the HD189 stellar flux file
    and produces sane bin extents."""
    from state import load_stellar_flux
    import vulcan_cfg

    flux = load_stellar_flux(vulcan_cfg)
    assert flux.wavelength_nm.shape[0] > 100, (
        f"HD189 stellar flux file produced only "
        f"{flux.wavelength_nm.shape[0]} wavelengths"
    )
    assert flux.flux.shape == flux.wavelength_nm.shape
    assert 1.99 <= flux.def_bin_min <= 5.0  # max(lambda[0], 2.0)
    assert 100.0 <= flux.def_bin_max <= 700.0  # min(lambda[-1], 700.0)
