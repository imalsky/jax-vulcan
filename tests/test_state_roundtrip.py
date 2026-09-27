"""Typed pre-loop pytree schema.

`pytree_from_store(var, atm)` must carry every runner-visible field, and
the RunState-backed `.vul` writer must expose VULCAN-master's parameter
keys. Pins the schema so future setup-pipeline edits cannot silently drop
a field.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def test_roundtrip_field_set_complete(hd189_state):
    """The pytree must capture all fields needed for the runner. This
    test enumerates the AtmData attribute names on the HD189 reference
    state and asserts each runner-visible array attribute is present in
    the AtmInputs slot."""
    from vulcan_jax.state import pytree_from_store, AtmInputs

    pt = pytree_from_store(hd189_state.var, hd189_state.atm)

    # Every AtmInputs field must have a non-None / non-empty backing.
    for name in AtmInputs._fields:
        leaf = getattr(pt.atm, name)
        assert leaf is not None, f"AtmInputs.{name} is None"
        arr = np.asarray(leaf)
        assert arr.size > 0 or name in {
            "top_flux",
            "bot_flux",
            "bot_vdep",
            "bot_fix_sp",
        }, f"AtmInputs.{name} unexpectedly empty (shape={arr.shape})"

    # Rate constants: densified (nr+1, nz) array, non-empty.
    k_arr = np.asarray(pt.rate.k)
    assert k_arr.ndim == 2, f"rate.k must be 2D, got {k_arr.shape}"
    assert k_arr.shape[1] == hd189_state.atm.Tco.shape[0], (
        f"rate.k second dim must be nz, got {k_arr.shape}"
    )

    # PhotoInputs: sflux_top is populated when use_photo=True.
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    if bool(getattr(vulcan_cfg, "use_photo", False)):
        sflux_arr = np.asarray(pt.photo.sflux_top)
        assert sflux_arr.size > 0, (
            "PhotoInputs.sflux_top empty even though use_photo=True"
        )


def test_runstate_output_parameter_schema(hd189_state):
    """RunState-backed `.vul` output exposes VULCAN-master parameter keys."""
    import vulcan_jax.legacy_io as legacy_io
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    from vulcan_jax.state import runstate_from_store

    rs = runstate_from_store(hd189_state.var, hd189_state.atm, hd189_state.para)
    _, _, param = legacy_io._synthesize_save_dicts(
        rs._replace(photo_static=getattr(hd189_state.solver, "_photo_static", None)),
        vulcan_cfg,
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
