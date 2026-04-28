"""Phase 22e: validate `photo_setup.build_photo_static` against the
checked-in npz fixtures generated from the Phase 22b legacy oracle.

The fixtures (`tests/data/photo_setup_hd189_baseline.npz` and
`tests/data/photo_setup_hd189_T_dep.npz`) were captured by
`tests/_gen_photo_baseline.py` on the green pre-22e tree and are the
only oracle for the photo cross-section preprocessing now that
`legacy_io.ReadRate.make_bins_read_cross` is retired.

Tolerance is 0.0 abs/rel err — the static is derived from the same
NumPy CSV pipeline as the fixture, so any drift is a real bug.

Run from VULCAN-JAX/:
    pytest tests/test_photo_setup.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

FIXTURE_DIR = ROOT / "tests" / "data"


def _build_state_through_read_rate():
    """Run the pre-photo VULCAN setup and return (var, atm)."""
    import legacy_io as op
    import store
    from atm_setup import Atm

    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    data_var = op.ReadRate().read_rate(data_var, data_atm)
    return data_var, data_atm


def _check_static_against_fixture(static, fixture_path: Path) -> None:
    """Bit-exact compare every dense array in `static` to its fixture entry."""
    fx = np.load(fixture_path)

    np.testing.assert_array_equal(np.asarray(static.bins), fx["bins"])
    assert int(static.nbin) == int(fx["nbin"])
    assert float(static.dbin1) == float(fx["dbin1"])
    assert float(static.dbin2) == float(fx["dbin2"])

    for i, sp in enumerate(static.absp_sp):
        np.testing.assert_array_equal(
            np.asarray(static.absp_cross[i]),
            fx[f"cross__{sp}"],
            err_msg=f"absp_cross[{i}] (sp={sp}) mismatch",
        )

    for i, sp in enumerate(static.absp_T_sp):
        np.testing.assert_array_equal(
            np.asarray(static.absp_T_cross[i]),
            fx[f"cross_T__{sp}"],
            err_msg=f"absp_T_cross[{i}] (sp={sp}) mismatch",
        )

    for i, (sp, br) in enumerate(static.branch_keys):
        np.testing.assert_array_equal(
            np.asarray(static.cross_J[i]),
            fx[f"cross_J__{sp}__{br}"],
            err_msg=f"cross_J[{i}] (sp={sp}, br={br}) mismatch",
        )

    for i, (sp, br) in enumerate(static.branch_T_keys):
        np.testing.assert_array_equal(
            np.asarray(static.cross_J_T[i]),
            fx[f"cross_J_T__{sp}__{br}"],
            err_msg=f"cross_J_T[{i}] (sp={sp}, br={br}) mismatch",
        )

    for i, sp in enumerate(static.scat_sp):
        np.testing.assert_array_equal(
            np.asarray(static.scat_cross[i]),
            fx[f"cross_scat__{sp}"],
            err_msg=f"scat_cross[{i}] (sp={sp}) mismatch",
        )

    for i, (sp, br) in enumerate(static.ion_branch_keys):
        np.testing.assert_array_equal(
            np.asarray(static.cross_Jion[i]),
            fx[f"cross_Jion__{sp}__{br}"],
            err_msg=f"cross_Jion[{i}] (sp={sp}, br={br}) mismatch",
        )


def test_photo_setup_matches_baseline_fixture():
    """HD189 default — T_cross_sp=[], use_ion=False."""
    import photo_setup
    import vulcan_cfg

    if not bool(getattr(vulcan_cfg, "use_photo", False)):
        import pytest
        pytest.skip("use_photo=False; nothing to compare.")

    var, atm = _build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)
    _check_static_against_fixture(
        static, FIXTURE_DIR / "photo_setup_hd189_baseline.npz",
    )


def test_photo_setup_matches_T_dep_fixture():
    """HD189 with T_cross_sp=['CO2','H2O','NH3'] patched on."""
    import photo_setup
    import vulcan_cfg

    saved_T_cross_sp = list(vulcan_cfg.T_cross_sp)
    vulcan_cfg.T_cross_sp = ["CO2", "H2O", "NH3"]
    try:
        var, atm = _build_state_through_read_rate()
        static = photo_setup._build_photo_static_dense(var, atm)
        _check_static_against_fixture(
            static, FIXTURE_DIR / "photo_setup_hd189_T_dep.npz",
        )
    finally:
        vulcan_cfg.T_cross_sp = saved_T_cross_sp
