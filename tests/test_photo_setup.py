"""Validate `photo_setup.build_photo_static` against checked-in npz
fixtures.

The fixtures (`tests/data/photo_setup_hd189_baseline.npz` and
`tests/data/photo_setup_hd189_T_dep.npz`) were captured by
`tests/_gen_photo_baseline.py` and are the canonical oracle for the
photo cross-section preprocessing.

The fixture comparison is exact except for NumPy-version ULP drift in
`np.arange` wavelength bins and the interpolated cross sections derived
from those bins.

Run from VULCAN-JAX/:
    pytest tests/test_photo_setup.py
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

FIXTURE_DIR = ROOT / "tests" / "data"
BIN_ATOL = 2e-14
CROSS_ATOL = 1e-30


def _build_state_through_read_rate():
    """Run the pre-photo VULCAN setup and return (var, atm).

    `state._Variables` / `_AtmData` are the private legacy mutable
    containers. `photo_setup._build_photo_static_dense` reads dict attrs
    that `legacy_io.ReadRate.read_rate` writes onto `var`, so we need the
    legacy containers here rather than a `legacy_view(rs)` shim.
    """
    import legacy_io as op
    from atm_setup import Atm
    from state import _Variables, _AtmData

    data_var = _Variables()
    data_atm = _AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    data_var = op.ReadRate().read_rate(data_var, data_atm)
    return data_var, data_atm


def _check_static_against_fixture(
    static,
    fixture_path: Path,
    *,
    expected_T_sp: tuple[str, ...] = (),
) -> None:
    """Bit-exact compare every dense array in `static` to its fixture entry."""
    fx = np.load(fixture_path)

    np.testing.assert_allclose(
        np.asarray(static.bins), fx["bins"], rtol=0.0, atol=BIN_ATOL,
    )
    assert int(static.nbin) == int(fx["nbin"])
    assert float(static.dbin1) == float(fx["dbin1"])
    assert float(static.dbin2) == float(fx["dbin2"])
    assert tuple(static.absp_T_sp) == expected_T_sp
    assert tuple(static.absp_T_sp) == tuple(
        sp for sp in static.absp_sp if sp in expected_T_sp
    )
    assert {f"cross__{sp}" for sp in static.absp_sp} == {
        key for key in fx.files if key.startswith("cross__")
    }
    assert {f"cross_T__{sp}" for sp in static.absp_T_sp} == {
        key for key in fx.files if key.startswith("cross_T__")
    }
    assert {f"cross_J__{sp}__{br}" for sp, br in static.branch_keys} == {
        key for key in fx.files if key.startswith("cross_J__")
    }
    assert {f"cross_J_T__{sp}__{br}" for sp, br in static.branch_T_keys} == {
        key for key in fx.files if key.startswith("cross_J_T__")
    }
    assert {f"cross_scat__{sp}" for sp in static.scat_sp} == {
        key for key in fx.files if key.startswith("cross_scat__")
    }
    assert {f"cross_Jion__{sp}__{br}" for sp, br in static.ion_branch_keys} == {
        key for key in fx.files if key.startswith("cross_Jion__")
    }

    for i, sp in enumerate(static.absp_sp):
        np.testing.assert_allclose(
            np.asarray(static.absp_cross[i]),
            fx[f"cross__{sp}"],
            rtol=0.0,
            atol=CROSS_ATOL,
            err_msg=f"absp_cross[{i}] (sp={sp}) mismatch",
        )

    for i, sp in enumerate(static.absp_T_sp):
        np.testing.assert_allclose(
            np.asarray(static.absp_T_cross[i]),
            fx[f"cross_T__{sp}"],
            rtol=0.0,
            atol=CROSS_ATOL,
            err_msg=f"absp_T_cross[{i}] (sp={sp}) mismatch",
        )

    for i, (sp, br) in enumerate(static.branch_keys):
        np.testing.assert_allclose(
            np.asarray(static.cross_J[i]),
            fx[f"cross_J__{sp}__{br}"],
            rtol=0.0,
            atol=CROSS_ATOL,
            err_msg=f"cross_J[{i}] (sp={sp}, br={br}) mismatch",
        )

    for i, (sp, br) in enumerate(static.branch_T_keys):
        np.testing.assert_allclose(
            np.asarray(static.cross_J_T[i]),
            fx[f"cross_J_T__{sp}__{br}"],
            rtol=0.0,
            atol=CROSS_ATOL,
            err_msg=f"cross_J_T[{i}] (sp={sp}, br={br}) mismatch",
        )

    for i, sp in enumerate(static.scat_sp):
        np.testing.assert_allclose(
            np.asarray(static.scat_cross[i]),
            fx[f"cross_scat__{sp}"],
            rtol=0.0,
            atol=CROSS_ATOL,
            err_msg=f"scat_cross[{i}] (sp={sp}) mismatch",
        )

    for i, (sp, br) in enumerate(static.ion_branch_keys):
        np.testing.assert_allclose(
            np.asarray(static.cross_Jion[i]),
            fx[f"cross_Jion__{sp}__{br}"],
            rtol=0.0,
            atol=CROSS_ATOL,
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


@pytest.mark.strict_isolation
def test_photo_setup_matches_T_dep_fixture(monkeypatch):
    """HD189 with T_cross_sp=['CO2','H2O','NH3'] patched on."""
    import photo_setup
    import vulcan_cfg

    monkeypatch.setattr(vulcan_cfg, "T_cross_sp", ["CO2", "H2O", "NH3"])
    var, atm = _build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)
    _check_static_against_fixture(
        static,
        FIXTURE_DIR / "photo_setup_hd189_T_dep.npz",
        expected_T_sp=("CO2", "H2O", "NH3"),
    )
