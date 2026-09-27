"""Validate `photo_setup._build_photo_static_dense` against locally captured
npz fixtures.

The fixtures (`tests/data/photo_setup_hd189_{baseline,T_dep}.npz`) are
gitignored; `python tests/gen_fixtures.py --all` regenerates them.

The fixture comparison is exact except for NumPy-version ULP drift in
`np.arange` wavelength bins and the interpolated cross sections derived
from those bins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _gen_photo_baseline import build_state_through_read_rate

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "data"
_BASELINE_FIXTURE = FIXTURE_DIR / "photo_setup_hd189_baseline.npz"
_T_DEP_FIXTURE = FIXTURE_DIR / "photo_setup_hd189_T_dep.npz"
_REGEN_HINT = (
    "local fixture missing; regenerate with: python tests/_gen_photo_baseline.py"
)
BIN_ATOL = 2e-14
CROSS_ATOL = 1e-30
# (dense array, its key list, fixture prefix); no prefix is a prefix of another.
_CROSS_TABLES = (
    ("absp_cross", "absp_sp", "cross__"),
    ("absp_T_cross", "absp_T_sp", "cross_T__"),
    ("cross_J", "branch_keys", "cross_J__"),
    ("cross_J_T", "branch_T_keys", "cross_J_T__"),
    ("scat_cross", "scat_sp", "cross_scat__"),
    ("cross_Jion", "ion_branch_keys", "cross_Jion__"),
)


def _check_static_against_fixture(
    static,
    fixture_path: Path,
    *,
    expected_T_sp: tuple[str, ...] = (),
) -> None:
    """Bit-exact compare every dense array in `static` to its fixture entry."""
    fx = np.load(fixture_path)

    np.testing.assert_allclose(
        np.asarray(static.bins),
        fx["bins"],
        rtol=0.0,
        atol=BIN_ATOL,
    )
    assert int(static.nbin) == int(fx["nbin"])
    assert float(static.dbin1) == float(fx["dbin1"])
    assert float(static.dbin2) == float(fx["dbin2"])
    assert tuple(static.absp_T_sp) == expected_T_sp
    assert tuple(static.absp_T_sp) == tuple(
        sp for sp in static.absp_sp if sp in expected_T_sp
    )
    for arr, keys, prefix in _CROSS_TABLES:
        names = [
            prefix + ("__".join(map(str, k)) if isinstance(k, tuple) else k)
            for k in getattr(static, keys)
        ]
        assert set(names) == {f for f in fx.files if f.startswith(prefix)}
        for i, name in enumerate(names):
            np.testing.assert_allclose(
                np.asarray(getattr(static, arr)[i]),
                fx[name],
                rtol=0.0,
                atol=CROSS_ATOL,
                err_msg=name,
            )


@pytest.mark.skipif(not _BASELINE_FIXTURE.exists(), reason=_REGEN_HINT)
def test_photo_setup_matches_baseline_fixture():
    """HD189 default: T_cross_sp=[], use_ion=False."""
    import vulcan_jax.photo_setup as photo_setup

    var, atm = build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)
    _check_static_against_fixture(static, _BASELINE_FIXTURE)


@pytest.mark.strict_isolation
@pytest.mark.skipif(not _T_DEP_FIXTURE.exists(), reason=_REGEN_HINT)
def test_photo_setup_matches_T_dep_fixture(monkeypatch):
    """HD189 with T_cross_sp=['CO2','H2O','NH3'] patched on."""
    import vulcan_jax.photo_setup as photo_setup
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    monkeypatch.setattr(vulcan_cfg, "T_cross_sp", ["CO2", "H2O", "NH3"])
    var, atm = build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)
    _check_static_against_fixture(
        static,
        _T_DEP_FIXTURE,
        expected_T_sp=("CO2", "H2O", "NH3"),
    )


@pytest.mark.strict_isolation
def test_branch_key_order_is_deterministic():
    """Branch rows must be sorted, not in set-iteration order.

    `var.photo_sp` / `var.ion_sp` are sets and string hashing is per-process,
    so unsorted iteration changes the compiled program every run and makes the
    persistent compile cache write-only. Sortedness pins the invariant in one
    process; the failure it guards against only shows up across processes.
    """
    import vulcan_jax.photo_setup as photo_setup

    var, atm = build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)

    for name in ("branch_keys", "branch_T_keys", "ion_branch_keys"):
        keys = getattr(static, name)
        if not keys:
            continue
        species = [sp for sp, _ in keys]
        assert species == sorted(species), (
            f"{name} is not in sorted species order: {species[:8]}..."
        )
        # Branches within one species stay in ascending branch number.
        for sp in set(species):
            branches = [br for s, br in keys if s == sp]
            assert branches == sorted(branches), f"{name}: {sp} branches {branches}"
