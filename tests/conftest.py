"""Shared pytest setup for VULCAN-JAX tests.

Script-style test files keep a `main()` entry plus a thin
`test_main(): assert main() == 0` wrapper so pytest collects them.
The upstream oracle is optional: oracle tests run master comparisons in
fresh subprocesses and skip cleanly when it is absent. The `hd189_state`
fixture hands out a per-test deep copy of the session-built HD189
pre-loop state so tests avoid re-running rates / the EQ seed / photo setup.
"""

from __future__ import annotations

import copy
import fcntl as _fcntl
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Many tests assume cwd == ROOT for relative data paths.
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def _assert_testing_repo_checkout() -> None:
    """Fail collection loudly if `import vulcan_jax` resolves outside this
    repo: a non-editable install shadows the checkout and the suite silently
    tests stale code.
    """
    import vulcan_jax

    pkg_path = Path(vulcan_jax.__file__).resolve()
    expected = (ROOT / "src" / "vulcan_jax").resolve()
    if not pkg_path.is_relative_to(expected):
        raise pytest.UsageError(
            f"`import vulcan_jax` resolves to {pkg_path}, not this checkout "
            f"({expected}). You are testing a stale installed copy, not the "
            "working tree. Fix with an editable install:\n"
            "    pip install -e . --no-deps"
        )


_assert_testing_repo_checkout()


# Config snapshot/restore fixtures. Upstream modules are only ever imported
# in subprocesses, so the parent's import state needs no restoring.


def _clear_jax_caches() -> None:
    """Clear JAX compilation caches when strict test isolation requests it."""
    jax_mod = sys.modules.get("jax")
    clear = getattr(jax_mod, "clear_caches", None)
    if clear is not None:
        clear()


def _snapshot_cfg_attrs(cfg_module) -> dict:
    """Deep-copy every public attribute of `cfg_module` for later restore."""
    snap = {}
    for name in dir(cfg_module):
        if name.startswith("_"):
            continue
        val = getattr(cfg_module, name)
        if isinstance(val, type) or hasattr(val, "__loader__"):
            continue
        try:
            snap[name] = copy.deepcopy(val)
        except Exception:
            pass
    return snap


@pytest.fixture(scope="session", autouse=True)
def _cfg_snapshot_session():
    """Snapshot the process default config (a single object that
    `state._cfg_overlay` mutates in place) so `_cfg_guard` can restore it
    after every test."""
    from vulcan_jax.config import default_config

    canonical = default_config()
    snap = {"cfg": canonical, "attrs": _snapshot_cfg_attrs(canonical)}
    yield snap
    _restore_cfg(snap)


def _restore_cfg(snap: dict) -> None:
    """Restore every snapshotted attribute on the process default config and
    drop any attribute a test added."""
    canonical = snap["cfg"]
    snap_attrs = snap["attrs"]
    for name, val in snap_attrs.items():
        try:
            setattr(canonical, name, copy.deepcopy(val))
        except Exception:
            setattr(canonical, name, val)
    for name in list(vars(canonical).keys()):
        if name.startswith("_") or name in snap_attrs:
            continue
        try:
            delattr(canonical, name)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _cfg_guard(request, _cfg_snapshot_session):
    """Restore the process default config after every test."""
    strict = request.node.get_closest_marker("strict_isolation") is not None
    if strict:
        _restore_cfg(_cfg_snapshot_session)
        _clear_jax_caches()
    try:
        yield
    finally:
        _restore_cfg(_cfg_snapshot_session)
        if strict:
            _clear_jax_caches()


@pytest.fixture(autouse=True, scope="module")
def _release_jax_caches_per_module():
    """Drop a module's compiled programs when it finishes, so per-xdist-worker
    caches do not exhaust runner memory."""
    yield
    _clear_jax_caches()


# Cross-process serialisation for master-touching tests.

_MASTER_LOCK = ROOT / "tests" / ".master_lock"


@pytest.fixture(autouse=True)
def _master_lock(request):
    """Serialise master-touching tests via cross-process flock."""
    if request.node.get_closest_marker("master_serial") is None:
        yield
        return
    _MASTER_LOCK.touch(exist_ok=True)
    with open(_MASTER_LOCK, "r") as lock_f:
        _fcntl.flock(lock_f, _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(lock_f, _fcntl.LOCK_UN)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "master_serial: serialize across pytest-xdist workers via "
        "tests/.master_lock for tests that read or write VULCAN-master.",
    )
    config.addinivalue_line(
        "markers",
        "strict_isolation: restore the default config and clear JAX caches "
        "before and after the test.",
    )


@dataclass
class HD189State:
    """Canonical HD189 pre-loop reference state for tests."""

    var: Any
    atm: Any
    para: Any
    solver: Any


@pytest.fixture(scope="session")
def _hd189_pristine() -> HD189State:
    """One-time HD189 pre-loop build."""
    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.config import default_config
    from vulcan_jax.state import RunState, legacy_view

    cfg = default_config()
    rs = RunState.with_pre_loop_setup(cfg)
    data_var, data_atm, data_para = legacy_view(rs)

    solver = op_jax.Ros2JAX()
    if cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static

    return HD189State(
        var=data_var,
        atm=data_atm,
        para=data_para,
        solver=solver,
    )


@pytest.fixture
def hd189_state(_hd189_pristine: HD189State) -> HD189State:
    """Fresh per-test deep copy of the HD189 pre-loop state."""
    p = _hd189_pristine
    return HD189State(
        var=copy.deepcopy(p.var),
        atm=copy.deepcopy(p.atm),
        para=copy.deepcopy(p.para),
        solver=p.solver,
    )


# --- numerical-oracle fixture guard -----------------------------------------
# Untracked .npz oracles (~36 MB; `python tests/gen_fixtures.py --all`). A
# missing one would make its tests skip and the suite pass, so collection
# fails instead. path -> the generator that writes it.
_EXPECTED_FIXTURES = {
    "tests/data/adj_state_hd189.npz": "tests/_gen_adj_state.py hd189",
    "tests/data/adj_state_w39b.npz": "tests/_gen_adj_state.py w39b",
    "tests/data/photo_setup_hd189_baseline.npz": "tests/_gen_photo_baseline.py",
    "tests/data/photo_setup_hd189_T_dep.npz": "tests/_gen_photo_baseline.py",
}


def pytest_collection_finish():
    """Fail loudly when a numerical oracle is missing (skipped != passed)."""
    root = Path(__file__).resolve().parents[1]
    missing = {
        rel: how
        for rel, how in _EXPECTED_FIXTURES.items()
        if not (root / rel).is_file()
    }
    if not missing:
        return
    lines = [
        f"  - {rel}\n      (individually: python {how})"
        for rel, how in sorted(missing.items())
    ]
    msg = (
        f"{len(missing)} numerical-regression oracle(s) are missing, so the tests "
        f"comparing against them would SKIP and the suite would still report "
        f"green:\n" + "\n".join(lines) + "\n\n"
        "Build them all with one command:\n"
        "    python tests/gen_fixtures.py --all\n"
        "(then `--verify` checks them against tests/data/FIXTURES.json).\n\n"
        "VULCAN_JAX_ALLOW_MISSING_FIXTURES=1 accepts the reduced coverage. That "
        "is a DEVELOPER escape hatch: it must not be set in release CI, where "
        "the point is that these oracles ran."
    )
    if os.environ.get("VULCAN_JAX_ALLOW_MISSING_FIXTURES") == "1":
        warnings.warn(msg, RuntimeWarning, stacklevel=1)
        return
    raise pytest.UsageError(msg)
