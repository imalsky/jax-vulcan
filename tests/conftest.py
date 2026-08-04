"""Shared pytest setup for VULCAN-JAX tests.

Each `tests/test_*.py` retains its existing `def main()` script entry
point and adds a thin `def test_main(): assert main() == 0` wrapper so
`pytest tests/` collects and runs them.

VULCAN-master sibling: VULCAN-JAX is standalone; the upstream repo
serves as an *optional* validation oracle. If `../VULCAN-master/` is
present, oracle tests run their master comparisons in fresh subprocesses
and add the sibling path only inside those subprocesses. If absent, those
tests skip cleanly -- the rest of the suite is unaffected.

The `hd189_state` fixture provides a per-test deep copy of the HD189
pre-loop state (Variables, AtmData, Parameters) built once per session
via `_hd189_pristine`. Tests that just need a clean reference state can
request it instead of re-running the rate parser / FastChem / photo
cross-section read every time.
"""

from __future__ import annotations

import copy
import fcntl as _fcntl
import importlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Oracle location comes from $VULCAN_MASTER_DIR, never from a sibling guess.
# An auto-detected ../VULCAN-master pins no revision, and the copy on this
# project's machine is not a git checkout at all (it was found in 2026-07 to
# contain VULCAN-JAX's own code, which had then been cited as upstream
# evidence). `tests/oracle.py` resolves the path and verifies the pinned commit
# and a clean worktree before any comparison runs. This constant exists only so
# the import-leak cleanup below can strip the path from sys.path.
_master_env = os.environ.get("VULCAN_MASTER_DIR")
VULCAN_MASTER = (Path(_master_env).expanduser().resolve() if _master_env
                 else Path("/nonexistent/VULCAN-oracle-unset"))
VULCAN_MASTER_STR = str(VULCAN_MASTER)

# Many tests assume cwd == ROOT for relative paths in vulcan_cfg.py.
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def _assert_testing_repo_checkout() -> None:
    """Fail collection loudly if `import vulcan_jax` resolves outside this repo.

    The package uses a src layout, so the suite imports the *installed*
    `vulcan_jax`. A non-editable install (e.g. `pip install .` from a release)
    shadows the checkout and the suite silently tests stale code — branch
    changes appear to fail (or pass) for reasons unrelated to the working tree.
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


# ---------------------------------------------------------------------------
# vulcan_cfg snapshot/restore fixtures.
# ---------------------------------------------------------------------------

_MASTER_ONLY_MODULE_NAMES = (
    "op",
    "build_atm",
    "store",
)

_VULCAN_JAX_MODULE_NAMES = (
    "vulcan_jax.legacy_io",
    "vulcan_jax.chem_funs",
    "vulcan_jax.network",
    "vulcan_jax.rates",
    "vulcan_jax.gibbs",
    "vulcan_jax.chem",
    "vulcan_jax.atm_setup",
    "vulcan_jax.ini_abun",
    "vulcan_jax.photo_setup",
    "vulcan_jax.outer_loop",
    "vulcan_jax.state",
    "vulcan_jax.jax_step",
    "vulcan_jax.op_jax",
    "vulcan_jax.composition",
)


def _module_is_under(mod: Any, root: Path) -> bool:
    """Return True when a loaded module came from `root`."""
    module_file = getattr(mod, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _restore_import_state(snap: dict | None = None) -> None:
    """Drop sibling-master modules and restore canonical VULCAN-JAX modules."""
    # Remove VULCAN-master path leakage.
    sys.path[:] = [p for p in sys.path if p != VULCAN_MASTER_STR]
    for name in _MASTER_ONLY_MODULE_NAMES:
        mod = sys.modules.get(name)
        if mod is not None and _module_is_under(mod, VULCAN_MASTER):
            sys.modules.pop(name, None)
    if snap is None:
        return
    for name, mod in snap.get("modules", {}).items():
        if sys.modules.get(name) is not mod:
            sys.modules[name] = mod


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
    """Capture the process default config (`config.default_config()`) + a
    deep-copy of every public attribute, plus canonical VULCAN-JAX modules.

    The default config is a single process-wide `Config` object that
    `state._cfg_overlay` mutates in place during setup; snapshotting its
    attributes lets `_cfg_guard` restore it after any test that overlays or
    mutates it directly."""
    _restore_import_state()
    from vulcan_jax.config import default_config

    canonical = default_config()
    canonical_modules: dict[str, Any] = {}
    for name in _VULCAN_JAX_MODULE_NAMES:
        try:
            canonical_modules[name] = importlib.import_module(name)
        except Exception:
            pass
    snap = {
        "cfg": canonical,
        "attrs": _snapshot_cfg_attrs(canonical),
        "modules": canonical_modules,
    }
    yield snap
    _restore_cfg(snap)


def _restore_cfg(snap: dict) -> None:
    """Restore every snapshotted attribute on the process default config and
    drop any attribute a test added."""
    canonical = snap["cfg"]
    _restore_import_state(snap)
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


# ---------------------------------------------------------------------------
# Cross-process serialisation for master-touching tests.
# ---------------------------------------------------------------------------

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
        "strict_isolation: restore VULCAN-JAX import/config state and clear "
        "JAX caches before and after the test.",
    )


@dataclass
class HD189State:
    """Canonical HD189 pre-loop reference state for tests."""

    var: Any
    atm: Any
    para: Any
    make_atm: Any
    output: Any
    solver: Any


@pytest.fixture(scope="session")
def _hd189_pristine() -> HD189State:
    """One-time HD189 pre-loop build."""
    import vulcan_jax.legacy_io as op

    from vulcan_jax.config import default_config

    cfg = default_config()

    from vulcan_jax.atm_setup import Atm
    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    make_atm = Atm()
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static

    return HD189State(
        var=data_var,
        atm=data_atm,
        para=data_para,
        make_atm=make_atm,
        output=output,
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
        make_atm=p.make_atm,
        output=p.output,
        solver=p.solver,
    )


# --- numerical-oracle fixture guard -----------------------------------------
# These .npz oracles stay OUT of git on purpose (~36 MB of binaries, all
# regenerable). A blanket `*.npz` gitignore rule once left them untracked, so on
# a fresh clone the tests comparing against them SKIPPED and the suite still
# went green.
#
# Every one is now reproducible FROM THIS REPOSITORY with a single command --
# `python tests/gen_fixtures.py --all` -- with no sibling checkout. Until
# 2026-08-03 the two adjoint states could only be built through the unpublished
# `jax_paper/` sibling, and those generators had themselves rotted (a hardcoded
# `/Users/.../Desktop/...` root that no longer exists), so the default suite
# could not run from the published source at all.
#
# path -> the generator that writes it.
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
