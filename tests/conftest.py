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
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
VULCAN_MASTER_STR = str(VULCAN_MASTER)

HAS_VULCAN_MASTER = VULCAN_MASTER.is_dir()

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


@pytest.fixture(scope="session")
def vulcan_master_op():
    """Import VULCAN-master's `op` for oracle-comparison tests.

    Returns the imported module. Skips the test cleanly if
    `../VULCAN-master/` isn't present in this workspace.
    """
    if not HAS_VULCAN_MASTER:
        pytest.skip(
            f"VULCAN-master not present at {VULCAN_MASTER}; "
            "oracle test skipped (VULCAN-JAX is standalone)."
        )
    old_path = list(sys.path)
    try:
        sys.path.insert(0, VULCAN_MASTER_STR)
        sys.modules.pop("op", None)
        import op

        return op
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# vulcan_cfg snapshot/restore fixtures.
# ---------------------------------------------------------------------------

# Modules that captured `vulcan_cfg` at import time. When the module
# identity drifts, these need re-binding.
_VCFG_REBIND_TARGETS = (
    "vulcan_jax.legacy_io",
    "vulcan_jax.outer_loop",
    "vulcan_jax.state",
    "vulcan_jax.atm_setup",
    "vulcan_jax.ini_abun",
    "vulcan_jax.photo_setup",
    "vulcan_jax.jax_step",
    "vulcan_jax.op_jax",
    "vulcan_jax.rates",
    "vulcan_jax.composition",
    "vulcan_jax.chem_funs",
)

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
    """Capture the canonical `vulcan_cfg` module + a deep-copy of every
    public attribute, plus canonical VULCAN-JAX module objects."""
    _restore_import_state()
    import vulcan_jax.legacy_io as _io

    canonical = _io.vulcan_cfg
    sys.modules["vulcan_jax.vulcan_cfg"] = canonical
    # Also set bare key for any remaining legacy references.
    sys.modules["vulcan_cfg"] = canonical
    canonical_modules: dict[str, Any] = {}
    for name in _VULCAN_JAX_MODULE_NAMES:
        try:
            canonical_modules[name] = importlib.import_module(name)
        except Exception:
            pass
    snap = {
        "module": canonical,
        "id": id(canonical),
        "attrs": _snapshot_cfg_attrs(canonical),
        "modules": canonical_modules,
    }
    yield snap
    _restore_cfg(snap)


def _restore_cfg(snap: dict) -> None:
    """Re-bind `sys.modules["vulcan_jax.vulcan_cfg"]` + downstream modules
    to the canonical object and restore every snapshotted attribute."""
    canonical = snap["module"]
    canonical_id = snap["id"]
    _restore_import_state(snap)
    current = sys.modules.get("vulcan_jax.vulcan_cfg")
    if current is None or id(current) != canonical_id:
        sys.modules["vulcan_jax.vulcan_cfg"] = canonical
        sys.modules["vulcan_cfg"] = canonical
        for name in _VCFG_REBIND_TARGETS:
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, "vulcan_cfg", None) is not canonical:
                mod.vulcan_cfg = canonical
    snap_attrs = snap["attrs"]
    for name, val in snap_attrs.items():
        try:
            setattr(canonical, name, copy.deepcopy(val))
        except Exception:
            setattr(canonical, name, val)
    for name in list(vars(canonical).keys()):
        if name.startswith("_") or name in snap_attrs:
            continue
        val = getattr(canonical, name)
        if isinstance(val, type) or hasattr(val, "__loader__"):
            continue
        try:
            delattr(canonical, name)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _cfg_guard(request, _cfg_snapshot_session):
    """Restore canonical `vulcan_cfg` state after every test."""
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
    import vulcan_jax.outer_loop as outer_loop

    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_jax.vulcan_cfg"] = vulcan_cfg
    sys.modules["vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg

    from vulcan_jax.atm_setup import Atm
    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    make_atm = Atm()
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
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
