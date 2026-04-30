"""Shared pytest setup for VULCAN-JAX tests.

Each `tests/test_*.py` retains its existing `def main()` script entry
point and adds a thin `def test_main(): assert main() == 0` wrapper so
`pytest tests/` collects and runs them. This conftest pins working
directory and keeps the VULCAN-JAX root as the only normal import root.

VULCAN-master sibling: VULCAN-JAX is standalone; the upstream repo
serves as an *optional* validation oracle. If `../VULCAN-master/` is
present, oracle tests run their master comparisons in fresh subprocesses
and add the sibling path only inside those subprocesses. If absent, those
tests skip cleanly — the rest of the suite is unaffected.

The `hd189_state` fixture provides a per-test deep copy of the HD189
pre-loop state (Variables, AtmData, Parameters) built once per session
via `_hd189_pristine`. Tests that just need a clean reference state can
request it instead of re-running the rate parser / FastChem / photo
cross-section read every time. See `tests/test_chem_jac_sparse.py` for
the canonical migration pattern.
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
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
ROOT_STR = str(ROOT)
VULCAN_MASTER_STR = str(VULCAN_MASTER)


def _restore_sys_path() -> None:
    """Keep VULCAN-JAX first and remove sibling-master path leakage."""
    cleaned = [
        path for path in sys.path
        if path not in (ROOT_STR, VULCAN_MASTER_STR)
    ]
    sys.path[:] = [ROOT_STR, *cleaned]


# Make sure VULCAN-JAX is importable regardless of where pytest was launched.
_restore_sys_path()

HAS_VULCAN_MASTER = VULCAN_MASTER.is_dir()

# Many tests assume cwd == ROOT for relative paths in vulcan_cfg.py.
os.chdir(ROOT)

warnings.filterwarnings("ignore")


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
#
# Test-suite state pollution: a few tests mutate `vulcan_cfg.*` attributes
# at module-import time (e.g. `count_max = 5`, `use_fix_H2He = True`), and
# a single offender (`tests/test_chem.py`) does
# `sys.modules.pop("vulcan_cfg")` mid-run to swap in VULCAN-master's
# `vulcan_cfg`. The pop forks a fresh module object on the next
# `import vulcan_cfg`, breaking the canonical reference that `outer_loop`,
# `legacy_io`, `state`, and others captured at import time.
#
# `_cfg_snapshot_session` (session-scoped, autouse) captures the canonical
# `vulcan_cfg` module identity plus a deep-copy of every public attribute
# the first time it is consulted, so we have a known-good reference state.
#
# `_cfg_guard` (function-scoped, autouse) runs *after* every test:
#   1. If the current `sys.modules["vulcan_cfg"]` differs in identity from
#      the canonical module, re-bind sys.modules + the four downstream
#      modules that captured `vulcan_cfg` at import time.
#   2. Restore every snapshotted public attribute on the canonical module
#      so per-test mutations (count_max, rtol, use_fix_*, use_fix_sp_bot,
#      ...) don't bleed into the next test.
#
# We do NOT snapshot `sys.modules` wholesale — that would break the
# session-scoped `_hd189_pristine` cache and other legitimate import
# memoisation. Only `vulcan_cfg`'s public attributes are tracked.
# ---------------------------------------------------------------------------

# Modules that did `import vulcan_cfg` at their own import time and now
# hold a captured reference. When `vulcan_cfg` gets forked via
# `sys.modules.pop`, these references become stale and need re-binding.
_VCFG_REBIND_TARGETS = (
    "legacy_io",
    "outer_loop",
    "state",
    "atm_setup",
    "ini_abun",
    "photo_setup",
    "jax_step",
    "op_jax",
    "rates",
    "composition",
    "diagnose",
    "chem_funs",
)

# Tests that insert `../VULCAN-master/` at the front of `sys.path` and
# `import op` (e.g. `test_gibbs`, `test_chem`, `test_diffusion`,
# `test_oracle_*`) rebind `sys.modules["op"]` / `sys.modules["build_atm"]`
# etc. to VULCAN-master's incompatible classes. The guard drops master-only
# modules and restores the canonical VULCAN-JAX entries on every teardown so
# subsequent tests get the right modules.
#
# VULCAN-JAX has no `store.py`; the legacy mutable containers live in
# `state._Variables` / `_AtmData` / `_Parameters`. Any `import store`
# resolves uniquely to VULCAN-master's `store.py` (when present on
# sys.path), so the conftest does not pin a canonical `store` module.
_MASTER_ONLY_MODULE_NAMES = (
    "op",
    "build_atm",
    "store",
)

_VULCAN_JAX_MODULE_NAMES = (
    "legacy_io",
    "chem_funs",
    "network",
    "rates",
    "gibbs",
    "chem",
    "atm_setup",
    "ini_abun",
    "photo_setup",
    "outer_loop",
    "state",
    "jax_step",
    "op_jax",
    "composition",
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
    """Restore VULCAN-JAX import roots and drop sibling-master modules."""
    _restore_sys_path()
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
        # Skip imported modules / classes / submodules; only snapshot data.
        if isinstance(val, type) or hasattr(val, "__loader__"):
            continue
        try:
            snap[name] = copy.deepcopy(val)
        except Exception:
            # Non-deepcopyable objects (file handles, ...) — skip.
            pass
    return snap


@pytest.fixture(scope="session", autouse=True)
def _cfg_snapshot_session():
    """Capture the canonical `vulcan_cfg` module + a deep-copy of every
    public attribute, plus canonical VULCAN-JAX module objects. Yields a
    dict that `_cfg_guard` reads on each teardown.

    Runs once per session before any test imports `vulcan_cfg`. We import
    via `legacy_io` so that the canonical anchor is the same module
    object the production import graph uses (legacy_io captures it at
    its top-level `import vulcan_cfg`).
    """
    _restore_import_state()
    import legacy_io as _io
    canonical = _io.vulcan_cfg
    sys.modules["vulcan_cfg"] = canonical
    # Force-import the VULCAN-JAX modules whose names collide with
    # VULCAN-master's, then snapshot their canonical objects.
    canonical_modules: dict[str, Any] = {}
    for name in _VULCAN_JAX_MODULE_NAMES:
        try:
            canonical_modules[name] = __import__(name)
        except Exception:
            pass
    snap = {
        "module": canonical,
        "id": id(canonical),
        "attrs": _snapshot_cfg_attrs(canonical),
        "modules": canonical_modules,
    }
    yield snap
    # Final restore on session teardown so the post-suite repo state
    # doesn't carry per-test mutations.
    _restore_cfg(snap)


def _restore_cfg(snap: dict) -> None:
    """Re-bind `sys.modules["vulcan_cfg"]` + downstream modules to the
    canonical object and restore every snapshotted attribute."""
    canonical = snap["module"]
    canonical_id = snap["id"]
    # Step 0: restore VULCAN-JAX canonical modules. Tests that insert
    # `../VULCAN-master/` at the front of sys.path and re-import these
    # names (e.g. test_gibbs) rebind the entries to incompatible
    # upstream copies; the guard puts them back.
    _restore_import_state(snap)
    # Step 1: rebind sys.modules + every captured reference if the module
    # got forked since the snapshot.
    current = sys.modules.get("vulcan_cfg")
    if current is None or id(current) != canonical_id:
        sys.modules["vulcan_cfg"] = canonical
        for name in _VCFG_REBIND_TARGETS:
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, "vulcan_cfg", None) is not canonical:
                mod.vulcan_cfg = canonical
    # Step 2: restore every snapshotted attribute. Drop attributes that
    # the test added but were never in the snapshot (rare; keeps cfg
    # tidy across the suite).
    snap_attrs = snap["attrs"]
    for name, val in snap_attrs.items():
        try:
            setattr(canonical, name, copy.deepcopy(val))
        except Exception:
            setattr(canonical, name, val)
    # Drop any attributes added by the test that weren't in the snapshot,
    # but only those that look like data (skip dunder + module/class).
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
    """Restore canonical `vulcan_cfg` state after every test.

    Detects two failure modes:
      - sys.modules.pop("vulcan_cfg") forks (test_chem-style) → rebind.
      - per-test attribute mutations (count_max, rtol, use_fix_*) → restore.
    """
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
# Cross-process serialisation for tests that mutate or import VULCAN-master.
#
# Three of the oracle tests overwrite `VULCAN-master/vulcan_cfg.py` and
# regenerate `VULCAN-master/chem_funs.py` for the duration of the test.
# Several other tests `import op` / `import chem_funs` from master and
# expect a consistent on-disk state. Under `pytest -n auto` two oracle
# tests can interleave: worker A backs up master → modifies → starts;
# worker B backs up master (now in worker A's modified state!) → modifies
# → finishes → restores its (already-tainted) backup, leaving master
# permanently corrupted.
#
# `_master_lock` is an autouse fixture gated on the `master_serial`
# marker. Tests with the marker acquire an exclusive `fcntl.flock` on
# `tests/.master_lock` for the test's full duration, serialising every
# master access across pytest workers. Other tests run in parallel
# unaffected.
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
    """Canonical HD189 pre-loop reference state for tests.

    `var`, `atm`, `para` are mutable per-test instances (deep-copied
    from the session-scoped pristine state — safe to modify). The
    helper objects `make_atm`, `output`, `solver` are shared across
    tests; their internal state is treated as immutable post-setup.
    """
    var: Any
    atm: Any
    para: Any
    make_atm: Any
    output: Any
    solver: Any


@pytest.fixture(scope="session")
def _hd189_pristine() -> HD189State:
    """One-time HD189 pre-loop build.

    Runs the expensive setup steps (rate parser, reverse-rate Gibbs,
    FastChem initial abundances, photo cross-section read, atmospheric
    structure) exactly once per pytest session. The result is held
    read-only; per-test fresh copies come from `hd189_state` below.

    Also pins the vulcan_cfg module identity so test-side mutations on
    the canonical cfg are visible to outer_loop / legacy_io.
    Suite-ordering hazard: `test_chem`'s `sys.modules.pop` forks fresh
    vulcan_cfg objects on subsequent imports, and any test that writes
    `vulcan_cfg.count_max = N` on the wrong copy silently fails because
    the runner reads from the other one. The autouse fixtures above
    catch and rebind that.
    """
    # Sync vulcan_cfg references against legacy_io's anchor.
    import legacy_io as op
    import outer_loop
    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg

    # Build the runner state via the canonical pre-loop constructor; derive
    # a `(var, atm, para)` shim so legacy tests that index `var.attr` /
    # `atm.attr` keep working unchanged.
    from atm_setup import Atm
    import op_jax
    from state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    make_atm = Atm()
    output = op.Output()

    # Reuse the photo cross-section pytree the pre-loop pipeline built
    # so `solver.compute_tau` / `compute_flux` / `compute_J` don't
    # rebuild it lazily off the (now-removed) `var.cross*` dict surface.
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
    """Fresh per-test deep copy of the HD189 pre-loop state.

    Use this from any test that just needs a canonical (var, atm, para)
    starting point — chemistry/Jacobian comparisons, runner smoke tests,
    etc. The mutable fields are deep-copied so in-place mutation
    (`integ(var, atm, para, ...)`) won't bleed across tests; the
    helper objects (`make_atm`, `output`, `solver`) are shared session
    instances.
    """
    p = _hd189_pristine
    return HD189State(
        var=copy.deepcopy(p.var),
        atm=copy.deepcopy(p.atm),
        para=copy.deepcopy(p.para),
        make_atm=p.make_atm,
        output=p.output,
        solver=p.solver,
    )
