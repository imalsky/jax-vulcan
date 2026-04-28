"""Shared pytest setup for VULCAN-JAX tests.

Each `tests/test_*.py` retains its existing `def main()` script entry
point and adds a thin `def test_main(): assert main() == 0` wrapper so
`pytest tests/` collects and runs them. This conftest pins working
directory and `sys.path` once per session.

VULCAN-master sibling: VULCAN-JAX is standalone; the upstream repo
serves as an *optional* validation oracle. If `../VULCAN-master/` is
present, it gets appended to sys.path so the 10 oracle tests
(test_rates, test_chem, test_diffusion, test_diffusion_variants,
test_photo, test_photo_wired, test_ros2_step, test_gibbs,
test_step_control, test_outer_loop_atm_refresh) can `import op` and
compare. If absent, those tests skip cleanly — the rest of the suite
is unaffected.

Phase 18 fixture: `hd189_state` provides a per-test deep copy of the
HD189 pre-loop state (Variables, AtmData, Parameters) built once per
session via `_hd189_pristine`. Tests that just need a clean reference
state can request it instead of re-running the rate parser / FastChem /
photo cross-section read every time. See `tests/test_chem_jac_sparse.py`
for the canonical migration pattern.
"""
from __future__ import annotations

import copy
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
VULCAN_MASTER = ROOT.parent / "VULCAN-master"

# Make sure VULCAN-JAX is importable regardless of where pytest was launched.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# VULCAN-master is optional — only append when present.
HAS_VULCAN_MASTER = VULCAN_MASTER.is_dir()
if HAS_VULCAN_MASTER and str(VULCAN_MASTER) not in sys.path:
    sys.path.append(str(VULCAN_MASTER))

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
    import op
    return op


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

    Also pins the vulcan_cfg module identity so test-side mutations
    on the canonical cfg are visible to outer_loop / legacy_io. See
    Phase 16 suite-ordering note: `test_chem`'s `sys.modules.pop`
    forks fresh vulcan_cfg objects on subsequent imports, and any
    test that writes `vulcan_cfg.count_max = N` on the wrong copy
    silently fails because the runner reads from the other one.
    """
    # Sync vulcan_cfg references against legacy_io's anchor.
    import legacy_io as op
    import outer_loop
    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg

    from atm_setup import Atm
    from ini_abun import InitialAbun
    import op_jax
    import store

    import rates as _rates

    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()
    make_atm = Atm()
    output = op.Output()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    # `read_rate` is retained for metadata population (var.Rf,
    # var.pho_rate_index, var.n_branch, var.photo_sp, ...). Rate
    # *values* come from `build_rate_array` (Phase 22a path).
    data_var = rate.read_rate(data_var, data_atm)
    _network = _rates.setup_var_k(vulcan_cfg, data_var, data_atm)
    ini_abun = InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo:
        import photo_setup as _photo_setup
        _photo_setup.populate_photo_arrays(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)
        solver.compute_tau(data_var, data_atm)
        solver.compute_flux(data_var, data_atm)
        solver.compute_J(data_var, data_atm)
        _rates.apply_photo_remove(vulcan_cfg, data_var, _network, data_atm)

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
