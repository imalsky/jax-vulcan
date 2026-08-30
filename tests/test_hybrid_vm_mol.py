"""In-loop hybrid molecular-diffusion phase flip (use_hybrid_vm_mol).

The runner blends the vm/central-difference diffusion by a traced carry value
`hybrid_use_vm` (1.0 upwind, 0.0 central). When use_hybrid_vm_mol is on, the
body flips it 1.0 -> 0.0 the first time phase 0 (upwind) ends — by convergence,
runtime, or step count — so a run that stops on the loop's own termination test
finishes in the central-difference phase. That state is a central-difference
fixed point only when phase 1 converged too. Non-hybrid runs never flip it, so
their trace is bit-identical to the pre-change static path.

Ports the two-stage strategy of vm_branch op.py stop() into a single
lax.while_loop (keeps forward-mode jvp working end-to-end, unlike a host-side
two-stage driver which retrieval's inner-runner path would bypass).

Fast checks run count-terminated (small count_max). The convergence/flip probe
needs a real integration and is gated behind VULCAN_JAX_RUN_SLOW=1.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from _helpers import fast_cfg

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def _pin_cfg():
    """Fast, FastChem-free (const_mix), photo-off isothermal config."""
    return fast_cfg(
        trun_min=1e-30,
        use_condense=False,
        Tiso=1200.0,
        ini_mix="const_mix",
        use_moldiff=True,
        nz=40,
    )
def _run(vc, *, use_vm, hybrid, count_max):
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.state import RunState

    vc.use_vm_mol = bool(use_vm)
    vc.use_hybrid_vm_mol = bool(hybrid)
    vc.count_max = int(count_max)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    rs = RunState.with_pre_loop_setup(vc)
    init_state, atm_static = integ.prepare_runstate(rs)
    return integ._runner(init_state, atm_static)


def test_central_run_stays_use_vm_zero():
    """use_vm_mol=False: hybrid_use_vm is pinned to 0.0 (central diff)."""
    vc = _pin_cfg()
    final = _run(vc, use_vm=False, hybrid=False, count_max=30)
    assert float(final.hybrid_use_vm) == 0.0
    assert np.all(np.isfinite(np.asarray(final.y)))


def test_pure_upwind_stays_use_vm_one():
    """use_vm_mol=True, hybrid=False: hybrid_use_vm stays 1.0 (never flips)."""
    vc = _pin_cfg()
    final = _run(vc, use_vm=True, hybrid=False, count_max=30)
    assert float(final.hybrid_use_vm) == 1.0
    assert np.all(np.isfinite(np.asarray(final.y)))


@pytest.mark.skipif(
    os.environ.get("VULCAN_JAX_RUN_SLOW") != "1",
    reason="set VULCAN_JAX_RUN_SLOW=1 (a completed hybrid run integrates phase 1)",
)
def test_hybrid_flips_and_extends_budget():
    """A completed hybrid run always ends in phase 1: phase 0 ends (here by
    hitting the small count_max), flips hybrid_use_vm 1.0 -> 0.0, and extends
    the step budget for the central-difference phase (vm_branch count+1000)."""
    vc = _pin_cfg()
    count_max = 30
    final = _run(vc, use_vm=True, hybrid=True, count_max=count_max)
    # The run flipped to central diff...
    assert float(final.hybrid_use_vm) == 0.0
    # ...and ran well past the phase-0 count_max because phase 1 got its own
    # extended budget (count+1000), not the static cap.
    assert int(final.accept_count) > count_max + 100
    assert np.all(np.isfinite(np.asarray(final.y)))


def test_returned_atm_carries_refreshed_zmco_and_vm():
    """The returned atm (what the .vul writer stores) must carry the fields
    upstream's update_mu_dz refreshes in-loop: zmco from the refreshed zco
    (op.py:972-973) and, under use_vm_mol, vm from the refreshed g/Hpi/dzi
    (vm_branch op.py:945-992). Setup-time copies are the silent failure."""
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax import phy_const as C
    from vulcan_jax.atm_refresh import recompute_vm_jax
    from vulcan_jax.state import RunState

    vc = _pin_cfg()
    vc.use_vm_mol, vc.use_hybrid_vm_mol = True, False
    vc.count_max, vc.use_atm_refresh, vc.update_frq = 40, True, 5
    rs = RunState.with_pre_loop_setup(vc)
    a0 = rs.atm
    a = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())(rs).atm
    zco = np.asarray(a.zco)
    assert np.max(np.abs(zco - np.asarray(a0.zco))) > 0.0  # the refresh ran
    np.testing.assert_array_equal(np.asarray(a.zmco), 0.5 * (zco[:-1] + zco[1:]))
    vm_ref = recompute_vm_jax(a.g, a.Hpi, a.dzi, a.Dzz, a.ms, a.alpha, a.Tco,
                              float(C.kb), float(C.Navo))
    np.testing.assert_array_equal(np.asarray(a.vm), np.asarray(vm_ref))
