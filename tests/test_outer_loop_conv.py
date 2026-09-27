"""The runner is one-shot and its in-loop convergence check ends the run.

A count_max=50 HD189 run must:
(1) leave the conv ring with 51 strictly increasing times, read from slot
    (accept_count - L) % conv_step; the first is a post-step state (as
    op.save_step) and the last is the final t;
(2) leave longdy and longdydt finite and positive;
(3) end with count == count_max + 1 and end_case 3;
(4) keep every atom_loss under MAX_ATOM_LOSS and end with finite, positive
    dt and t.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Roundoff the conservation projection leaves over 50 HD189 steps (~3.5e-10); a 10x regression still trips 1e-8.
MAX_ATOM_LOSS = 1.0e-8


@pytest.mark.strict_isolation
def test_count_capped_run_conv_ring_and_termination():
    import vulcan_jax.outer_loop as outer_loop
    import vulcan_jax.legacy_io as op

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = 50
    vulcan_cfg.count_min = 1
    # A count-capped run needs the hybrid vm_mol budget extension OFF: phase-0
    # budget exhaustion flips to phase 1 and extends count_max by +1000, which
    # breaks this test's exact count == count_max + 1 contract.
    vulcan_cfg.use_vm_mol = False
    vulcan_cfg.use_hybrid_vm_mol = False
    vulcan_cfg.use_print_prog = False

    import vulcan_jax.op_jax as op_jax
    from vulcan_jax.state import RunState

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    rs_out = integ(rs)
    # The same run on the raw runner, for the ring the RunState does not carry.
    final = integ._runner(*integ.prepare_runstate(rs))

    assert float(final.t) == float(rs_out.step.t), "runner t != integ(rs) t"

    # ---- 1. Ring buffer + chronological reconstruction ----
    count = int(final.accept_count)
    conv_step = int(vulcan_cfg.conv_step)
    n_kept = min(count, conv_step)
    start = (count - n_kept) % conv_step
    order = [(start + i) % conv_step for i in range(n_kept)]
    t_arr = np.asarray(final.t_time_ring)[order]
    assert np.all(np.diff(t_arr) > 0), "ring reconstruction is in the wrong order"
    assert t_arr[-1] == float(final.t), "last ring t != final t"

    # ---- 2. longdy / longdydt populated ----
    longdy, longdydt = rs_out.step.longdy, rs_out.step.longdydt
    assert np.isfinite(longdy) and longdy > 0, longdy
    assert np.isfinite(longdydt) and longdydt > 0, longdydt

    # ---- 3. Single-shot termination via count_max ----
    assert int(rs_out.params.count) == vulcan_cfg.count_max + 1
    assert int(rs_out.params.end_case) == 3, "expected end_case 3 (count_max exceeded)"

    # ---- 4. Conservation and a finite clock ----
    for atom, loss in zip(rs_out.atoms.atom_order, np.asarray(rs_out.atoms.atom_loss)):
        assert abs(loss) <= MAX_ATOM_LOSS, f"atom_loss[{atom}] = {loss:.3e}"
    t, dt = float(rs_out.step.t), float(rs_out.step.dt)
    assert dt > 0 and math.isfinite(dt), dt
    assert t > 0 and math.isfinite(t), t
