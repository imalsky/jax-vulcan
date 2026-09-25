"""Validate that the JAX runner is one-shot and the in-runner
convergence check terminates the integration without Python-side polling.

Three assertions:

  1. **Ring buffer + chronology**: a count_max=50 HD189 run leaves the
     convergence ring holding min(count, conv_step) = 51 entries (the
     runner exits when `accept_count > count_max`, so the final
     accept_count is 51), in strictly increasing time order when read
     from slot `(accept_count - L) % conv_step` on. The first entry is the
     post-step-1 state (not the pre-loop initial state — matches
     `op.save_step` semantics, which appends AFTER the accepted step). The
     last entry equals the final `t`.

  2. **longdy / longdydt populated**: after the runner returns,
     `rs.step.longdy` and `rs.step.longdydt` are finite and positive (the
     in-runner conv check ran and updated them at every accepted step).

  3. **Single-shot termination via count_max**: with count_max=50, the
     runner exits exactly when `accept_count > count_max`. The count
     becomes 51 (50 accepted body iterations + the off-by-one
     terminating attempt that triggers `>`) and end_case = 3
     ("Maximal allowed steps exceeded").

The ring buffer is sized at `conv_step` (500 by default), so 50 < 500 and
it holds the full trajectory; longer runs overwrite the oldest slots.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def main() -> int:
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

    ok = True
    if float(final.t) != float(rs_out.step.t):
        print(f"FAIL: runner t {float(final.t):.6e} != integ(rs) t {rs_out.step.t:.6e}")
        ok = False

    # ---- 1. Ring buffer + chronological reconstruction ----
    count = int(final.accept_count)
    conv_step = int(vulcan_cfg.conv_step)
    n_kept = min(count, conv_step)
    start = (count - n_kept) % conv_step
    order = [(start + i) % conv_step for i in range(n_kept)]
    t_arr = np.asarray(final.t_time_ring)[order]

    # Chronology: t should be strictly increasing.
    if not np.all(np.diff(t_arr) > 0):
        print(
            "FAIL: ring times not strictly increasing — ring "
            "reconstruction is in the wrong order"
        )
        ok = False

    # Last entry == final t (most recent ring slot).
    if t_arr[-1] != float(final.t):
        print(f"FAIL: last ring t ({t_arr[-1]:.3e}) != t ({float(final.t):.3e})")
        ok = False

    print(
        f"ring chronology  OK ({n_kept} entries, t spans "
        f"{t_arr[0]:.3e}..{t_arr[-1]:.3e})"
    )

    # ---- 2. longdy / longdydt populated ----
    longdy, longdydt = rs_out.step.longdy, rs_out.step.longdydt
    if not (np.isfinite(longdy) and longdy > 0):
        print(f"FAIL: longdy = {longdy} is not finite/positive")
        ok = False
    if not (np.isfinite(longdydt) and longdydt > 0):
        print(f"FAIL: longdydt = {longdydt} is not finite/positive")
        ok = False
    print(f"longdy/longdydt  OK (longdy={longdy:.3e}, longdydt={longdydt:.3e})")

    # ---- 3. Single-shot termination via count_max ----
    count_out = int(rs_out.params.count)
    end_case = int(rs_out.params.end_case)
    if count_out != vulcan_cfg.count_max + 1:
        print(f"FAIL: count={count_out}, expected {vulcan_cfg.count_max + 1}")
        ok = False
    if end_case != 3:
        print(f"FAIL: end_case={end_case}, expected 3 (count_max exceeded)")
        ok = False
    print(f"count_max exit   OK (count={count_out}, end_case={end_case})")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper around main()."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
