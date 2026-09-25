"""`save_evolution` time-series capture in the OuterLoop runner.

When `vulcan_cfg.save_evolution=True`, master appends `var.y` and `var.t`
to `var.y_time` / `var.t_time` every accepted step (op.py:1096-1099) and
slices by `save_evo_frq` at save time. The JAX OuterLoop captures the
trajectory directly at the configured cadence into a fixed-size buffer.

This test runs HD189 with `save_evolution=True, save_evo_frq=10` for
`count_max=50` accepted steps and asserts:
  1. `rs.step.t_evo` has the expected length:
     `ceil((count_max + 1) / save_evo_frq)` (6).
  2. `t_evo` is monotonic.
  3. `y_evo[i]` snapshot for each i >= 1 is a valid (nz, ni) array
     with no NaN / inf.
  4. The pickle save round-trips the time-series under the same key names
     as master (`y_time` / `t_time`, loadable by `plot_py/plot_evolution.py`).

Standalone — no `../VULCAN-master/` oracle needed.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def main() -> int:
    # Re-anchor cwd at every test entry — earlier tests in the suite may
    # have chdir'd elsewhere, breaking the save_out's relative output_dir.
    os.chdir(ROOT)
    import vulcan_jax.legacy_io as legacy_io
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    vulcan_cfg = legacy_io.default_config()
    save_evo_frq = 10
    count_max = 50
    # Master runs `count_max + 1` save_steps (op.py:1080 uses `>`, not `>=`),
    # then post-slices `var.y_time[::save_evo_frq]`, so the kept length is
    # `ceil((count_max + 1) / save_evo_frq)` = (count_max + save_evo_frq)
    # // save_evo_frq. For (50, 10) that's 6 entries (indices 0,10,20,30,
    # 40,50 of y_time[0..50]).
    expected_n = (count_max + save_evo_frq) // save_evo_frq  # 6

    original_save_evo = vulcan_cfg.save_evolution
    original_frq = vulcan_cfg.save_evo_frq
    vulcan_cfg.save_evolution = True
    vulcan_cfg.save_evo_frq = save_evo_frq
    vulcan_cfg.count_max = count_max
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    try:
        rs = RunState.with_pre_loop_setup(vulcan_cfg)
        output = legacy_io.Output()
        rs_out = outer_loop.OuterLoop(op_jax.Ros2JAX(), output)(rs)

        y_time = np.asarray(rs_out.step.y_evo)
        t_time = np.asarray(rs_out.step.t_evo)
        print(f"y_evo.shape = {y_time.shape}; expected ({expected_n}, nz, ni)")
        print(f"t_evo.shape = {t_time.shape}; expected ({expected_n},)")
        ok_shape = (
            y_time.ndim == 3
            and t_time.ndim == 1
            and y_time.shape[0] == expected_n
            and t_time.shape[0] == expected_n
        )
        ok_finite = bool(np.isfinite(y_time).all() and np.isfinite(t_time).all())
        ok_monotonic = bool(np.all(np.diff(t_time) > 0))
        print(f"shape={ok_shape}, finite={ok_finite}, monotonic={ok_monotonic}")

        # Round-trip check via legacy_io.save_out + pickle load.
        dname = str(ROOT)
        original_out_name = vulcan_cfg.out_name
        vulcan_cfg.out_name = "test_save_evolution.vul"
        try:
            output.save_out(rs_out, dname)
            output_file = str(ROOT) + "/" + vulcan_cfg.output_dir + vulcan_cfg.out_name
            with open(output_file, "rb") as f:
                payload = pickle.load(f)
            ok_keys = (
                "y_time" in payload["variable"] and "t_time" in payload["variable"]
            )
            y_round = np.asarray(payload["variable"]["y_time"])
            t_round = np.asarray(payload["variable"]["t_time"])
            ok_round = (
                y_round.shape == y_time.shape
                and np.allclose(y_round, y_time)
                and np.allclose(t_round, t_time)
            )
            print(f"pickle keys ok = {ok_keys}; round-trip ok = {ok_round}")
            os.remove(output_file)
        finally:
            vulcan_cfg.out_name = original_out_name

        ok = ok_shape and ok_finite and ok_monotonic and ok_keys and ok_round
    finally:
        vulcan_cfg.save_evolution = original_save_evo
        vulcan_cfg.save_evo_frq = original_frq
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
