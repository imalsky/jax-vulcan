"""save_evolution time series from the OuterLoop runner.

Master appends y/t every accepted step (op.py:1096-1099) and keeps every
save_evo_frq-th. HD189 with count_max=50 and save_evo_frq=10 must give 6
finite snapshots at increasing t, and the .vul must round-trip them under
master's y_time/t_time keys (plot_py/plot_evolution.py).
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def main() -> int:
    # Re-anchor cwd at every test entry: earlier tests in the suite may
    # have chdir'd elsewhere, breaking the save_out's relative output_dir.
    os.chdir(ROOT)
    import vulcan_jax.legacy_io as legacy_io
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    vulcan_cfg = legacy_io.default_config()
    save_evo_frq = 10
    count_max = 50
    # Master saves count_max + 1 steps (op.py:1080 uses `>`) and keeps y_time[::save_evo_frq].
    expected_n = (count_max + save_evo_frq) // save_evo_frq  # 6

    vulcan_cfg.save_evolution = True
    vulcan_cfg.save_evo_frq = save_evo_frq
    vulcan_cfg.count_max = count_max
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
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
    vulcan_cfg.out_name = "test_save_evolution.vul"
    output.save_out(rs_out, str(ROOT))
    output_file = str(ROOT) + "/" + vulcan_cfg.output_dir + vulcan_cfg.out_name
    with open(output_file, "rb") as f:
        payload = pickle.load(f)
    ok_keys = "y_time" in payload["variable"] and "t_time" in payload["variable"]
    y_round = np.asarray(payload["variable"]["y_time"])
    t_round = np.asarray(payload["variable"]["t_time"])
    ok_round = (
        y_round.shape == y_time.shape
        and np.allclose(y_round, y_time)
        and np.allclose(t_round, t_time)
    )
    print(f"pickle keys ok = {ok_keys}; round-trip ok = {ok_round}")
    os.remove(output_file)

    ok = ok_shape and ok_finite and ok_monotonic and ok_keys and ok_round
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0
