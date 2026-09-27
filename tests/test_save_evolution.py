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
from _helpers import set_cfg

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def test_save_evolution_snapshots_and_vul_round_trip():
    import vulcan_jax.legacy_io as legacy_io
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    save_evo_frq = 10
    count_max = 50
    # Master saves count_max + 1 steps (op.py:1080 uses `>`) and keeps y_time[::save_evo_frq].
    expected_n = (count_max + save_evo_frq) // save_evo_frq  # 6

    vulcan_cfg = set_cfg(
        save_evolution=True,
        save_evo_frq=save_evo_frq,
        count_max=count_max,
        count_min=1,
        use_print_prog=False,
    )
    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    output = legacy_io.Output()
    rs_out = outer_loop.OuterLoop(op_jax.Ros2JAX(), output)(rs)

    y_time = np.asarray(rs_out.step.y_evo)
    t_time = np.asarray(rs_out.step.t_evo)
    assert y_time.ndim == 3 and y_time.shape[0] == expected_n, y_time.shape
    assert t_time.shape == (expected_n,), t_time.shape
    assert np.isfinite(y_time).all() and np.isfinite(t_time).all()
    assert np.all(np.diff(t_time) > 0), t_time

    # Round trip through legacy_io.save_out and a pickle load.
    vulcan_cfg.out_name = "test_save_evolution.vul"
    output_file = ROOT / vulcan_cfg.output_dir / vulcan_cfg.out_name
    try:
        output.save_out(rs_out, str(ROOT))
        with open(output_file, "rb") as f:
            payload = pickle.load(f)
    finally:
        output_file.unlink(missing_ok=True)
    y_round = np.asarray(payload["variable"]["y_time"])
    t_round = np.asarray(payload["variable"]["t_time"])
    assert y_round.shape == y_time.shape
    np.testing.assert_allclose(y_round, y_time, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(t_round, t_time, rtol=1e-5, atol=1e-8)
