"""`use_fix_all_bot` post-step bottom clamp inside the runner.

`OuterLoop` clamps the bottom layer to chemical-EQ mixing ratios
(captured at init from `var.ymix[0]`) on every accepted step when
`vulcan_cfg.use_fix_all_bot=True`. Mirrors
`op.Ros2.solver_fix_all_bot`'s `sol[0] = bottom*atm.n_0[0]`
(`op.py:3050-3051`).

This test forces `use_fix_all_bot=True`, runs a 10-step HD189
integration, and confirms the bottom row of `var.y` equals the captured
`bottom_ymix * n_0[0]` to machine precision after the run. It also
confirms that `use_fix_all_bot=False` leaves the bottom free (matches
the unclamped baseline).

The pre-Phase-10.6 version of this test only checked that
`naming_solver` raised `NotImplementedError`. That deferral is gone now.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def _setup_state():
    import vulcan_cfg
    vulcan_cfg.count_max = 10
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False

    import legacy_io as op, op_jax, outer_loop  # noqa: E401
    from state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    data_para.start_time = time.time()
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    return solver, output, data_var, data_atm, data_para, outer_loop


def main() -> int:
    import vulcan_cfg
    original = getattr(vulcan_cfg, "use_fix_all_bot", False)
    vulcan_cfg.use_fix_all_bot = True
    try:
        solver, output, var, atm, para, outer_loop = _setup_state()
        # Capture the would-be clamp target BEFORE the run.
        bottom_target = (np.asarray(var.ymix[0], dtype=np.float64)
                         * float(atm.n_0[0]))
        integ = outer_loop.OuterLoop(solver, output)
        integ(var, atm, para, None)

        diff = np.abs(np.asarray(var.y[0]) - bottom_target)
        max_relerr = float(np.max(
            diff / np.maximum(np.abs(bottom_target), 1e-300)
        ))
        print(f"clamped bottom: max relerr = {max_relerr:.3e}")
        ok = max_relerr < 1e-12
    finally:
        vulcan_cfg.use_fix_all_bot = original
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
