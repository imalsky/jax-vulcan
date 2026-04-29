"""Chunked runner equivalence with single-shot.

When `use_print_prog` / `use_live_plot` / `use_live_flux` /
`use_save_movie` is on, `OuterLoop.__call__` runs the integration in
`print_prog_num`-sized chunks via `_run_chunked` so the host can print /
plot / save a movie frame between chunks. The chunked path must produce
the same final state as the single-shot path — the cap on accepted
steps per chunk should not change physics.

This test:
  1. Runs HD189 for `count_max=30` accepted steps in single-shot mode.
  2. Resets `OuterLoop` and runs again with `use_print_prog=True,
     print_prog_num=10` so the chunked driver fires three times.
  3. Asserts `var.y`, `var.t`, `var.dt`, `var.longdy` are bit-equivalent.

Standalone — no `../VULCAN-master/` oracle needed.
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
    import outer_loop
    import legacy_io
    vulcan_cfg = legacy_io.vulcan_cfg
    sys.modules["vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg
    vulcan_cfg.count_max = 30
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.use_save_movie = False
    vulcan_cfg.use_flux_movie = False
    vulcan_cfg.use_plot_end = False
    vulcan_cfg.use_plot_evo = False
    vulcan_cfg.save_evolution = False

    import op_jax
    from state import RunState, legacy_view
    op = legacy_io

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    data_para.start_time = time.time()
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    return solver, output, data_var, data_atm, data_para, outer_loop


def _run(use_chunked: bool):
    import legacy_io
    vulcan_cfg = legacy_io.vulcan_cfg
    vulcan_cfg.use_chunked_runner = bool(use_chunked)
    vulcan_cfg.use_print_prog = bool(use_chunked)
    vulcan_cfg.print_prog_num = 10  # 30/10 → 3 chunks (or 1 single-shot)
    solver, output, var, atm, para, outer_loop = _setup_state()
    integ = outer_loop.OuterLoop(solver, output)
    integ(var, atm, para, None)
    return var, atm, para


def main() -> int:
    # Stash original module-cache state so we don't pollute follow-on
    # tests that do plain `import vulcan_cfg` and rely on module identity.
    original_cfg = sys.modules.get("vulcan_cfg")
    try:
        # Single-shot reference run.
        var_ss, _, para_ss = _run(use_chunked=False)
        y_ss = np.asarray(var_ss.y, dtype=np.float64)
        t_ss = float(var_ss.t)
        dt_ss = float(var_ss.dt)
        longdy_ss = float(var_ss.longdy)

        # Chunked run.
        var_ch, _, para_ch = _run(use_chunked=True)
        y_ch = np.asarray(var_ch.y, dtype=np.float64)
        t_ch = float(var_ch.t)
        dt_ch = float(var_ch.dt)
        longdy_ch = float(var_ch.longdy)

        y_diff = np.max(np.abs(y_ss - y_ch) / np.maximum(np.abs(y_ss), 1e-300))
        t_match = abs(t_ss - t_ch) / max(abs(t_ss), 1e-300) < 1e-15
        dt_match = abs(dt_ss - dt_ch) / max(abs(dt_ss), 1e-300) < 1e-15
        count_match = int(para_ss.count) == int(para_ch.count)
        longdy_match = abs(longdy_ss - longdy_ch) / max(abs(longdy_ss), 1e-300) < 1e-12

        print(f"y max relerr  = {y_diff:.3e}")
        print(f"t single-shot = {t_ss:.6e}; chunked = {t_ch:.6e}")
        print(f"dt single-shot = {dt_ss:.6e}; chunked = {dt_ch:.6e}")
        print(f"longdy single-shot = {longdy_ss:.3e}; chunked = {longdy_ch:.3e}")
        print(f"count single-shot = {para_ss.count}; chunked = {para_ch.count}")
        ok = (
            y_diff < 1e-12
            and t_match
            and dt_match
            and count_match
            and longdy_match
        )
    finally:
        # Reset all the test-mutated cfg flags so follow-on tests aren't
        # surprised by a leftover use_print_prog=True or count_max=30.
        # Restore vulcan_cfg.py's defaults; tests that depend on these
        # values reset them in their own _setup_state, but leaving stale
        # values can confuse cross-test module-aliased reads.
        import legacy_io
        cfg = legacy_io.vulcan_cfg
        cfg.use_chunked_runner = False
        cfg.use_print_prog = False
        cfg.print_prog_num = 500
        cfg.count_max = int(3e4)
        cfg.count_min = 120
        # Restore the original sys.modules entry so plain
        # `import vulcan_cfg` in subsequent tests gets the same module
        # they were getting before this test ran.
        if original_cfg is not None:
            sys.modules["vulcan_cfg"] = original_cfg
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
