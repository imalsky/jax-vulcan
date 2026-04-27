"""Validate Phase 10.5 — the JAX runner is one-shot and the in-runner
convergence check terminates the integration without Python-side polling.

Three assertions:

  1. **Ring buffer + chronology**: a count_max=50 HD189 run produces
     `var.y_time` and `var.t_time` of length min(count, conv_step) = 51
     (the runner exits when `accept_count > count_max`, so the final
     accept_count is 51), in strictly increasing time order. The first
     entry is the post-step-1 state (not the pre-loop initial state —
     matches `op.save_step` semantics, which appends AFTER the accepted
     step). The last entry equals `var.y` / `var.t`.

  2. **longdy / longdydt populated**: after the runner returns,
     `var.longdy` and `var.longdydt` are finite and positive (the
     in-runner conv check ran and updated them at every accepted step).

  3. **Single-shot termination via count_max**: with count_max=50, the
     runner exits exactly when `accept_count > count_max`. para.count
     becomes 51 (50 accepted body iterations + the off-by-one
     terminating attempt that triggers `>`). para.end_case = 3
     ("Maximal allowed steps exceeded").

The ring buffer is sized at `vulcan_cfg.conv_step` (500 by default), so
50 < 500 and we get the full trajectory. For longer runs the ring
overwrites; that trade-off is documented in `_unpack_ring`.
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


def main() -> int:
    import vulcan_cfg
    vulcan_cfg.count_max = 50
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False

    import store, build_atm, legacy_io as op, op_jax, outer_loop  # noqa: E401

    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()
    data_para.start_time = time.time()
    make_atm = build_atm.Atm()
    output = op.Output()

    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    data_var = rate.remove_rate(data_var)
    ini_abun = build_atm.InitialAbun()
    data_var = ini_abun.ini_y(data_var, data_atm)
    data_var = ini_abun.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    solver = op_jax.Ros2JAX()
    if vulcan_cfg.use_photo:
        rate.make_bins_read_cross(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)
        solver.compute_tau(data_var, data_atm)
        solver.compute_flux(data_var, data_atm)
        solver.compute_J(data_var, data_atm)
        data_var = rate.remove_rate(data_var)

    integ = outer_loop.OuterLoop(solver, output)
    integ(data_var, data_atm, data_para, make_atm)

    ok = True

    # ---- 1. Ring buffer + chronological reconstruction ----
    n_kept = min(int(data_para.count), int(vulcan_cfg.conv_step))
    if len(data_var.y_time) != n_kept:
        print(f"FAIL: var.y_time has {len(data_var.y_time)} entries, "
              f"expected {n_kept}")
        ok = False
    if len(data_var.t_time) != n_kept:
        print(f"FAIL: var.t_time has {len(data_var.t_time)} entries, "
              f"expected {n_kept}")
        ok = False

    # Chronology: t_time should be strictly increasing.
    t_arr = np.asarray(data_var.t_time)
    if not np.all(np.diff(t_arr) > 0):
        print("FAIL: var.t_time not strictly increasing — ring "
              "reconstruction is in the wrong order")
        ok = False

    # Last entry == final var.t (most recent ring slot).
    if t_arr[-1] != data_var.t:
        print(f"FAIL: last var.t_time ({t_arr[-1]:.3e}) != var.t "
              f"({data_var.t:.3e})")
        ok = False

    print(f"ring chronology  OK ({n_kept} entries, t spans "
          f"{t_arr[0]:.3e}..{t_arr[-1]:.3e})")

    # ---- 2. longdy / longdydt populated ----
    if not (np.isfinite(data_var.longdy) and data_var.longdy > 0):
        print(f"FAIL: var.longdy = {data_var.longdy} is not finite/positive")
        ok = False
    if not (np.isfinite(data_var.longdydt) and data_var.longdydt > 0):
        print(f"FAIL: var.longdydt = {data_var.longdydt} is not finite/positive")
        ok = False
    print(f"longdy/longdydt  OK (longdy={data_var.longdy:.3e}, "
          f"longdydt={data_var.longdydt:.3e})")

    # ---- 3. Single-shot termination via count_max ----
    if data_para.count != vulcan_cfg.count_max + 1:
        print(f"FAIL: para.count={data_para.count}, expected "
              f"{vulcan_cfg.count_max + 1}")
        ok = False
    if data_para.end_case != 3:
        print(f"FAIL: para.end_case={data_para.end_case}, expected 3 "
              f"(count_max exceeded)")
        ok = False
    print(f"count_max exit   OK (count={data_para.count}, end_case="
          f"{data_para.end_case})")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
