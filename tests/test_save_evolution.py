"""Phase 15 — `save_evolution` time-series capture in the OuterLoop runner.

When `vulcan_cfg.save_evolution=True`, master appends `var.y` and `var.t`
to `var.y_time` / `var.t_time` every accepted step (op.py:1099-1102) and
slices by `save_evo_frq` at save time. The JAX OuterLoop captures the
trajectory directly at the configured cadence into a fixed-size buffer.

This test runs HD189 with `save_evolution=True, save_evo_frq=10` for
`count_max=50` accepted steps and asserts:
  1. `var.t_time` has the expected length: `count_max // save_evo_frq` (5).
  2. `var.t_time` is monotonic.
  3. `var.y_time[i]` snapshot for each i ≥ 1 is a valid (nz, ni) array
     with no NaN / inf.
  4. The pickle save round-trips the time-series under the same key names
     as master (loadable by `plot_py/plot_evolution.py`).

Standalone — no `../VULCAN-master/` oracle needed.
"""
from __future__ import annotations

import os
import pickle
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
    return solver, output, data_var, data_atm, data_para, outer_loop


def main() -> int:
    # Re-anchor cwd at every test entry — earlier tests in the suite may
    # have chdir'd elsewhere, breaking the save_out's relative output_dir.
    os.chdir(ROOT)
    # Earlier tests (e.g. test_chem.py) `sys.modules.pop("vulcan_cfg")`
    # which forks fresh module instances on subsequent imports. legacy_io
    # and outer_loop may now hold *different* vulcan_cfg objects from the
    # test's `import vulcan_cfg`. Use legacy_io's reference so save_out
    # reads the values we set; sync outer_loop's by direct assignment.
    import outer_loop
    import legacy_io
    vulcan_cfg = legacy_io.vulcan_cfg
    sys.modules["vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg
    save_evo_frq = 10
    count_max = 50
    expected_n = count_max // save_evo_frq  # 5

    original_save_evo = getattr(vulcan_cfg, "save_evolution", False)
    original_frq = getattr(vulcan_cfg, "save_evo_frq", 1)
    original_plot_end = getattr(vulcan_cfg, "use_plot_end", False)
    original_plot_evo = getattr(vulcan_cfg, "use_plot_evo", False)
    vulcan_cfg.save_evolution = True
    vulcan_cfg.save_evo_frq = save_evo_frq
    # Some upstream tests in the suite flip these on and don't restore.
    # Force them off so legacy_io.save_out doesn't raise.
    vulcan_cfg.use_plot_end = False
    vulcan_cfg.use_plot_evo = False
    try:
        solver, output, var, atm, para, outer_loop = _setup_state()
        integ = outer_loop.OuterLoop(solver, output)
        integ(var, atm, para, None)

        y_time = np.asarray(var.y_time)
        t_time = np.asarray(var.t_time)
        print(f"y_time.shape = {y_time.shape}; expected ({expected_n}, nz, ni)")
        print(f"t_time.shape = {t_time.shape}; expected ({expected_n},)")
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
            output.save_out(var, atm, para, dname)
            output_file = (
                str(ROOT) + "/" + vulcan_cfg.output_dir
                + vulcan_cfg.out_name
            )
            with open(output_file, "rb") as f:
                payload = pickle.load(f)
            ok_keys = (
                "y_time" in payload["variable"]
                and "t_time" in payload["variable"]
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
        vulcan_cfg.use_plot_end = original_plot_end
        vulcan_cfg.use_plot_evo = original_plot_evo
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
