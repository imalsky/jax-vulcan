"""Phase 14 — `use_fix_H2He` Hycean-world bottom pin (op.py:2938-2944).

When `vulcan_cfg.use_fix_H2He=True`, master snapshots the bottom-layer
mixing ratios of H2 and He at the first per-step iteration with
`var.t > 1e6` and pins those values via `vulcan_cfg.use_fix_sp_bot`
thereafter. The JAX port replicates the snapshot+pin inside
`outer_loop._make_runner`'s body via a one-shot `h2he_pinned` carry
flag.

This test:
  1. Seeds `var.t = 1.5e6` so the trip fires on the first accepted step.
  2. Captures `var.ymix[0, H2]` / `var.ymix[0, He]` BEFORE the run.
  3. Runs a 5-step HD189 integration with `use_fix_H2He=True`.
  4. Asserts:
       a. `var.y[0, H2_idx]` / `var.y[0, He_idx]` after the run equal
          `pre_ymix[0, sp] * atm.n_0[0]` to machine precision.
       b. `vulcan_cfg.use_fix_sp_bot` post-run contains the snapshotted
          values (master mutates the dict — JAX writes them back via
          `_unpack_state`).

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
    import vulcan_cfg
    vulcan_cfg.count_max = 5
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
    import vulcan_cfg
    import chem_funs
    species = list(chem_funs.spec_list)
    if "H2" not in species or "He" not in species:
        print("SKIP: H2 or He not in network — use_fix_H2He requires both.")
        return 0

    original_h2he = getattr(vulcan_cfg, "use_fix_H2He", False)
    original_fix_sp_bot = dict(getattr(vulcan_cfg, "use_fix_sp_bot", {}) or {})
    vulcan_cfg.use_fix_H2He = True
    # Make sure use_fix_sp_bot starts empty so we can detect master-style
    # post-run mutation.
    vulcan_cfg.use_fix_sp_bot = {}
    try:
        solver, output, var, atm, para, outer_loop = _setup_state()
        # Seed t past 1e6 so the trip fires on the first accepted step.
        var.t = 1.5e6
        h2_idx = species.index("H2")
        he_idx = species.index("He")
        h2_mix_pre = float(var.ymix[0, h2_idx])
        he_mix_pre = float(var.ymix[0, he_idx])
        n0_bot = float(atm.n_0[0])
        h2_target = h2_mix_pre * n0_bot
        he_target = he_mix_pre * n0_bot

        integ = outer_loop.OuterLoop(solver, output)
        integ(var, atm, para, None)

        h2_post = float(np.asarray(var.y[0, h2_idx]))
        he_post = float(np.asarray(var.y[0, he_idx]))
        h2_relerr = abs(h2_post - h2_target) / max(abs(h2_target), 1e-300)
        he_relerr = abs(he_post - he_target) / max(abs(he_target), 1e-300)
        print(f"H2 pre-pin ymix = {h2_mix_pre:.6e}; "
              f"target n = {h2_target:.6e}; post-run y[0] = {h2_post:.6e}; "
              f"relerr = {h2_relerr:.3e}")
        print(f"He pre-pin ymix = {he_mix_pre:.6e}; "
              f"target n = {he_target:.6e}; post-run y[0] = {he_post:.6e}; "
              f"relerr = {he_relerr:.3e}")
        ok_pin = (h2_relerr < 1e-12) and (he_relerr < 1e-12)

        # Master mutates vulcan_cfg.use_fix_sp_bot in-place at the moment
        # of pinning (op.py:2939-2944). We mirror that in _unpack_state so
        # downstream tooling sees the same state.
        post_dict = getattr(vulcan_cfg, "use_fix_sp_bot", {}) or {}
        ok_dict = ("H2" in post_dict and "He" in post_dict
                   and abs(post_dict["H2"] - h2_mix_pre) < 1e-12
                   and abs(post_dict["He"] - he_mix_pre) < 1e-12)
        print(f"vulcan_cfg.use_fix_sp_bot post-run: {dict(post_dict)}; "
              f"ok={ok_dict}")
        ok = ok_pin and ok_dict
    finally:
        vulcan_cfg.use_fix_H2He = original_h2he
        vulcan_cfg.use_fix_sp_bot = original_fix_sp_bot
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
