"""Benchmark the current VULCAN-JAX step and runner paths.

Reports three timings:
  1. Upstream `VULCAN-master` Ros2 single-step wall time (isolated subprocess).
  2. Local pure-JAX `jax_ros2_step` kernel wall time.
  3. Local `OuterLoop` 50-step smoke wall time per accepted step.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT.parent / "VULCAN-master"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def _setup_hd189_state():
    import build_atm
    import chem
    import legacy_io as op
    import network
    import op_jax
    import store
    import vulcan_cfg

    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    if vulcan_cfg.use_lowT_limit_rates:
        data_var = rate.lim_lowT_rates(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    data_var = rate.remove_rate(data_var)
    ini = build_atm.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)

    if vulcan_cfg.use_photo:
        rate.make_bins_read_cross(data_var, data_atm)
        make_atm.read_sflux(data_var, data_atm)
        solver = op_jax.Ros2JAX()
        solver.compute_tau(data_var, data_atm)
        solver.compute_flux(data_var, data_atm)
        solver.compute_J(data_var, data_atm)
        if vulcan_cfg.use_ion:
            solver.compute_Jion(data_var, data_atm)
        data_var = rate.remove_rate(data_var)

    net = chem.to_jax(network.parse_network(vulcan_cfg.network))
    return data_var, data_atm, make_atm, net


def _pack_k_arr(k_dict, nr: int, nz: int) -> np.ndarray:
    out = np.zeros((nr + 1, nz), dtype=np.float64)
    for ridx, vec in k_dict.items():
        if 1 <= ridx <= nr:
            out[ridx] = np.asarray(vec, dtype=np.float64)
    return out


def _bench_master_single_step() -> float | None:
    if not MASTER.is_dir():
        return None

    script = r"""
import os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
import store, build_atm, op, vulcan_cfg
data_var = store.Variables()
data_atm = store.AtmData()
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if vulcan_cfg.use_condense:
    make_atm.sp_sat(data_atm)
rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)
if vulcan_cfg.use_lowT_limit_rates:
    data_var = rate.lim_lowT_rates(data_var, data_atm)
data_var = rate.rev_rate(data_var, data_atm)
data_var = rate.remove_rate(data_var)
ini = build_atm.InitialAbun()
data_var = ini.ini_y(data_var, data_atm)
data_var = ini.ele_sum(data_var)
data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)
data_var.dt = 1e-10
data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))
solver = op.Ros2()
para = store.Parameters()
solver.naming_solver(para)
for _ in range(3):
    var_tmp = store.Variables()
    var_tmp.y = data_var.y.copy()
    var_tmp.ymix = data_var.ymix.copy()
    var_tmp.dt = data_var.dt
    var_tmp.k = data_var.k
    para_tmp = store.Parameters()
    para_tmp.solver_str = "solver"
    solver.solver(var_tmp, data_atm, para_tmp)
t0 = time.time()
for _ in range(20):
    var_tmp = store.Variables()
    var_tmp.y = data_var.y.copy()
    var_tmp.ymix = data_var.ymix.copy()
    var_tmp.dt = data_var.dt
    var_tmp.k = data_var.k
    para_tmp = store.Parameters()
    para_tmp.solver_str = "solver"
    solver.solver(var_tmp, data_atm, para_tmp)
print((time.time() - t0) / 20.0 * 1000.0)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=MASTER,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip().splitlines()[-1])


def _bench_jax_kernel(data_var, data_atm, net) -> float:
    import jax
    import jax.numpy as jnp
    from jax_step import jax_ros2_step, make_atm_static

    nz, ni = data_var.y.shape
    k_arr = _pack_k_arr(data_var.k, net.nr, nz)
    atm_static = make_atm_static(data_atm, ni, nz)
    y_j = jnp.asarray(data_var.y)
    k_j = jnp.asarray(k_arr)
    dt = jnp.float64(1e-10)

    for _ in range(3):
        sol, _ = jax_ros2_step(y_j, k_j, dt, atm_static, net)
        sol.block_until_ready()

    t0 = time.time()
    for _ in range(20):
        sol, _ = jax_ros2_step(y_j, k_j, dt, atm_static, net)
        sol.block_until_ready()
    return (time.time() - t0) / 20.0 * 1000.0


def _bench_outer_loop_50(data_var, data_atm, make_atm) -> float:
    import legacy_io as op
    import op_jax
    import outer_loop
    import store
    import vulcan_cfg

    old_flags = {
        "use_live_plot": vulcan_cfg.use_live_plot,
        "use_live_flux": vulcan_cfg.use_live_flux,
        "count_max": vulcan_cfg.count_max,
        "runtime": vulcan_cfg.runtime,
    }
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.count_max = 50
    vulcan_cfg.runtime = 1e22

    solver = op_jax.Ros2JAX()
    output = op.Output()
    integ = outer_loop.OuterLoop(solver, output)
    para = store.Parameters()

    t0 = time.time()
    try:
        integ(data_var, data_atm, para, make_atm)
    finally:
        vulcan_cfg.use_live_plot = old_flags["use_live_plot"]
        vulcan_cfg.use_live_flux = old_flags["use_live_flux"]
        vulcan_cfg.count_max = old_flags["count_max"]
        vulcan_cfg.runtime = old_flags["runtime"]
        integ.reset()

    return (time.time() - t0) / max(para.count, 1) * 1000.0


def main() -> None:
    data_var, data_atm, make_atm, net = _setup_hd189_state()
    master_ms = _bench_master_single_step()
    kernel_ms = _bench_jax_kernel(data_var, data_atm, net)
    outer_ms = _bench_outer_loop_50(data_var, data_atm, make_atm)

    print(f"master_ros2_step_ms={master_ms:.3f}" if master_ms is not None else "master_ros2_step_ms=unavailable")
    print(f"jax_ros2_step_ms={kernel_ms:.3f}")
    print(f"outer_loop_50step_ms_per_step={outer_ms:.3f}")


if __name__ == "__main__":
    main()
