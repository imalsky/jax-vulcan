"""Benchmark VULCAN-JAX vs VULCAN-master per-step performance.

Measures the time for one warm Ros2 step on the same HD189 state.
Run from VULCAN-JAX/:
    python benchmarks/bench_step.py
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
sys.path.append(str(ROOT.parent / "VULCAN-master"))
warnings.filterwarnings("ignore")


def main():
    import vulcan_cfg
    import store
    import build_atm
    import op
    import chem_funs
    import op_jax
    import jax_step as js_mod
    import chem as chem_mod
    import jax.numpy as jnp

    # Build the canonical HD189 state once
    print("Setting up HD189 atmosphere...")
    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    ini = build_atm.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)
    make_atm.BC_flux(data_atm)
    data_var.dt = 1e-10
    data_var.ymix = data_var.y / np.vstack(np.sum(data_var.y, axis=1))

    nz, ni = data_var.y.shape
    print(f"  nz={nz}, ni={ni}")

    # === VULCAN-master Ros2 ===
    print("\nBenchmarking VULCAN-master Ros2.solver...")
    para_master = store.Parameters()
    solver_master = op.Ros2()
    solver_master.naming_solver(para_master)

    n_warmup = 3
    n_iter = 20

    for _ in range(n_warmup):
        var_temp = store.Variables()
        var_temp.y = data_var.y.copy()
        var_temp.ymix = data_var.ymix.copy()
        var_temp.dt = 1e-10
        var_temp.k = data_var.k
        para_temp = store.Parameters()
        para_temp.solver_str = "solver"
        solver_master.solver(var_temp, data_atm, para_temp)

    t0 = time.time()
    for _ in range(n_iter):
        var_temp = store.Variables()
        var_temp.y = data_var.y.copy()
        var_temp.ymix = data_var.ymix.copy()
        var_temp.dt = 1e-10
        var_temp.k = data_var.k
        para_temp = store.Parameters()
        para_temp.solver_str = "solver"
        solver_master.solver(var_temp, data_atm, para_temp)
    t_master = (time.time() - t0) / n_iter * 1000

    # === VULCAN-JAX Ros2JAX ===
    print("Benchmarking VULCAN-JAX Ros2JAX.solver...")
    solver_jax = op_jax.Ros2JAX()
    para_jax = store.Parameters()
    solver_jax.naming_solver(para_jax)

    for _ in range(n_warmup):
        var_temp = store.Variables()
        var_temp.y = data_var.y.copy()
        var_temp.ymix = data_var.ymix.copy()
        var_temp.dt = 1e-10
        var_temp.k = data_var.k
        para_temp = store.Parameters()
        para_temp.solver_str = "solver"
        solver_jax.solver(var_temp, data_atm, para_temp)

    t0 = time.time()
    for _ in range(n_iter):
        var_temp = store.Variables()
        var_temp.y = data_var.y.copy()
        var_temp.ymix = data_var.ymix.copy()
        var_temp.dt = 1e-10
        var_temp.k = data_var.k
        para_temp = store.Parameters()
        para_temp.solver_str = "solver"
        solver_jax.solver(var_temp, data_atm, para_temp)
    t_jax = (time.time() - t0) / n_iter * 1000

    # === Pure JAX kernel (no overhead) ===
    print("Benchmarking pure-JAX jax_ros2_step (no overhead)...")
    atm_static = js_mod.make_atm_static(data_atm, ni, nz)
    k_arr = np.zeros((1192 + 1, nz), dtype=np.float64)
    for i, vec in data_var.k.items():
        if 1 <= i <= 1192:
            k_arr[i] = np.asarray(vec, dtype=np.float64)
    y_j = jnp.asarray(data_var.y)
    k_j = jnp.asarray(k_arr)

    for _ in range(n_warmup):
        sol, delta = js_mod.jax_ros2_step(y_j, k_j, 1e-10, atm_static, op_jax._NET_JAX)
        sol.block_until_ready()

    t0 = time.time()
    for _ in range(n_iter):
        sol, delta = js_mod.jax_ros2_step(y_j, k_j, 1e-10, atm_static, op_jax._NET_JAX)
        sol.block_until_ready()
    t_pure = (time.time() - t0) / n_iter * 1000

    # === Report ===
    print()
    print(f"{'Solver':<35} {'Time/step (ms)':>15} {'Relative':>10}")
    print("-" * 62)
    print(f"{'VULCAN-master Ros2.solver':<35} {t_master:>15.1f} {1.0:>10.2f}x")
    print(f"{'VULCAN-JAX Ros2JAX.solver':<35} {t_jax:>15.1f} {t_jax/t_master:>10.2f}x")
    print(f"{'Pure JAX jax_ros2_step (kernel)':<35} {t_pure:>15.1f} {t_pure/t_master:>10.2f}x")

    print()
    print(f"Speedup vs VULCAN-master: {t_master/t_jax:.2f}x ({(1-t_jax/t_master)*100:.1f}% faster)")


if __name__ == "__main__":
    main()
