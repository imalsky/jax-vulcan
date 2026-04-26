"""Profile one Ros2JAX step to find the bottleneck."""
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

    data_para = store.Parameters()
    solver = op_jax.Ros2JAX()
    solver.naming_solver(data_para)

    nz, ni = data_var.y.shape

    # === Profile each piece ===

    # Step 1: pack k_arr
    n_iter = 10
    t0 = time.time()
    for _ in range(n_iter):
        k_arr = np.zeros((1192 + 1, nz), dtype=np.float64)
        for i, vec in data_var.k.items():
            if 1 <= i <= 1192:
                k_arr[i] = np.asarray(vec, dtype=np.float64)
    t_pack = (time.time() - t0) / n_iter * 1000

    # Step 2: build atm_static
    t0 = time.time()
    for _ in range(n_iter):
        atm_static = js_mod.make_atm_static(data_atm, ni, nz)
    t_atm = (time.time() - t0) / n_iter * 1000

    # Step 3: jax_ros2_step (compiled + warm)
    # Warm up
    sol, delta = js_mod.jax_ros2_step(
        jnp.asarray(data_var.y), jnp.asarray(k_arr), float(data_var.dt),
        atm_static, op_jax._NET_JAX
    )
    sol.block_until_ready()
    t0 = time.time()
    for _ in range(n_iter):
        sol, delta = js_mod.jax_ros2_step(
            jnp.asarray(data_var.y), jnp.asarray(k_arr), float(data_var.dt),
            atm_static, op_jax._NET_JAX
        )
        sol.block_until_ready()
    t_step = (time.time() - t0) / n_iter * 1000

    # Step 4: full Ros2JAX.solver call
    t0 = time.time()
    for _ in range(n_iter):
        # Need to reset var.y between calls
        data_var2 = store.Variables()
        data_var2.y = data_var.y.copy()
        data_var2.ymix = data_var.ymix.copy()
        data_var2.dt = 1e-10
        data_var2.k = data_var.k
        para2 = store.Parameters()
        para2.solver_str = "solver"
        solver.solver(data_var2, data_atm, para2)
    t_solver = (time.time() - t0) / n_iter * 1000

    print(f"Per-call timings (avg over {n_iter} iters):")
    print(f"  Pack k_arr:        {t_pack:6.1f} ms")
    print(f"  Build atm_static:  {t_atm:6.1f} ms")
    print(f"  jax_ros2_step:     {t_step:6.1f} ms")
    print(f"  Full solver():     {t_solver:6.1f} ms")
    print(f"  Overhead:          {t_solver - t_step:6.1f} ms (= solver - jax_step)")

    # === Sub-profile inside jax_ros2_step ===
    import jax
    y_j = jnp.asarray(data_var.y)
    M_j = jnp.asarray(data_atm.M)
    k_j = jnp.asarray(k_arr)

    # chem_rhs alone
    chem_rhs_jit = jax.jit(chem_mod.chem_rhs)
    chem_rhs_jit(y_j, M_j, k_j, op_jax._NET_JAX).block_until_ready()
    t0 = time.time()
    for _ in range(n_iter):
        chem_rhs_jit(y_j, M_j, k_j, op_jax._NET_JAX).block_until_ready()
    t_rhs = (time.time() - t0) / n_iter * 1000

    # chem_jac alone
    chem_jac_jit = jax.jit(chem_mod.chem_jac)
    chem_jac_jit(y_j, M_j, k_j, op_jax._NET_JAX).block_until_ready()
    t0 = time.time()
    for _ in range(n_iter):
        chem_jac_jit(y_j, M_j, k_j, op_jax._NET_JAX).block_until_ready()
    t_jac = (time.time() - t0) / n_iter * 1000

    # block_thomas alone (with random LHS)
    import solver as solver_mod
    rng = np.random.default_rng(0)
    diag_test = jnp.asarray(rng.standard_normal((nz, ni, ni)) + 1e10 * np.eye(ni))
    sup_test = jnp.asarray(rng.standard_normal((nz - 1, ni, ni)) * 1e-3)
    sub_test = jnp.asarray(rng.standard_normal((nz - 1, ni, ni)) * 1e-3)
    rhs_test = jnp.asarray(rng.standard_normal((nz, ni)))
    bt_jit = jax.jit(solver_mod.block_thomas)
    bt_jit(diag_test, sup_test, sub_test, rhs_test).block_until_ready()
    t0 = time.time()
    for _ in range(n_iter):
        bt_jit(diag_test, sup_test, sub_test, rhs_test).block_until_ready()
    t_bt = (time.time() - t0) / n_iter * 1000

    print()
    print("Sub-pieces of jax_ros2_step:")
    print(f"  chem_rhs (jit):      {t_rhs:6.1f} ms")
    print(f"  chem_jac (jit):      {t_jac:6.1f} ms")
    print(f"  block_thomas (jit):  {t_bt:6.1f} ms")
    print(f"  --> jac is largest expected component")


if __name__ == "__main__":
    main()
