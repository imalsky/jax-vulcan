"""Validate pure-JAX integration loop against op_jax.Ros2JAX.

Compares 10 fixed-dt steps via the JIT'd `jax_integrate_fixed_dt` against
sequential calls to `op_jax.Ros2JAX.solver`. They should agree.
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


def main() -> int:
    import jax.numpy as jnp
    import vulcan_cfg
    import store
    import build_atm
    import op
    import op_jax
    import outer_loop  # _NET_JAX moved here in Phase 10.1
    import jax_step as js_mod
    import integrate as int_mod

    # Build canonical state
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
    nz, ni = data_var.y.shape
    print(f"State: nz={nz}, ni={ni}")

    atm_static = js_mod.make_atm_static(data_atm, ni, nz)
    k_arr = np.zeros((1192 + 1, nz), dtype=np.float64)
    for i, vec in data_var.k.items():
        if 1 <= i <= 1192:
            k_arr[i] = np.asarray(vec, dtype=np.float64)

    dt = 1e-10
    N = 10

    # === Path A: Pure JAX integration (lax.scan) ===
    print(f"\nPath A: jax_integrate_fixed_dt for N={N} steps...")
    t0 = time.time()
    y_jax_final, deltas = int_mod.jax_integrate_fixed_dt(
        jnp.asarray(data_var.y), jnp.asarray(k_arr), dt, n_steps=N, atm=atm_static, net=outer_loop._NET_JAX
    )
    y_jax_final.block_until_ready()
    t_jax = time.time() - t0
    print(f"  first call (compile + execute): {t_jax:.2f}s")

    t0 = time.time()
    y_jax_final, deltas = int_mod.jax_integrate_fixed_dt(
        jnp.asarray(data_var.y), jnp.asarray(k_arr), dt, n_steps=N, atm=atm_static, net=outer_loop._NET_JAX
    )
    y_jax_final.block_until_ready()
    t_jax_warm = time.time() - t0
    print(f"  warm call: {t_jax_warm:.2f}s ({t_jax_warm/N*1000:.1f}ms/step)")

    # === Path B: jax_ros2_step in a Python loop with same fixed dt ===
    # (Pre-Phase-10.1 used op_jax.Ros2JAX.solver here; that method moved
    # into outer_loop's body_fn. Calling jax_ros2_step directly is the
    # equivalent reference — same kernel, just driven from Python.)
    print(f"\nPath B: jax_ros2_step Python loop for N={N} steps (fixed dt)...")
    y_loop = jnp.asarray(data_var.y)
    k_arr_j = jnp.asarray(k_arr)
    t0 = time.time()
    for _ in range(N):
        y_loop, _delta = js_mod.jax_ros2_step(y_loop, k_arr_j, dt, atm_static, outer_loop._NET_JAX)
    y_loop.block_until_ready()
    t_py = time.time() - t0
    print(f"  python loop: {t_py:.2f}s ({t_py/N*1000:.1f}ms/step)")
    y_py_final = np.asarray(y_loop)

    # === Compare ===
    y_jax_np = np.asarray(y_jax_final)
    relerr = np.abs(y_jax_np - y_py_final) / np.maximum(np.abs(y_py_final), 1e-30)
    max_relerr = relerr.max()
    print(f"\nMax relerr (Path A vs Path B): {max_relerr:.3e}")
    print(f"Speedup of jax_integrate vs python loop: {t_py/t_jax_warm:.2f}x")

    # === Mixing ratio comparison (more meaningful) ===
    ymix_jax = y_jax_np / np.sum(y_jax_np, axis=1, keepdims=True)
    ymix_py = y_py_final / np.sum(y_py_final, axis=1, keepdims=True)
    relerr_ymix = np.abs(ymix_jax - ymix_py) / np.maximum(np.abs(ymix_py), 1e-30)
    print(f"Max ymix relerr: {relerr_ymix.max():.3e}")

    # === Test adaptive version ===
    print(f"\nPath C: jax_integrate_adaptive (max_steps=20, rtol=0.25)...")
    t0 = time.time()
    y_adapt, dt_final, t_final, count_final = int_mod.jax_integrate_adaptive(
        jnp.asarray(data_var.y), jnp.asarray(k_arr),
        dt0=1e-10, max_steps=20, rtol=0.25,
        dt_min=1e-14, dt_max=1e10, atm=atm_static, net=outer_loop._NET_JAX
    )
    y_adapt.block_until_ready()
    t_adapt = time.time() - t0
    print(f"  20 adaptive steps: {t_adapt:.2f}s ({t_adapt/20*1000:.1f}ms/step)")
    print(f"  final dt={float(dt_final):.3e}, t={float(t_final):.3e}, count={int(count_final)}")

    print()
    ok = max_relerr < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
